import os
from typing import Iterable, Dict, List, Any, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from src.quant_utils import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    dequantize_linear_weight,
    quantize,
)
import src.quant_utils as quant_utils
from src import dist_utils
from src.common_utils import to, maybe_first_element
from src.model_utils import (
    InputCollector,
    ForwardInterrupt,
    LINEAR_LAYERS,
    select_layers,
)
from src.gptq import GPTQ

class Quantizer:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        quantizer_kwargs: Dict[str, Any],
        pre_block_modules: List[str],
        post_block_modules: List[str],
        block_modules: str,
        save_dir: str,
        quant_non_block_modules: bool = False,
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        # Quantization params
        self.quantizable_modules = quantizable_modules
        self.quantizer_kwargs = quantizer_kwargs
        # Model params
        self.pre_block_modules = pre_block_modules
        self.post_block_modules = post_block_modules
        self.block_modules = block_modules
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.quant_non_block_modules = quant_non_block_modules
        self.verbose = verbose
        self.save_dir = save_dir

    @torch.no_grad()
    def quantize(self, quant_config: Dict[str, GGMLQuantizationType]) -> None:
        device = self.device or next(self.model.parameters()).device
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [
            (module_name, self._get_submodule(module_name)) for module_name in self.pre_block_modules
        ]
        post_blocks = [
            (module_name, self._get_submodule(module_name)) for module_name in self.post_block_modules
        ]
        blocks[0] = blocks[0].to(device)
        for (_, module) in pre_blocks:
            module.to(device)
        # Cache
        if hasattr(self.model.config, "use_cache"):
            use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        # Input preparation #
        blocks[0] = InputCollector(blocks[0], cpu_offload=self.cpu_offload_activations)
        # TODO make namedtuple
        for inp_args, inp_kwargs in self.data_loader:
            try:
                self.model(
                    *to(inp_args, device=device), **to(inp_kwargs, device=device)
                )
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        for i, (name, module) in enumerate(pre_blocks):
            if not self.quant_non_block_modules:
                continue
            # get layer prefix to select layers only within the block
            layer_prefix = f"{name}"
            if self.verbose:
                dist_utils.print_on_main(
                    f"Processing {layer_prefix}."
                )
            module = module.to(device)

            # get layer prefix to select layers only within the block
            q_type = quant_config.get(
                layer_prefix.split(".")[-1], GGMLQuantizationType.Q6_K
            )
            qweight, super_group_scale, group_scale_quant, super_group_zero, group_zero_quant = self._quant_non_block_module(module.weight, q_type)
            w_qweight = dequantize_linear_weight(
                q_type,
                qweight,
                super_group_scale,
                group_scale_quant,
                super_group_zero,
                group_zero_quant,
            )
            module.weight.data = w_qweight.to(module.weight.data.dtype)

            os.makedirs(os.path.join(self.save_dir, layer_prefix), exist_ok=True)
            torch.save({
                "q_type": q_type.value,
                "qweight": qweight.cpu(),
                "super_group_scale": super_group_scale.cpu(),
                "super_group_zero": super_group_zero.cpu(),
                "group_scale_quant": group_scale_quant.cpu(),
                "group_zero_quant": group_zero_quant.cpu(),
            }, os.path.join(self.save_dir, layer_prefix, f"data.pth"))
               

        # offload pre_blocks
        if self.cpu_offload_modules:
            for (_, module) in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):
            if self.verbose:
                dist_utils.print_on_main(
                    f"Processing {self.block_modules} {block_id}/{len(blocks)}."
                )
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(
                self.model, layer_prefix, self.quantizable_modules, LINEAR_LAYERS
            )
            handles, hooks = self._prepare_hooks_and_handles(layers)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._quant_group(handles, quant_config)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))  # me
                out = maybe_first_element(out)
                if self.cpu_offload_activations:
                    out = out.cpu()
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        for i, (name, module) in enumerate(post_blocks):
            if not self.quant_non_block_modules:
                continue
            # get layer prefix to select layers only within the block
            layer_prefix = f"{name}"
            if self.verbose:
                dist_utils.print_on_main(
                    f"Processing {layer_prefix}"
                )
            module = module.to(device)
            q_type = quant_config.get(
                layer_prefix.split(".")[-1], GGMLQuantizationType.Q6_K
            )

            qweight, super_group_scale, group_scale_quant, super_group_zero, group_zero_quant = self._quant_non_block_module(module.weight, q_type)
            w_qweight = dequantize_linear_weight(
                q_type,
                qweight,
                super_group_scale,
                group_scale_quant,
                super_group_zero,
                group_zero_quant,
            )
            module.weight.data = w_qweight.to(module.weight.data.dtype)
            
            os.makedirs(os.path.join(self.save_dir, layer_prefix), exist_ok=True)
            torch.save({
                "q_type": q_type.value,
                "qweight": qweight.cpu(),
                "super_group_scale": super_group_scale.cpu(),
                "super_group_zero": super_group_zero.cpu(),
                "group_scale_quant": group_scale_quant.cpu(),
                "group_zero_quant": group_zero_quant.cpu(),
            }, os.path.join(self.save_dir, layer_prefix, f"data.pth"))

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])

                return _hook

            handles[layer_name] = self._create_handle(layer)
            hooks[layer_name] = layer.register_forward_hook(
                update_handle_hook(layer_name)
            )
        return handles, hooks

    def _create_handle(self, layer):
        return GPTQ(layer, **self.quantizer_kwargs)

    def _quant_group(
        self,
        handles: Dict[str, Union[GPTQ]],
        quant_config: Dict[str, GGMLQuantizationType],
    ):
        # Create a dictionary to store the quantization parameters for each handle
        for handle_name, handle in handles.items():
            q_type = quant_config.get(
                handle_name.split(".")[-1], GGMLQuantizationType.Q4_K
            )

            if self.verbose:
                dist_utils.print_on_main(f"Quantizing {handle_name} with {q_type}.")

            qweight, super_group_scale, group_scale_quant, super_group_zero, group_zero_quant = handle.quantize(q_type)
            handle.layer.weight.data = dequantize_linear_weight(
                q_type,
                qweight,
                super_group_scale,
                group_scale_quant,
                super_group_zero,
                group_zero_quant,
            ).to(handle.layer.weight.data.dtype)
            handle.reset()

            os.makedirs(os.path.join(self.save_dir, handle_name), exist_ok=True)
            torch.save({
                "q_type": q_type.value,
                "qweight": qweight.cpu(),
                "super_group_scale": super_group_scale.cpu(),
                "super_group_zero": super_group_zero.cpu(),
                "group_scale_quant": group_scale_quant.cpu(),
                "group_zero_quant": group_zero_quant.cpu(),
            }, os.path.join(self.save_dir, handle_name, f"data.pth"))


    def _quant_non_block_module(self, w: torch.tensor, q_type: GGMLQuantizationType) -> None:
        d_row, d_col = w.shape

        # --- setup quantizer ---
        bits, clamp_min_max, scale_maxq, group_size, supergroup_size, scale_zero_dtype, qweight_dtype = GGML_QUANT_SIZES[q_type]

        # --- set up quantizer (you already did this) ---
        quantizer = quant_utils.Quantizer()
        quantizer.configure(
            bits=bits,
            scale_maxq=scale_maxq,
            super_group_size=supergroup_size,
            group_size=group_size,
            group_type=scale_zero_dtype,
            quant_scale=self.quantizer_kwargs.get(
                "quant_scale", "absmax"
            ),
            rmin=self.quantizer_kwargs.get("rmin", -1.0),
            rdelta=self.quantizer_kwargs.get("rdelta", 0.1),
            nstep=self.quantizer_kwargs.get("nstep", 20),
        )

        # Compute scales and zeros for supergroups and groups
        _super_group_scale, _super_group_zero, _group_scale_quant, _group_zero_quant = ([], [], [], [])
        for c in range(0, d_col, supergroup_size):
            scale, scale_quant, zero, zero_quant = quantizer.get_scale_and_zero(
                w[:, c : c + supergroup_size], q_type
            )
            _super_group_scale.append(scale)
            _super_group_zero.append(zero)
            _group_scale_quant.append(scale_quant)
            _group_zero_quant.append(zero_quant)

        super_group_scale = torch.stack(_super_group_scale, dim=1)
        super_group_zero = torch.stack(_super_group_zero, dim=1)
        group_scale_quant = torch.cat(_group_scale_quant, dim=1)
        group_zero_quant = torch.cat(_group_zero_quant, dim=1)

        # Quantize the weights
        qweight = torch.zeros_like(w, dtype=qweight_dtype)
        scale = super_group_scale.repeat_interleave(supergroup_size, dim=1)
        scale_quant = group_scale_quant.repeat_interleave(group_size, dim=1)
        zero = super_group_zero.repeat_interleave(supergroup_size, dim=1)
        zero_quant = group_zero_quant.repeat_interleave(group_size, dim=1)
        
        qweight = quantize(w, scale, scale_quant, zero, zero_quant, clamp_min_max)
        return (
            qweight.to(qweight_dtype),
            super_group_scale,
            group_scale_quant,
            super_group_zero,
            group_zero_quant,
        )
