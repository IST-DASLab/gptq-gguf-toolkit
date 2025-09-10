from typing import Iterable, Dict, List, Any, Optional, Union


import os
import torch
import torch.nn as nn
import torch.distributed as dist


from src import dist_utils
from src.common_utils import to, maybe_first_element
from src.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers
from src.quant_utils import QLinear

from src.fast_obq import FastOBQ


class Quantizer:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        save_dir: Union[str, os.PathLike],
        block_modules: str,
        obq_kwargs: Dict[str, Any] = {},
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.obq_kwargs = obq_kwargs
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.verbose = verbose

    @torch.no_grad()
    def quantize(self, bitwidth_options: List[int], calibration_bitwidth: int):
        device = self.device or next(self.model.parameters()).device
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [self._get_submodule(module_name) for module_name in self.pre_block_modules]
        blocks[0] = blocks[0].to(device)
        for module in pre_blocks:
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
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # offload pre_blocks
        if self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.quantizable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(bitwidth_options, layers)

            targets = []
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._quant_group(handles, bitwidth_options, calibration_bitwidth)

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

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, bitwidth_options: List[int], layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])

                return _hook

            handles[layer_name] = self._create_handle(bitwidth_options, layer)
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, bitwidth_options, layer):
        return FastOBQ(layer, bitwidth_options=bitwidth_options, **self.obq_kwargs)

    def _quant_group(self, handles: Dict[str, FastOBQ], bitwidth_options: List[int], calibration_bitwidth: int):
        for handle_name, handle in handles.items():

            if self.verbose:
                dist_utils.print_on_main(f"Quantizing {handle_name}")
            qweight_dict, scale_dict, zero_dict, perm = handle.quantize(bitwidth_options)

            for bits in bitwidth_options:
                qlayer = QLinear(
                    qweight_dict[bits],
                    scale_dict[bits],
                    zero_dict[bits],
                    bias=handle.layer.bias,
                    perm=perm,
                    bits=8 if bits > 4 else 4,
                )
                dequantized_weight = qlayer.get_weight()
                os.makedirs(os.path.join(self.save_dir, handle_name), exist_ok=True)
                # Map tensor to CPU before save
                torch.save(dequantized_weight.cpu(), os.path.join(self.save_dir, handle_name, f"{int(bits)}.pth"))
                # Replace original layer by quantized layer with given bitwidth
                if bits == calibration_bitwidth:
                    parent_name, child_name = handle_name.rsplit(".", 1)
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, qlayer)
            handle.reset()
