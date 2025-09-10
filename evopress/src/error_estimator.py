import os
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils
from src.common_utils import to, maybe_first_element
from src.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers, get_number_of_rows_and_cols


class LayerErrorEstimator:

    def __init__(self, layer: nn.Module):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = get_number_of_rows_and_cols(layer)
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "OBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        # synchronize Hessians
        if dist_utils.is_dist_available_and_initialized():
            dist.reduce(self.H, dst=0, op=dist.ReduceOp.AVG)
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # flag pre step as completed
        self.pre_step_completed = True
        # cast weight to float32
        self.W = self.W.float()

    def estimate(self, W_c: Tensor) -> float:
        """
        Compute (W - W_c)^T @ H (W - W_c) 
        """
        assert self.pre_step_completed
        H, W = self.H, self.W
        error = ((W - W_c) @ H * (W - W_c)).sum()
        norm = (W @ H * W).sum()
        return (error / norm).item()


class ErrorEstimator:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        target_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        compressed_weights_path: str,
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.target_modules = target_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.verbose = verbose
        self.compressed_weights_path = compressed_weights_path

    @torch.no_grad()
    def estimate(self, group_by_numel: bool = False) -> Dict[int, Dict[str, List[float]]]:
        """
        Args:
            group_by_numel: whether to group by layer size
        Returns:
            Normalized L2 error for each layer.
        """
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

        all_errors = defaultdict(dict)
        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.target_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
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

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            block_errors = self._estimate_errors_group(handles)
            # Update block errors:
            if group_by_numel:
                for k, v in block_errors.items():
                    all_errors[handles[k].W.numel()][k] = v
            else:
                all_errors[-1].update(block_errors)

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

        return all_errors

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
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, layer):
        return LayerErrorEstimator(layer)

    def _estimate_errors_group(self, handles: Dict[str, LayerErrorEstimator]) -> Dict[str, List[float]]:
        errors = defaultdict(list)
        for handle_name, handle in handles.items():
            if self.verbose:
                dist_utils.print_on_main(f"Pruning {handle_name}")
            # Sync Hessians across processes
            handle.pre_step()
            if dist_utils.is_main():
                for weight_path in sorted(os.listdir(os.path.join(self.compressed_weights_path, handle_name)), key=lambda x: int(x.split(".")[0])):
                    w = torch.load(
                        os.path.join(self.compressed_weights_path, handle_name, weight_path),
                        map_location=handle.W_device
                    ).float()
                    error = handle.estimate(w)
                    errors[handle_name].append(error)

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()
            handle.reset()
        return errors
