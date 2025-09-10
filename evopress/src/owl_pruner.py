import os
from copy import deepcopy
from typing import Iterable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from src import dist_utils
from src.fast_obc import FastOBC
from src.common_utils import to, maybe_first_element
from src.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers, get_number_of_rows_and_cols


class OWLUtil:

    def __init__(self, layer) -> None:
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = get_number_of_rows_and_cols(layer)
        # init hessian
        self.H_diag = None
        self.num_samples = 0

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
        if self.H_diag is None:
            self.H_diag = torch.zeros((self.d_col,), device=input.device, dtype=torch.float32)
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
        self.H_diag.mul_(beta).add_(input.pow(2).sum(dim=0), alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size

    def get_metric(self) -> Tensor:
        assert self.H_diag is not None
        # synchronize Hessian diagonals
        if dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H_diag, op=dist.ReduceOp.AVG)
        return self.W.abs() * self.H_diag.sqrt()

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H_diag = None
        self.num_samples = 0
        torch.cuda.empty_cache()


@torch.no_grad()
def get_outlier_ratio(outlier_score: Tensor, m: float) -> float:
    return (outlier_score > m * outlier_score.mean()).float().mean().item()


class OWLFastOBCPruner:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        prunable_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        save_dir: Union[str, os.PathLike],
        rel_damp: float = 1.0e-2,
        block_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        verbose: bool = False,
        owl_m: List[float] = [5.0],
        owl_lambda: List[float] = [0.08],
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.prunable_modules = prunable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.rel_damp = rel_damp
        self.block_size = block_size
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.verbose = verbose
        self.owl_m = owl_m
        self.owl_lambda = owl_lambda

    @torch.no_grad()
    def prune(self, sparsity: float):
        """
        Args:
            sparsity: target average sparsity
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

        # Estimate sparsity distribution
        cached_input_args = deepcopy(input_args)
        cached_input_kwargs = deepcopy(input_kwargs)

        dist_utils.print_on_main(f"Estimating LOD.")
        layer_outlier_ratios = [[] for _ in self.owl_m]
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.prunable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers, "owl_util")

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

            # Collect metrics
            layer_metrics = []
            for _, handle in handles.items():
                layer_metrics.append(handle.get_metric().cpu())
            layer_metrics_agg = torch.cat([x.view(-1) for x in layer_metrics])

            # Get outlier ratio for each m
            for i, m in enumerate(self.owl_m):
                layer_outlier_ratios[i].append(get_outlier_ratio(layer_metrics_agg, m))

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        # Get sparsities
        sparsity_distributions = []
        for owl_lambda in self.owl_lambda:
            dist_utils.print_on_main("#" * 10)
            dist_utils.print_on_main(f"OWL lambda={owl_lambda:.2f}")
            for i, lor in enumerate(layer_outlier_ratios):
                lor = np.array(lor)
                lor = 2 * owl_lambda * (lor - lor.min()) / (lor.max() - lor.min())
                sparsity_distribution = sparsity - lor + np.mean(lor)
                sparsity_distributions.append(sparsity_distribution)
                dist_utils.print_on_main(f"OWL M={self.owl_m[i]}")
                dist_utils.print_on_main("-" * 10)
                dist_utils.print_on_main(f"Sparsity distribution")
                dist_utils.print_on_main("-" * 10)
                for layer_id, layer_sparsity in enumerate(sparsity_distribution):
                    dist_utils.print_on_main(f"{layer_id}: {layer_sparsity:.3f}")
        sparsity_distributions = np.stack(sparsity_distributions, axis=1)

        input_args = cached_input_args
        input_kwargs = cached_input_kwargs

        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.prunable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers, "fast_obc")

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._prune_group(handles, sparsity_distributions[block_id])

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

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, layers: Dict[str, nn.Module], handle_type: str):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])

                return _hook

            handles[layer_name] = self._create_handle(layer, handle_type)
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, layer, handle_type: str):
        if handle_type == "fast_obc":
            return FastOBC(layer, rel_damp=self.rel_damp, block_size=self.block_size)
        elif handle_type == "owl_util":
            return OWLUtil(layer)
        else:
            raise ValueError("Unknown handle type.")

    def _prune_group(self, handles: Dict[str, FastOBC], sparsities: List[float]):
        for handle_name, handle in handles.items():
            if self.verbose:
                dist_utils.print_on_main(f"Pruning {handle_name}")
            sparse_weights = handle.prune(sparsities)
            if dist_utils.is_main():
                for i, sparse_weight in enumerate(sparse_weights):
                    os.makedirs(os.path.join(self.save_dir, handle_name), exist_ok=True)
                    torch.save(sparse_weight, os.path.join(self.save_dir, handle_name, f"{i}.pth"))
            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()
            handle.reset()
