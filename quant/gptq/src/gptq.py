from enum import Enum
import itertools
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils
from src import model_utils
from src import linalg_utils
from src.quant_utils import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    QuantizationScale,
    Quantizer,
    dequantize,
    quantize,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(
        self,
        layer: nn.Module,
        rel_damp: float = 1e-2,
        block_size: Optional[int] = None,
        act_order: bool = False,
        quant_scale: str = "absmax",
        # quantization hyperparameters
        rmin: float = -1.0,
        rdelta: float = 0.1,
        nstep: int = 20,
        # quantization grid
        grid: int = 100,
        static_groups: bool = False,
        verbose: bool = False,
    ):
        if act_order:
            assert static_groups
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        # Quantizer hyperparameters
        self.quantizer = Quantizer()
        # GPTQ hyperparameters
        self.rel_damp = rel_damp
        self.block_size = block_size or self.d_col
        self.act_order = act_order
        self.quant_scale = QuantizationScale(quant_scale)
        self.static_groups = static_groups
        self.grid = grid
        self.rmin = rmin
        self.rdelta = rdelta
        self.nstep = nstep

        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # misc args
        self.verbose = verbose

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
            self.H = torch.zeros(
                (self.d_col, self.d_col), device=input.device, dtype=torch.float32
            )
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
    def quantization_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"

        # synchronize Hessians
        if dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        self.W[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def step(self, q_type: GGMLQuantizationType) -> Tensor:
        bits, clamp_min_max, scale_maxq, group_size, supergroup_size, scale_zero_dtype, qweight_dtype = GGML_QUANT_SIZES[q_type]

        # 1) unpack constants
        d_row, d_col = self.d_row, self.d_col
        block_size   = self.block_size
        device, dtype = self.W_device, self.W_dtype

        # 2) compute number of groups and supergroups
        num_groups = d_col // group_size
        num_super_groups = d_col // supergroup_size

        if dist_utils.is_main():
            w = self.W

            # allocate tensors
            qweight = torch.empty(d_row, d_col, device=device, dtype=qweight_dtype)
            super_group_scale = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            super_group_zero = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            group_scale_quant = torch.empty(d_row, num_super_groups, device=device, dtype=scale_zero_dtype)
            group_zero_quant = torch.empty(d_row, num_super_groups, device=device, dtype=scale_zero_dtype)

            # configure quantizer
            quantizer = self.quantizer 
            quantizer.configure(
                bits=bits, 
                scale_maxq=scale_maxq, 
                super_group_size = supergroup_size, 
                group_size=group_size, 
                quant_scale=self.quant_scale,
                grid=self.grid,
                rmin=self.rmin, 
                rdelta=self.rdelta, 
                nstep=self.nstep, 
                group_type=scale_zero_dtype
            )
            
            # Init scales and zeros
            if self.static_groups:
                _super_group_scale, _super_group_zero, _group_scale_quant, _group_zero_quant = ([], [], [], [])
                for c in range(0, d_col, supergroup_size):
                    scale, scale_quant, zero, zero_quant = quantizer.get_scale_and_zero(w[:, c : c + supergroup_size], q_type)
                    _super_group_scale.append(scale)
                    _super_group_zero.append(zero)
                    _group_scale_quant.append(scale_quant)
                    _group_zero_quant.append(zero_quant)

                super_group_scale = torch.stack(_super_group_scale, dim=1)
                super_group_zero = torch.stack(_super_group_zero, dim=1)
                group_scale_quant = torch.cat(_group_scale_quant, dim=1)
                group_zero_quant = torch.cat(_group_zero_quant, dim=1)
            else:
                super_group_scale = torch.empty(d_row, num_super_groups, device=device, dtype=torch.float16)
                super_group_zero = torch.empty(d_row, num_super_groups, device=device, dtype=torch.float16)
                group_scale_quant = torch.empty(d_row, num_groups, device=device, dtype=scale_zero_dtype)
                group_zero_quant = torch.empty(d_row, num_groups, device=device, dtype=scale_zero_dtype)    

            # Best option for Q3_K
            if q_type == GGMLQuantizationType.Q3_K:
                self.act_order = False
                self.static_groups = False

            # --- optional activation‐ordering permute ---
            perm = None
            group_idx = None
            if self.act_order:
                perm = torch.argsort(torch.diag(self.H), descending=True)
                self.W.data = w[:, perm]
                self.H.data = self.H[perm][:, perm]
                group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[perm]
                super_group_idx = torch.arange(num_super_groups, device=device).repeat_interleave(supergroup_size)[perm]

            # prepare weight and Cholesky of H^{-1}
            H_inv_cho = self._prepare()
            g_idx, s_g_idx = 0, 0
            # iterate over columns
            for c1 in range(0, d_col, block_size):
                c2 = min(c1 + block_size, d_col)
                ncols = c2 - c1  # number of columns
                w_blk = w[:, c1:c2].clone()  # column-wise weight slice
                errs = torch.zeros_like(w_blk)
                H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
                # 2) iterate over block
                for i in range(ncols):
                    w_ci = w_blk[:, i]
                    d = H_inv_cho_blk[i, i]

                    if self.act_order:
                        g_idx = group_idx[c1 + i]
                        s_g_idx = super_group_idx[c1 + i]
                    else:
                        g_idx = (c1 + i) // group_size
                        s_g_idx = (c1 + i) // supergroup_size

                    if not self.static_groups and (c1 + i) % supergroup_size == 0:
                        scale, scale_quant, zero, zero_quant = quantizer.get_scale_and_zero(w[:, (c1 + i) : (c1 + i + supergroup_size)], q_type)
                        super_group_scale[:, s_g_idx] = scale.flatten()
                        super_group_zero[:, s_g_idx] = zero.flatten()
                        group_scale_quant[:, g_idx:(g_idx + (supergroup_size // group_size))] = scale_quant
                        group_zero_quant[:, g_idx:(g_idx + (supergroup_size // group_size))] = zero_quant

                    q = quantize(
                        w_ci, 
                        super_group_scale[:, s_g_idx], 
                        group_scale_quant[:, g_idx], 
                        super_group_zero[:, s_g_idx], 
                        group_zero_quant[:, g_idx], 
                        clamp_min_max
                    )
                    w_q = dequantize(
                        q, 
                        super_group_scale[:, s_g_idx], 
                        group_scale_quant[:, g_idx], 
                        super_group_zero[:, s_g_idx], 
                        group_zero_quant[:, g_idx]
                    )
                    
                    qweight[:, c1 + i] = q.flatten()
                    err = (w_ci - w_q) / d

                    w[:, c1 + i] = w_q
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) update the weights after block
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

            # undo the activation‐order permute
            if perm is not None:
                invperm  = torch.argsort(perm)
                self.H   = self.H[invperm][:, invperm]
                qweight = qweight[:, invperm]

        else:
            # non-main replicas just allocate
            qweight = torch.empty(d_row, d_col, device=device, dtype=qweight_dtype)
            super_group_scale = torch.empty(d_row, num_super_groups, device=device, dtype=torch.float16)
            super_group_zero = torch.empty(d_row, num_super_groups, device=device, dtype=torch.float16)
            group_scale_quant = torch.empty(d_row, num_groups, device=device, dtype=scale_zero_dtype)
            group_zero_quant = torch.empty(d_row, num_groups, device=device, dtype=scale_zero_dtype)

        # broadcast to all ranks
        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(qweight, src=0)
            dist.broadcast(super_group_scale, src=0)
            dist.broadcast(super_group_zero, src=0)
            dist.broadcast(group_scale_quant, src=0)
            dist.broadcast(group_zero_quant, src=0)

        return qweight, super_group_scale, group_scale_quant, super_group_zero, group_zero_quant

    def quantize(
        self,
        q_type: GGMLQuantizationType
    ) -> Tuple[Tensor, Tensor, Tensor, Any]:
        self.quantization_pre_step()
        return self.step(q_type)

    @torch.no_grad()
    def _prepare(self):
        w = self.W
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(H).mean()
        H[range(self.d_col), range(self.d_col)] += damp
        # invert
        try:
            H = linalg_utils.inv_sym(H)
            H_inv_cho = torch.linalg.cholesky(H, upper=True)
        except:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
            self.issue_non_invertible = True
        return H_inv_cho
