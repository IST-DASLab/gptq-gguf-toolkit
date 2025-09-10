import itertools
import math
from enum import Enum, IntEnum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from gguf.constants import QK_K

class GGMLQuantizationType(IntEnum):
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14


GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int, int]] = {
    # bits, q_clamp_values, max_scale_q, group_size, super_group_size, dtype_scale_zero, dtype_qweight
    GGMLQuantizationType.Q2_K: (2, (0, 2**2 - 1), 2**4 - 1, 16, QK_K, torch.uint8, torch.uint8),
    GGMLQuantizationType.Q3_K: (3, (-4, 3), 2**5 - 1, 16, QK_K, torch.int8, torch.int8),
    GGMLQuantizationType.Q4_K: (4, (0, 2**4 - 1), 2**6 - 1, 32, QK_K, torch.uint8, torch.uint8),
    GGMLQuantizationType.Q5_K: (5, (0, 2**5 - 1), 2**6 - 1, 32, QK_K, torch.uint8, torch.uint8),
    GGMLQuantizationType.Q6_K: (6, (-32, 31), 2**6 - 1, 16, QK_K, torch.int8, torch.int8),
}


class QuantizationScale(str, Enum):
    ABSMAX = "absmax"
    MSE = "mse"


def quantize(x, super_group_scale, group_scale_quants, super_group_zero, group_zero_quants, clamp_min_max, eps=1e-9):
    q = torch.round(
        (x + (super_group_zero.to(torch.float32) * group_zero_quants))
        / (super_group_scale.to(torch.float32) * group_scale_quants).clamp_min(eps)
    )
    q = torch.clamp(q, clamp_min_max[0], clamp_min_max[1])
    return q


def dequantize(q, super_group_scale, group_scale_quants, super_group_zero, group_zero_quants):
    return (super_group_scale.to(torch.float32) * group_scale_quants) * q - (
        super_group_zero.to(torch.float32) * group_zero_quants
    )


class Quantizer(nn.Module):

    def __init__(self):
        super().__init__()

    def configure(
        self,
        bits,
        scale_maxq: int,
        group_size: int,
        group_type: torch.dtype,
        super_group_size: int,
        # Scale search parameters
        quant_scale: QuantizationScale = QuantizationScale.ABSMAX,
        grid: int = 100,
        maxshrink: float = 0.80,
        norm: float = 2.0,
        rmin: float = -1.0,
        rdelta: float = 0.1,
        nstep: int = 20,
        eps: float = 1e-9,
    ):
        self.bits = bits
        self.maxq = 2**bits - 1
        self.scale_maxq = scale_maxq
        self.group_size = group_size
        self.supergroup_size = super_group_size
        self.group_type = group_type

        # Scale & Zero search parameters for K-Quants
        self.rmin = rmin
        self.rdelta = rdelta
        self.nstep = nstep
        self.eps = eps

        # Scale search parameters
        self.quant_scale = quant_scale
        self.grid = grid
        self.maxshrink = maxshrink
        self.norm = norm

    def get_scale_and_zero(
        self, x: torch.Tensor, q_type: GGMLQuantizationType
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 2 and x.shape[1] == QK_K, f"expected (rows, {QK_K})"

        d_row, d_col = x.shape
        num_groups = d_col // self.group_size

        if q_type == GGMLQuantizationType.Q2_K:
            make_quant = self.make_k_quants
        elif q_type == GGMLQuantizationType.Q3_K:
            make_quant = self.make_quants
        elif q_type == GGMLQuantizationType.Q4_K:
            make_quant = self.make_k_quants
        elif q_type == GGMLQuantizationType.Q5_K:
            make_quant = self.make_k_quants
        elif q_type == GGMLQuantizationType.Q6_K:
            make_quant = self.make_quants
        else:
            raise ValueError(f"Unsupported quantization type: {q_type}")

        # Calculate the scale and zero for the quantization
        group_scale, group_zero = make_quant(
            x.view(d_row, num_groups, self.group_size).reshape(-1, self.group_size)
        )

        # reshape scale and zero to (d_row, num_groups)
        scale_per_super_group = group_scale.view(d_row, num_groups)
        zero_per_super_group = group_zero.view(d_row, num_groups)

        # calculate the maximum scale and zero for each super group
        max_scale, max_zero = scale_per_super_group.amax(dim=1), zero_per_super_group.amax(dim=1)

        # calculate the scale and zero for each super group
        super_group_scale = (max_scale / self.scale_maxq).to(torch.float16)
        super_group_zero = (max_zero / self.scale_maxq).to(torch.float16)

        # calculate the inverse scale and zero for quantization
        inv_scale = torch.where(max_scale > 0, self.scale_maxq / max_scale, torch.zeros_like(max_scale))
        inv_zero = torch.where(max_zero > 0, self.scale_maxq / max_zero, torch.zeros_like(max_zero))

        # calculate the quantized scales and zeros
        group_scale_quant = (
            (inv_scale.unsqueeze(1) * scale_per_super_group)
            .round()
            .clamp(0, self.scale_maxq)
            .to(self.group_type)
        )
        group_zero_quant = (
            (inv_zero.unsqueeze(1) * zero_per_super_group)
            .round()
            .clamp(0, self.scale_maxq)
            .to(self.group_type)
        )

        return super_group_scale, group_scale_quant, super_group_zero, group_zero_quant

    def make_quants(self, x: torch.Tensor) -> torch.Tensor | torch.Tensor:
        x = x.flatten(1)

        xmin = x.min(dim=1, keepdim=True).values
        xmax = x.max(dim=1, keepdim=True).values

        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / self.maxq
        zero = torch.full_like(scale, (self.maxq + 1) / 2)

        if self.quant_scale == QuantizationScale.MSE:
            min_loss = torch.full((x.shape[0],), float("inf"), device=x.device)
            best_scale = torch.zeros_like(scale)
            best_zero = torch.zeros_like(zero)

            for i in range(int(self.maxshrink * self.grid) + 1):
                alpha = 1 - i / (self.maxshrink * self.grid)
                cand_max = torch.max(xmax, torch.abs(xmin)) * alpha

                xmax1 = torch.min(xmax, +cand_max)
                xmin1 = torch.max(xmin, -cand_max)

                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = zero

                q_int = torch.clamp(
                    ((x - zero1) / scale1.clamp_min(1e-9).round()), 0, self.maxq
                )
                y_rec = q_int * scale1 + zero1  # ← de-quantize
                loss = (y_rec - x).pow(self.norm).sum(dim=1)

                is_better = loss < min_loss
                if is_better.any():
                    min_loss[is_better] = loss[is_better]
                    best_scale[is_better] = scale1[is_better]
                    best_zero[is_better] = zero1[is_better]
            scale.copy_(best_scale)
            zero.copy_(best_zero)

        # overwrite zero to be ignored in quantization, since when this method is used
        # we use systematic quantization and don't encode qweight to 0..maxq
        zero = torch.zeros(zero.shape, dtype=zero.dtype, device=zero.device)

        return scale, zero

    def make_k_quants(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute per-group weights
        sum_x2 = (x * x).sum(dim=1, keepdim=True)  # (rows,1)
        av_x = torch.sqrt(sum_x2 / x.shape[1])  # (rows,1)
        weights = av_x + x.abs()  # (rows,32)

        # initial min/max
        x_min = x.min(dim=1, keepdim=True).values  # (rows,1)
        x_max = x.max(dim=1, keepdim=True).values
        x_min = torch.minimum(x_min, torch.zeros_like(x_min))
        const_mask = (x_max == x_min).squeeze(1)  # (rows,)

        # compute initial sums
        sum_w = weights.sum(dim=1, keepdim=True)  # (rows,1)
        sum_x = (weights * x).sum(dim=1, keepdim=True)  # (rows,1)

        # initial scale and inverse scale
        scale = (x_max - x_min) / self.maxq  # (rows,1)
        scale = torch.where(const_mask.unsqueeze(1), torch.zeros_like(scale), scale)
        iscale = torch.reciprocal(scale.clamp_min(self.eps))  # (rows,1)

        # initial q
        q = torch.clamp(torch.round((x - x_min) * iscale), 0, self.maxq).to(torch.uint8)
        if const_mask.any():
            q[const_mask, :] = 0

        # inital error
        best_min = x_min  # (rows,1)
        best_scale = scale  # (rows,1)
        diff = best_scale * q + best_min - x  # (rows,32)
        err = diff * diff
        best_err = (weights * err).sum(dim=1)  # (rows,)

        # early return if no refinement
        if self.nstep < 1:
            zero = (-best_min).squeeze(1)
            return best_scale.squeeze(1), zero

        # refinement loop
        for i in range(self.nstep + 1):
            cand_iscale = (self.rmin + self.rdelta * i + self.maxq) / (x_max - x_min).clamp_min(self.eps)
            new_q = torch.clamp(torch.round((x - x_min) * cand_iscale), 0, self.maxq).to(torch.uint8)
            new_q[const_mask, :] = 0

            sum_l = (weights * new_q).sum(dim=1, keepdim=True)  # (rows,1)
            sum_l2 = (weights * (new_q**2)).sum(dim=1, keepdim=True)  # (rows,1)
            sum_xl = (weights * x * new_q).sum(dim=1, keepdim=True)  # (rows,1)

            D = sum_w * sum_l2 - sum_l * sum_l  # (rows,1)
            valid = D > self.eps  # (rows,1)
            if not valid.any():
                continue

            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D  # (rows,1)
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D  # (rows,1)

            pos_mask = (this_min > 0).squeeze(1)
            if pos_mask.any():
                this_scale[pos_mask, :] = (sum_xl / (sum_l2.clamp_min(self.eps)))[pos_mask, :]
                this_min[pos_mask, :] = 0

            diff = this_scale * new_q + this_min - x
            err = diff * diff
            cand_err = (weights * err).sum(dim=1)  # (rows,)

            better = cand_err < best_err
            if better.any():
                best_err[better] = cand_err[better]
                best_scale[better, :] = this_scale[better, :]
                best_min[better, :] = this_min[better, :]
                q[better, :] = new_q[better, :]

        zero = (-best_min).squeeze(1)
        return best_scale.squeeze(1), zero


def dequantize_linear_weight(
    q_type: GGMLQuantizationType,
    qweight: torch.Tensor,
    super_group_scale: torch.Tensor,
    group_scale_quant: torch.Tensor,
    super_group_zero: torch.Tensor,
    group_zero_quant: torch.Tensor,
) -> torch.Tensor:
    _, _, _, group_size, supergroup_size, _, _ = GGML_QUANT_SIZES[q_type]
    d_row, d_col = qweight.shape
    num_groups = d_col // group_size
    num_super_groups = d_col // supergroup_size
    groups = supergroup_size // group_size

    scale = (
        super_group_scale.view(d_row, num_super_groups, 1)
        .expand(d_row, num_super_groups, groups)
        .reshape(d_row, num_groups, 1)
    )
    zero = (
        super_group_zero.view(d_row, num_super_groups, 1)
        .expand(d_row, num_super_groups, groups)
        .reshape(d_row, num_groups, 1)
    )

    weight = dequantize(
        qweight.view(d_row, num_groups, group_size),
        scale,
        group_scale_quant.view(d_row, num_groups, 1),
        zero,
        group_zero_quant.view(d_row, num_groups, 1),
    ).view_as(qweight)

    return weight
