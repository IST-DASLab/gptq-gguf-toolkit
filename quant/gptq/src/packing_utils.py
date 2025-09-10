import torch
import gguf

import numpy as np

from gguf.constants import QK_K

def pack_scale_min_torch(scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """
    Pack the scale and min values into a 12-byte format.
    """
    assert scale.shape == zero.shape and scale.shape[1] == 8
    assert scale.dtype == torch.uint8 and zero.dtype == torch.uint8

    n = scale.size(0)
    packed = torch.zeros((n, 12), dtype=torch.uint8, device=scale.device)

    packed[:, 0:4] = scale[:, 0:4]
    packed[:, 4:8] = zero[:, 0:4]

    sc_lo = scale[:, 4:8] & 0x0F  # bits 0-3
    mn_lo = zero[:, 4:8] & 0x0F
    sc_hi = (scale[:, 4:8] >> 4) << 6  # bits 6-7
    mn_hi = (zero[:, 4:8] >> 4) << 6

    packed[:, 8:12] = sc_lo | (mn_lo << 4)

    packed[:, 0:4] |= sc_hi
    packed[:, 4:8] |= mn_hi
    return packed


def pack_Q2K(
    qweights: torch.Tensor, 
    super_group_scale: torch.Tensor,  
    group_scale_quant: torch.Tensor, 
    super_group_zero: torch.Tensor,
    group_zero_quant: torch.Tensor,
) -> np.ndarray:
    """
    Pack Q2_K quantized weights into a format suitable for GGUF.
    """
    N, W = qweights.shape
    B, TYPE_SIZE = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_K]
    assert W % B == 0
    blocks_per_row = W // B

    # flatten into blocks
    qweights = qweights.reshape(-1, B)                  # (n_blocks,256)
    super_group_scale = super_group_scale.reshape(-1, 1)                      # (n_blocks,1)
    super_group_zero = super_group_zero.reshape(-1, 1)                        # (n_blocks,1)
    group_scale_quant = group_scale_quant.reshape(-1, QK_K // 16) # (n_blocks,16)
    group_zero_quant = group_zero_quant.reshape(-1, QK_K // 16)   # (n_blocks,16)
    n_blocks = qweights.shape[0]

    scales_bytes = ((group_scale_quant & 0x0F) | ((group_zero_quant & 0x0F) << 4)).to(torch.uint8)

    packed_qs = torch.zeros((n_blocks, QK_K // 4), dtype=torch.uint8, device=qweights.device) 
    for chunk in range(2):
        base = chunk * 128
        offset = chunk * 32
        idx = torch.arange(32, device=qweights.device)
        v0 = qweights[:, base + idx]
        v1 = qweights[:, base + 32 + idx]
        v2 = qweights[:, base + 64 + idx]
        v3 = qweights[:, base + 96 + idx]
        # pack 4×2-bit into one byte per position
        packed_qs[:, offset : offset + 32] = (
            (v0) | (v1 << 2) | (v2 << 4) | (v3 << 6)
        ).to(torch.uint8)

    d_bytes = super_group_scale.to(torch.float16).view(torch.uint8)
    dmin_bytes = super_group_zero.to(torch.float16).view(torch.uint8)

    blocks = torch.cat([scales_bytes, packed_qs, d_bytes, dmin_bytes], dim=1)  # (n_blocks,84)

    return blocks.view(N, blocks_per_row * TYPE_SIZE).contiguous().cpu().numpy()


def pack_Q3K(
    qweights: torch.Tensor,
    super_group_scale: torch.Tensor,
    group_scale_quant: torch.Tensor,
) -> np.ndarray:
    """
    Pack Q3_K quantized weights into a format suitable for GGUF.
    """
    N, W = qweights.shape
    B, TYPE_SIZE = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q3_K]
    assert W % B == 0, "W must be a multiple of 256"
    blocks_per_row = W // B

    # make quantized weights unsigned
    qweights += 4
    group_scale_quant += 32

    qweights = qweights.reshape(-1, B)                                  # (n_blocks,256)
    group_scale_quant = group_scale_quant.reshape(-1, QK_K // 16)       # (n_blocks,16)
    super_group_scale = super_group_scale.reshape(-1, 1)                # (n_blocks,1)
    n_blocks = qweights.shape[0]

    scales_bytes = torch.zeros((n_blocks, 12), dtype=torch.uint8, device=qweights.device)
    for j in range(16):
        lj = group_scale_quant[:, j].to(torch.uint8) 
        lo4 = lj & 0x0F
        hi2 = (lj >> 4) & 0x03
        # low nibble:
        if j < 8:
            scales_bytes[:, j] |= lo4
        else:
            scales_bytes[:, j - 8] |= lo4 << 4
        # high bits into bytes 8..11
        hi_idx = 8 + (j % 4)
        shift = 2 * (j // 4)
        scales_bytes[:, hi_idx] |= hi2 << shift

    L = qweights.clone()
    hmask = torch.zeros((n_blocks, QK_K // 8), dtype=torch.uint8, device=L.device)
    for j in range(QK_K):
        grp = j // (QK_K // 8)  # 0..7
        bit = 1 << grp
        idx = j % (QK_K // 8)  # 0..31
        gt = L[:, j] > 3
        hmask[:, idx] |= gt.to(torch.uint8) * bit
        L[:, j] = torch.where(gt, L[:, j] - 4, L[:, j])

    qs = torch.zeros((n_blocks, QK_K // 4), dtype=torch.uint8, device=L.device)
    for chunk in range(2):
        base = chunk * 128
        offset = chunk * 32
        idx = torch.arange(32, device=L.device)
        v0 = L[:, base + idx]
        v1 = L[:, base + 32 + idx]
        v2 = L[:, base + 64 + idx]
        v3 = L[:, base + 96 + idx]
        qs[:, offset : offset + 32] = ((v0) | (v1 << 2) | (v2 << 4) | (v3 << 6)).to(torch.uint8)

    d_bytes = super_group_scale.to(torch.float16).view(torch.uint8)

    blocks = torch.cat([hmask, qs, scales_bytes, d_bytes], dim=1)  # (n_blocks, 110)
    
    return blocks.view(N, blocks_per_row * TYPE_SIZE).contiguous().cpu().numpy()


def pack_Q4K(
    qweights: torch.Tensor,
    super_group_scale: torch.Tensor,
    group_scale_quant: torch.Tensor,
    super_group_zero: torch.Tensor,
    group_zero_quant: torch.Tensor,
) -> bytes:
    """
    Pack Q4_K quantized weights into a format suitable for GGUF.
    """
    N, W = qweights.shape 
    B, TYPE_SIZE = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_K]
    blocks_per_row = W // B 

    qweights = qweights.reshape(-1, B)
    super_group_scale = super_group_scale.reshape(-1, 1)
    group_scale_quant = group_scale_quant.reshape(-1, 8)
    super_group_zero = super_group_zero.reshape(-1, 1)
    group_zero_quant = group_zero_quant.reshape(-1, 8)

    n_blocks = qweights.shape[0]
    packed_qs = torch.zeros((n_blocks, QK_K // 2), dtype=torch.uint8, device=qweights.device)

    for base in range(0, 256, 64):
        lo = qweights[:, base : base + 32]      
        hi = qweights[:, base + 32 : base + 64] 
        idx = (base // 64) * 32 
        packed_qs[:, idx : idx + 32] = lo | (hi << 4)

    packed_scales_mins_bytes = pack_scale_min_torch(group_scale_quant, group_zero_quant)
    blocks = torch.cat(
        [
            super_group_scale.to(torch.float16).view(torch.uint8),
            super_group_zero.to(torch.float16).view(torch.uint8),
            packed_scales_mins_bytes.view(torch.uint8),
            packed_qs.view(torch.uint8),
        ],
        dim=1,
    ).to(torch.uint8)

    return (
        blocks.reshape(N, blocks_per_row * TYPE_SIZE)
        .view(torch.uint8)
        .contiguous()
        .numpy()
    )


def pack_Q5K(
    qweights: torch.Tensor,
    super_group_scale: torch.Tensor,
    group_scale_quant: torch.Tensor,
    super_group_zero: torch.Tensor,
    group_zero_quant: torch.Tensor,
) -> np.ndarray:
    """
    Pack Q5_K quantized weights into a format suitable for GGUF.
    """
    N, W = qweights.shape
    B, TYPE_SIZE = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q5_K]
    
    blocks_per_row = W // B
    assert W % B == 0, "W must be divisible by 256"

    qweights = qweights.reshape(-1, B)
    n_blocks = qweights.shape[0]

    super_group_scale = super_group_scale.reshape(-1, 1)
    super_group_zero = super_group_zero.reshape(-1, 1)
    group_scale_quant = group_scale_quant.reshape(-1, 8)
    group_zero_quant = group_zero_quant.reshape(-1, 8)

    packed_scales_mins_bytes = pack_scale_min_torch(group_scale_quant, group_zero_quant)
    
    qweights = qweights.to(torch.uint8)
    packed_qh = torch.zeros((n_blocks, QK_K // 8), dtype=torch.uint8, device=qweights.device)
    packed_ql = torch.zeros((n_blocks, QK_K // 2), dtype=torch.uint8, device=qweights.device)

    m1, m2 = 1, 2
    for base in range(0, B, 64):
        for j in range(32):
            l1 = qweights[:, base + j]
            l2 = qweights[:, base + j + 32]

            mask1 = l1 > 15
            mask2 = l2 > 15

            l1 = torch.where(mask1, l1 - 16, l1)
            l2 = torch.where(mask2, l2 - 16, l2)

            packed_qh[:, j] |= mask1.to(torch.uint8) * m1
            packed_qh[:, j] |= mask2.to(torch.uint8) * m2

            packed_ql[:, j + (base // 2)] = l1 | (l2 << 4)

        m1 <<= 2
        m2 <<= 2

    d_bytes = super_group_scale.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    dmin_bytes = super_group_zero.to(torch.float16).view(torch.uint8).reshape(-1, 2)

    blocks = torch.cat(
        [
            d_bytes,
            dmin_bytes,
            packed_scales_mins_bytes,
            packed_qh,
            packed_ql,
        ],
        dim=1,
    )

    return (
        blocks.reshape(N, blocks_per_row * TYPE_SIZE)
        .view(torch.uint8)
        .contiguous()
        .numpy()
    )


def pack_Q6K(
    qweights: torch.Tensor, super_group_scales: torch.Tensor, group_scale_quant: torch.Tensor
) -> np.ndarray:
    """
    Pack Q6_K quantized weights into a format suitable for GGUF.
    """
    N, W = qweights.shape
    QK_K = gguf.constants.QK_K
    B, TYPE_SIZE = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q6_K]
    
    assert W % B == 0
    blocks_per_row = W // B

    # make quantized weights unsigned
    qweights += 32 

    qweights = qweights.reshape(-1, B)                              # (n_blocks, 256)
    super_group_scales = super_group_scales.reshape(-1, 1)          # (n_blocks, 1)
    group_scale_quant = group_scale_quant.reshape(-1, QK_K // 16)   # (n_blocks, 16)
    n_blocks = qweights.shape[0]

    # allocate output containers
    packed_ql = torch.zeros((n_blocks, QK_K // 2), dtype=torch.uint8, device=qweights.device)
    packed_qh = torch.zeros((n_blocks, QK_K // 4), dtype=torch.uint8, device=qweights.device)

    # pack low and high nibbles in 2 chunks of 128 each
    for chunk in range(2):
        j = chunk * 128
        offset_ql = chunk * (QK_K // 4)
        offset_qh = chunk * (QK_K // 8)

        block_vals = qweights[:, j : j + 128]

        for l in range(32):
            v0 = block_vals[:, l + 0]
            v1 = block_vals[:, l + 32]
            v2 = block_vals[:, l + 64]
            v3 = block_vals[:, l + 96]

            # low nibble packing: ql: bits [0:4] of each value, spread across 2 bits apiece
            lo0 = v0 & 0xF
            lo1 = v1 & 0xF
            lo2 = v2 & 0xF
            lo3 = v3 & 0xF
            packed_ql[:, offset_ql + l] = (lo0 | (lo2 << 4)).to(torch.uint8)
            packed_ql[:, offset_ql + 32 + l] = (lo1 | (lo3 << 4)).to(torch.uint8)

            # high nibble packing: qh: bits [4:6] of each value, spread across 2 bits apiece
            hi0 = (v0 >> 4) & 0x3
            hi1 = (v1 >> 4) & 0x3
            hi2 = (v2 >> 4) & 0x3
            hi3 = (v3 >> 4) & 0x3
            packed_qh[:, offset_qh + l] = (
                hi0 | (hi1 << 2) | (hi2 << 4) | (hi3 << 6)
            ).to(torch.uint8)

    scales_bytes = group_scale_quant.to(torch.uint8)
    d_bytes = super_group_scales.to(torch.float16).view(torch.uint8)

    blocks = torch.cat([packed_ql, packed_qh, scales_bytes, d_bytes], dim=1)
    
    return blocks.view(N, blocks_per_row * TYPE_SIZE).contiguous().cpu().numpy()
