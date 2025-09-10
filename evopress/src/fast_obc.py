from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils
from src import model_utils
from src import linalg_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class FastOBC:

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, block_size: int = None, verbose: bool = False):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        # FastOBC hyperparameters
        self.rel_damp = rel_damp
        self.block_size = block_size or self.d_col
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
        assert isinstance(layer, (nn.Linear, _ConvNd)), "FastOBC supports only linear and convolutional layers."

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
    def pruning_pre_step(self) -> None:
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
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        self.W[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    def step(self, sparsities: List[float]) -> List[Tensor]:
        # 1) define constants and chunk
        d_col, block_size, device, dtype = self.d_col, self.block_size, self.W_device, self.W_dtype

        if dist_utils.is_main():
            torch.cuda.empty_cache()
            # prepare empty list for sparse weights
            sparse_weights = []
            # prepare weight and Cholesky of H^{-1}
            w_orig, H_inv_cho_orig = self._prepare()

            for i, sparsity in enumerate(sparsities):
                if i + 1 < len(sparsities):
                    w, H_inv_cho = w_orig.clone(), H_inv_cho_orig.clone()
                else:
                    w, H_inv_cho = w_orig, H_inv_cho_orig
                # iterate over columns
                for c1 in range(0, d_col, block_size):
                    c2 = min(c1 + block_size, d_col)
                    ncols = c2 - c1  # number of columns
                    w_blk = w[:, c1:c2].clone()  # column-wise weight slice
                    res = torch.zeros_like(w_blk)
                    errs = torch.zeros_like(w_blk)
                    losses_blk = torch.zeros_like(w_blk)
                    H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
                    # 1) score computation
                    scores = w_blk**2 / H_inv_cho_blk.diag().reshape(1, -1) ** 2
                    thr, _ = torch.kthvalue(scores.view(-1), round(w_blk.numel() * sparsity))
                    mask = scores > thr
                    # 2) iterate over block
                    for i in range(ncols):
                        w_ci = w_blk[:, i]
                        d = H_inv_cho_blk[i, i]

                        q = w_ci.clone()
                        q[~mask[:, i]] = 0

                        res[:, i] = q
                        err = (w_ci - q) / d
                        losses_blk[:, i] = err**2

                        w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                        errs[:, i] = err
                    # 3) update the weights after block
                    w[:, c1:c2] = res
                    w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)
                # add current weight to the list
                sparse_weights.append(w.to(device=device, dtype=dtype))
        # init placeholders on other workers
        else:
            sparse_weights = [torch.empty_like(self.W, device=device, dtype=dtype) for _ in sparsities]

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            for i, _ in enumerate(sparsities):
                dist.broadcast(sparse_weights[i], src=0)

        return sparse_weights

    def prune(self, sparsities: List[float]) -> List[Tensor]:
        self.pruning_pre_step()
        sparse_weights = self.step(sparsities)
        return sparse_weights

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
        # invert
        H = linalg_utils.inv_sym(H)
        H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w, H_inv_cho
