import torch
from torch import Tensor


__all__ = ["inv_sym"]


def inv_sym(X: Tensor):
    """
    More efficient and stable inversion of symmetric matrices.
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(X))
