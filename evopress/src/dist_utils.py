from typing import Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist


__all__ = [
    "is_dist_available_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main",
    "broadcast_parameters",
    "gather_into_tensor",
    "print_on_main",
]


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def is_main():
    return get_rank() == 0


def broadcast_parameters(module: nn.Module, src: Any = 0, group: Optional[Any] = None):
    for param in module.parameters():
        dist.broadcast(param.data, src=src, group=group)


def gather_into_tensor(tensor, dim: int = 0):
    world_size = get_world_size()
    if is_main():
        gathered_shape = (*tensor.shape[:dim], world_size * tensor.shape[dim], *tensor.shape[dim + 1 :])
        gathered_tensor = torch.empty(gathered_shape, device=tensor.device, dtype=tensor.dtype)
        gathered_tensor_chunks = list(gathered_tensor.chunk(world_size, dim=dim))
    else:
        gathered_tensor = None
        gathered_tensor_chunks = None
    dist.gather(tensor, gathered_tensor_chunks)
    return gathered_tensor


def print_on_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)
