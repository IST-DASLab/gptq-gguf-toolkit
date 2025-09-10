from types import MethodType
from typing import Iterable

import torch
from torch.optim import Optimizer


__all__ = ["wrap_optimizer", "unwrap_optimizer"]


# wrap optimizer
def wrap_optimizer(optimizer, params: Iterable[torch.Tensor], param_masks: Iterable[torch.Tensor]):
    optimizer.__step = optimizer.step

    def step(self, *args, **kwargs):
        # 1) apply original optimizer step
        self.__step(*args, **kwargs)
        # 2) mask updates
        with torch.no_grad():
            for param, param_mask in zip(params, param_masks):
                param.data.mul_(param_mask)

    optimizer.step = MethodType(step, optimizer)
    return optimizer


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    optimizer.step = optimizer.__step
    return optimizer
