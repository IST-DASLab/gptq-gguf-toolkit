from typing import Dict, Tuple

import torch


def square_head_loss(
    teacher_features: Dict[str, torch.Tensor], student_features: Dict[str, torch.Tensor], eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    feature_losses = {}
    for (t_k, t_v), (s_k, s_v) in zip(teacher_features.items(), student_features.items()):
        feature_losses[t_k] = (s_v - t_v).pow(2).mean() / (t_v.pow(2).mean() + eps)
    loss = sum([v for _, v in feature_losses.items()])
    return loss, feature_losses
