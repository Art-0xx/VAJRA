# Projected Gradient Descent (PGD) attack for VAJRA.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PGDResult:
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    clean_logits: torch.Tensor
    adversarial_logits: torch.Tensor
    epsilon: float
    num_iter: int


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float | None = None,
    num_iter: int = 10,
    criterion: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
) -> PGDResult:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if alpha is None:
        alpha = epsilon / 4.0

    was_training = model.training
    model.eval()
    x = x.detach().clone()

    with torch.no_grad():
        clean_logits = model(x)

    if random_start:
        delta = (torch.rand_like(x) * 2 - 1) * epsilon
        x_adv = (x + delta).clamp(clip_min, clip_max)
    else:
        x_adv = x.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach()
        with torch.enable_grad():
            x_req = x_adv.requires_grad_(True)
            logits = model(x_req)
            loss = criterion(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert x_req.grad is not None
            grad = x_req.grad.detach()

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = (x_adv - x).clamp(-epsilon, epsilon)
            x_adv = (x + delta).clamp(clip_min, clip_max)

    with torch.no_grad():
        adv_logits = model(x_adv)

    if was_training:
        model.train()

    return PGDResult(
        adversarial=x_adv.detach(),
        perturbation=(x_adv - x).detach(),
        clean_logits=clean_logits.detach(),
        adversarial_logits=adv_logits.detach(),
        epsilon=epsilon,
        num_iter=num_iter,
    )