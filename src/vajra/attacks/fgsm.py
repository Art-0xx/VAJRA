# Fast Gradient Sign Method (FGSM) attack for VAJRA.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FGSMResult:
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    clean_logits: torch.Tensor
    adversarial_logits: torch.Tensor
    epsilon: float


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> FGSMResult:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()
    x = x.detach().clone()

    # Force grad on — works even if caller is inside torch.no_grad().
    with torch.enable_grad():
        x_req = x.requires_grad_(True)
        logits = model(x_req)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert x_req.grad is not None
        grad = x_req.grad.detach()

    with torch.no_grad():
        perturbation = epsilon * grad.sign()
        x_adv = (x + perturbation).clamp(clip_min, clip_max)
        adv_logits = model(x_adv)

    if was_training:
        model.train()

    return FGSMResult(
        adversarial=x_adv.detach(),
        perturbation=(x_adv - x).detach(),
        clean_logits=logits.detach(),
        adversarial_logits=adv_logits.detach(),
        epsilon=epsilon,
    )


def attack_success_rate(
    clean_probs: torch.Tensor,
    adv_probs: torch.Tensor,
    threshold: float,
    true_labels: torch.Tensor,
) -> float:
    mask = (true_labels == 1) & (clean_probs > threshold)
    if mask.sum() == 0:
        return 0.0
    evaded = (adv_probs[mask] <= threshold).sum().item()
    return evaded / mask.sum().item()