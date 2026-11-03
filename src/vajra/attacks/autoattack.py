# APGD attack (AutoAttack core) for VAJRA.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class APGDResult:
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    clean_logits: torch.Tensor
    adversarial_logits: torch.Tensor
    epsilon: float
    num_iter: int
    converged_iter: int


def apgd_ce_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_iter: int = 100,
    criterion: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
    momentum: float = 0.75,
    step_size_reduction_iters: int = 10,
) -> APGDResult:
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="none")

    was_training = model.training
    model.eval()
    x = x.detach().clone()

    with torch.no_grad():
        clean_logits = model(x)

    if random_start:
        delta = (torch.rand_like(x) * 2 - 1) * epsilon
        x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    else:
        x_adv = x.clone().detach()

    x_best = x_adv.clone()
    loss_best = torch.full((x.size(0),), -float("inf"), device=x.device)
    x_prev = x_adv.clone()
    step_size = torch.full((x.size(0),), 2.0 * epsilon, device=x.device)
    patience = torch.zeros(x.size(0), device=x.device)
    converged_iter = num_iter

    for k in range(num_iter):
        with torch.enable_grad():
            z = (x_adv + momentum * (x_adv - x_prev)).detach().requires_grad_(True)
            logits = model(z)
            loss = criterion(logits, y).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert z.grad is not None
            grad = z.grad.detach()

        with torch.no_grad():
            grad_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
            grad_normed = grad / grad_norm.view(-1, 1)

            x_new = x_adv + step_size.view(-1, 1) * grad_normed
            delta = (x_new - x).clamp(-epsilon, epsilon)
            x_new = (x + delta).clamp(clip_min, clip_max).detach()

            new_logits = model(x_new)
            new_loss = criterion(new_logits, y).detach()

            better = new_loss > loss_best
            loss_best = torch.where(better, new_loss, loss_best)
            x_best[better] = x_new[better]

            patience = torch.where(better, torch.zeros_like(patience), patience + 1)
            reduce_mask = patience >= step_size_reduction_iters
            if reduce_mask.any():
                step_size = torch.where(reduce_mask, step_size / 2, step_size)
                patience = torch.where(reduce_mask, torch.zeros_like(patience), patience)

            x_prev = x_adv
            x_adv = x_new

        if (step_size < 0.01 * epsilon).all():
            converged_iter = k + 1
            break

    with torch.no_grad():
        adv_logits = model(x_best)

    if was_training:
        model.train()

    return APGDResult(
        adversarial=x_best.detach(),
        perturbation=(x_best - x).detach(),
        clean_logits=clean_logits.detach(),
        adversarial_logits=adv_logits.detach(),
        epsilon=epsilon,
        num_iter=num_iter,
        converged_iter=converged_iter,
    )