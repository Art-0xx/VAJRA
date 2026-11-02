# Fast Gradient Sign Method (FGSM) attack for VAJRA.
#
# White-box L-infinity attack. Perturbs each feature by at most epsilon
# in the direction that increases the loss on the true label. Because
# features are MinMax-scaled to [0, 1], the perturbation is clipped to
# that range after each step.
#
# Reference: Goodfellow, Shlens, Szegedy. "Explaining and Harnessing
# Adversarial Examples." ICLR 2015.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FGSMResult:
    """Output of a single FGSM call."""
    adversarial: torch.Tensor         # (batch, n_features) perturbed input
    perturbation: torch.Tensor        # (batch, n_features) = adversarial - clean
    clean_logits: torch.Tensor        # (batch, n_classes)
    adversarial_logits: torch.Tensor  # (batch, n_classes)
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
    """Generate FGSM adversarial examples.

    Args:
        model: trained classifier, will be placed in eval() mode.
        x: clean input, shape (batch, n_features), already normalized.
        y: ground-truth labels, shape (batch,). Should be the TRUE
           labels (not the model's predictions) — FGSM maximizes the
           loss on the true label, which is the standard formulation.
        epsilon: L-infinity perturbation budget.
        criterion: loss function. Defaults to CrossEntropyLoss.
        clip_min, clip_max: valid feature range after perturbation.

    Returns:
        FGSMResult containing the adversarial batch, the perturbation,
        and clean/adversarial logits for downstream analysis.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    x = x.detach().clone().requires_grad_(True)

    # Forward + backward to get the input gradient
    logits = model(x)
    loss = criterion(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # FGSM step: move x in the direction of the sign of the gradient
    assert x.grad is not None, "Gradient not computed — is the model on the right device?"
    with torch.no_grad():
        perturbation = epsilon * x.grad.sign()
        x_adv = (x + perturbation).clamp(clip_min, clip_max)

    with torch.no_grad():
        adv_logits = model(x_adv)

    return FGSMResult(
        adversarial=x_adv.detach(),
        perturbation=(x_adv - x.detach()).detach(),
        clean_logits=logits.detach(),
        adversarial_logits=adv_logits,
        epsilon=epsilon,
    )


def attack_success_rate(
    clean_probs: torch.Tensor,
    adv_probs: torch.Tensor,
    threshold: float,
    true_labels: torch.Tensor,
) -> float:
    """Fraction of true-attack samples that stop being detected after attack.

    A sample counts as "correctly detected clean" if clean_probs > threshold
    (assuming probs is P(attack) and true_labels is 1). It counts as
    "evaded" if adv_probs <= threshold.
    """
    mask = (true_labels == 1) & (clean_probs > threshold)
    if mask.sum() == 0:
        return 0.0
    evaded = (adv_probs[mask] <= threshold).sum().item()
    return evaded / mask.sum().item()