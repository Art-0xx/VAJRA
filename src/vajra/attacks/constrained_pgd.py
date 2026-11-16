# Constrained PGD Attack for Realistic Adversarial Evasion (Blueprint Section 7.2)
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConstrainedPGDResult:
    """Outcome of realistic constrained PGD optimization."""
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    clean_logits: torch.Tensor
    adversarial_logits: torch.Tensor
    epsilon: float
    num_iter: int
    controllable_mask: torch.Tensor
    n_controllable_features: int


# List of features that a realistic network attacker CAN manipulate via
# padding, packet fragmentation, transmission timing, and TCP options.
CONTROLLABLE_PATTERNS = [
    "Fwd Packet Length",
    "Total Length of Fwd Packets",
    "Total Fwd Packets",
    "Fwd IAT",
    "Flow IAT",
    "Fwd Header Length",
    "Fwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Min Packet Length",
    "Max Packet Length",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "FIN Flag",
    "SYN Flag",
    "RST Flag",
    "PSH Flag",
    "ACK Flag",
    "URG Flag",
    "ECE Flag",
    "Init_Win_bytes_forward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Max",
    "Idle Min",
]

# Non-controllable features (dictated by the victim server response or routing)
NON_CONTROLLABLE_EXCLUSIONS = [
    "Destination Port",  # Fixed by the target service being attacked
    "Bwd Packet Length",
    "Total Length of Bwd Packets",
    "Total Backward Packets",
    "Bwd IAT",
    "Bwd Header Length",
    "Bwd Packets/s",
    "Init_Win_bytes_backward",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Bwd Avg",
]


def build_controllable_mask(feature_names: List[str], device: torch.device) -> torch.Tensor:
    """Build binary gradient mask: 1.0 for controllable features, 0.0 for non-controllable."""
    mask = torch.zeros(len(feature_names), dtype=torch.float32, device=device)
    for i, name in enumerate(feature_names):
        is_controllable = any(pat.lower() in name.lower() for pat in CONTROLLABLE_PATTERNS)
        is_excluded = any(ex.lower() in name.lower() for ex in NON_CONTROLLABLE_EXCLUSIONS)
        if is_controllable and not is_excluded:
            mask[i] = 1.0
    return mask


def constrained_pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    feature_names: List[str],
    alpha: Optional[float] = None,
    num_iter: int = 10,
    criterion: Optional[nn.Module] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
) -> ConstrainedPGDResult:
    """Projected Gradient Descent constrained strictly to attacker-controllable features.
    
    Adheres to Blueprint Section 7.2:
    - Masks gradients on victim-controlled and protocol-immutable features to 0.0.
    - Perturbs only forward payload lengths, timing, and client TCP options.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if alpha is None:
        alpha = epsilon / 4.0

    was_training = model.training
    model.eval()
    device = x.device
    x = x.detach().clone()

    mask = build_controllable_mask(feature_names, device=device).unsqueeze(0)
    n_controllable = int(mask.sum().item())

    with torch.no_grad():
        clean_logits = model(x)

    # Random start within L-infinity ball only for controllable features
    if random_start:
        delta = (torch.rand_like(x) * 2.0 - 1.0) * epsilon
        delta = delta * mask
        x_adv = (x + delta).clamp(clip_min, clip_max)
    else:
        x_adv = x.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)

        with torch.enable_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert x_adv.grad is not None
            grad = x_adv.grad.detach()

        # Apply constraint mask: freeze non-controllable gradients to zero
        masked_grad = grad * mask

        # PGD step with sign of masked gradient
        x_adv = x_adv + alpha * masked_grad.sign()

        # Project back to L-infinity ball around clean sample x
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        # Ensure only controllable features received any change
        eta = eta * mask
        x_adv = torch.clamp(x + eta, min=clip_min, max=clip_max)

    with torch.no_grad():
        adv_logits = model(x_adv)

    if was_training:
        model.train()

    return ConstrainedPGDResult(
        adversarial=x_adv.detach(),
        perturbation=(x_adv - x).detach(),
        clean_logits=clean_logits.detach(),
        adversarial_logits=adv_logits.detach(),
        epsilon=epsilon,
        num_iter=num_iter,
        controllable_mask=mask,
        n_controllable_features=n_controllable,
    )

