# Unified attack evaluation harness for VAJRA.
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


AttackName = Literal["fgsm", "pgd", "apgd", "transfer", "noise"]


@dataclass
class AttackMetrics:
    attack: str
    epsilon: float
    n_total: int
    n_attacked: int
    n_successful: int
    asr: float
    clean_f1: float
    robust_f1: float
    delta_f1: float
    clean_accuracy: float
    robust_accuracy: float


def _predict_probs(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return P(attack) as a CPU tensor for metric computation."""
    with torch.no_grad():
        logits = model(x.to(device))
    return torch.softmax(logits, dim=1)[:, 1].cpu()


def evaluate_attack(
    model: nn.Module,
    loader: DataLoader,
    attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    attack_name: str,
    epsilon: float,
    threshold: float,
    device: torch.device,
    max_samples: int | None = None,
) -> AttackMetrics:
    model.eval()

    all_clean_preds, all_adv_preds, all_labels = [], [], []
    n_attacked = 0
    n_successful = 0
    n_total = 0

    for x, y in loader:
        if max_samples is not None and n_total >= max_samples:
            break

        x_dev = x.to(device)
        y_dev = y.to(device)

        # Clean probabilities — CPU tensor, dtype float32.
        clean_probs = _predict_probs(model, x_dev, device)

        # Attack with grad enabled (works even if caller wrapped in no_grad).
        with torch.enable_grad():
            x_adv = attack_fn(model, x_dev, y_dev).detach()

        adv_probs = _predict_probs(model, x_adv, device)

        # Move y to CPU — probs are already CPU, so mask stays on one device.
        y_cpu = y.cpu()

        clean_preds = (clean_probs > threshold).long()
        adv_preds = (adv_probs > threshold).long()

        all_clean_preds.append(clean_preds.numpy())
        all_adv_preds.append(adv_preds.numpy())
        all_labels.append(y_cpu.numpy())

        # ASR on the subset of true-attack samples the clean model detects.
        mask = (y_cpu == 1) & (clean_probs > threshold)
        n_attacked += int(mask.sum().item())
        flipped = (adv_probs[mask] <= threshold).sum().item()
        n_successful += int(flipped)

        n_total += x.size(0)

    clean_preds = np.concatenate(all_clean_preds)
    adv_preds = np.concatenate(all_adv_preds)
    labels = np.concatenate(all_labels)

    clean_f1 = float(f1_score(labels, clean_preds, zero_division=0))
    robust_f1 = float(f1_score(labels, adv_preds, zero_division=0))
    clean_acc = float((clean_preds == labels).mean())
    robust_acc = float((adv_preds == labels).mean())
    asr = n_successful / n_attacked if n_attacked > 0 else 0.0

    return AttackMetrics(
        attack=attack_name,
        epsilon=epsilon,
        n_total=n_total,
        n_attacked=n_attacked,
        n_successful=n_successful,
        asr=asr,
        clean_f1=clean_f1,
        robust_f1=robust_f1,
        delta_f1=clean_f1 - robust_f1,
        clean_accuracy=clean_acc,
        robust_accuracy=robust_acc,
    )


def metrics_to_dict(m: AttackMetrics) -> dict:
    return asdict(m)