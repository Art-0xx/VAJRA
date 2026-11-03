# Transfer attack: train a surrogate CNN, craft adversarial examples on it,
# evaluate them against the target Transformer.
#
# Threat model: the attacker does not have access to the target model but
# can train a surrogate on the same data distribution. Perturbations that
# fool the surrogate are tested against the target (transfer ASR).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vajra.attacks.pgd import pgd_attack


class SurrogateCNN(nn.Module):
    """Small 1D-CNN surrogate for transfer attack generation.

    Deliberately a different architecture from the Transformer target so
    that any successful transfer is a genuine cross-architecture result,
    not a self-transfer artifact.
    """

    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, F) -> (B, 1, F)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


@dataclass
class SurrogateTrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    patience: int = 5


def train_surrogate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    device: torch.device,
    config: SurrogateTrainConfig | None = None,
    class_weights: torch.Tensor | None = None,
) -> SurrogateCNN:
    if config is None:
        config = SurrogateTrainConfig()

    model = SurrogateCNN(n_features).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                p = model(x).argmax(1).cpu().numpy()
                preds.append(p)
                labels.append(y.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"  surrogate epoch {epoch:02d} | val_f1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def load_or_train_surrogate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    device: torch.device,
    checkpoint_path: str | Path,
    class_weights: torch.Tensor | None = None,
) -> SurrogateCNN:
    checkpoint_path = Path(checkpoint_path)
    model = SurrogateCNN(n_features).to(device)

    if checkpoint_path.exists():
        print(f"Loading surrogate from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model

    print(f"Training surrogate (no checkpoint at {checkpoint_path})")
    model = train_surrogate(
        train_loader, val_loader, n_features, device, class_weights=class_weights
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved surrogate to {checkpoint_path}")
    return model


def transfer_attack(
    surrogate: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_iter: int = 10,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Craft adversarial examples on the surrogate, return them for evaluation.

    Note: this returns the adversarial *inputs*, not results. The caller
    evaluates them against the target model.
    """
    result = pgd_attack(
        surrogate, x, y,
        epsilon=epsilon,
        num_iter=num_iter,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return result.adversarial