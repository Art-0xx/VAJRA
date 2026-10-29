# Train the VAJRA Transformer NIDS.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vajra.config import load_config
from vajra.data.dataset import NIDSDataset, build_loader
from vajra.models.registry import get_model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            n += y.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / n,
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "accuracy": float((preds == labels).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_transformer.yaml")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Train on first 5000 samples for 1 epoch")
    args = ap.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}")

    preprocessor_path = config.get("paths.preprocessor", "artifacts/preprocessor.joblib")
    batch_size = config.get("training.batch_size", 256)
    if args.smoke_test:
        batch_size = min(batch_size, 64)
        print(f"SMOKE TEST mode - batch_size={batch_size}, 1 epoch, 5000 samples")

    train_loader = build_loader(
        "data/processed/train.parquet", preprocessor_path,
        batch_size=batch_size, shuffle=True,
    )
    val_loader = build_loader(
        "data/processed/val.parquet", preprocessor_path,
        batch_size=batch_size, shuffle=False,
    )

    # Narrow the dataset type so Pylance sees NIDSDataset's attributes.
    train_ds = cast(NIDSDataset, train_loader.dataset)
    n_features = train_ds.n_features
    print(f"Features: {n_features}")

    model = get_model(config, n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    labels = train_ds.df["binary_label"].to_numpy()
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(
        [len(labels) / (2 * c) for c in counts], dtype=torch.float32
    ).to(device)
    print(f"Class weights: {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("training.lr", 1e-4),
        weight_decay=config.get("training.weight_decay", 1e-5),
    )
    epochs = 1 if args.smoke_test else config.get("training.epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]

    best_f1 = 0.0
    patience = config.get("training.early_stopping_patience", 5)
    patience_counter = 0
    checkpoint_path = Path(
        config.get("training.checkpoint_path",
                   "artifacts/checkpoints/baseline_model.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            n += y.size(0)

            if args.smoke_test and n >= 5000:
                break

        scheduler.step()
        val = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss {running_loss/n:.4f} | "
            f"val_loss {val['loss']:.4f} | "
            f"val_f1 {val['f1']:.4f} | "
            f"val_acc {val['accuracy']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best checkpoint (val_f1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    metrics = {
        "best_val_f1": best_f1,
        "n_features": n_features,
        "n_params": n_params,
        "device": str(device),
    }
    metrics_path = Path("artifacts/metrics/train_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nDone. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()