import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from vajra.config import load_config
from vajra.data.dataset import NIDSDataset
from vajra.models.registry import get_model
from vajra.defenses.pgd_at import adversarial_train_step

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
    return f1_score(labels, preds, average="binary", zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_transformer.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pre_path = config.get("paths.preprocessor", "artifacts/preprocessor.joblib")
    train_ds = NIDSDataset("data/processed/train.parquet", pre_path)
    val_ds = NIDSDataset("data/processed/val.parquet", pre_path)
    
    batch_size = config.get("training.batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_model(config, train_ds.n_features).to(device)
    
    labels = train_ds.df["binary_label"].to_numpy()
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor([len(labels) / (2 * c) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    epochs = 15  # Adversarial training converges differently
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    checkpoint_path = Path("artifacts/checkpoints/defended_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = adversarial_train_step(model, x, y, optimizer, criterion, device)
            running_loss += loss * y.size(0)
            n += y.size(0)

        scheduler.step()
        val_f1 = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs} | train_loss {running_loss/n:.4f} | val_f1 {val_f1:.4f} | {elapsed:.0f}s")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best checkpoint (val_f1={best_f1:.4f})")

    print(f"\nDone. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()