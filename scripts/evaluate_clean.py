# Evaluate the trained model on the clean test set (Table A).
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from vajra.config import load_config
from vajra.data.dataset import NIDSDataset
from vajra.models.registry import get_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_transformer.yaml")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline_model.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    config = load_config(args.config)
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}")

    preprocessor_path = config.get("paths.preprocessor", "artifacts/preprocessor.joblib")
    ds = NIDSDataset("data/processed/test.parquet", preprocessor_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(config, ds.n_features).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # FPR at 99% recall operating point.
    threshold_idx = int(0.99 * len(probs))
    threshold = float(np.sort(probs)[::-1][threshold_idx]) if threshold_idx < len(probs) else 0.5
    preds_at_thresh = (probs >= threshold).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(labels, preds_at_thresh).ravel()
    fpr_at_99_recall = float(fp2 / (fp2 + tn2)) if (fp2 + tn2) > 0 else 0.0

    metrics = {
        "n_test": int(len(labels)),
        "n_features": int(ds.n_features),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_binary": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr_at_99_recall": fpr_at_99_recall,
    }

    out = Path("artifacts/metrics/table_A_clean.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print("\n=== CLEAN TEST PERFORMANCE (Table A) ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v}")
    print(f"  confusion_matrix    : {metrics['confusion_matrix']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()