# Evaluate the trained model on the clean test set (Table A).
#
# Computes two sets of metrics:
#   1. At the default 0.5 threshold
#   2. At the F1-optimal threshold (tuned on validation set)
#
# Threshold tuning is standard for class-weighted models: the weighted loss
# shifts the decision boundary, so 0.5 is not the F1-maximizing cutoff.
# ROC-AUC measures separability independent of threshold.
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


def _inference(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_binary": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _find_best_threshold(labels, probs, grid=np.linspace(0.05, 0.99, 95)):
    best_f1, best_t = 0.0, 0.5
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _fpr_at_recall(labels, probs, target_recall=0.99):
    pos_probs = probs[labels == 1]
    if len(pos_probs) == 0:
        return float("nan"), float("nan")
    threshold = float(np.sort(pos_probs)[int((1 - target_recall) * len(pos_probs))])
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return fpr, threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_transformer.yaml")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline_model.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--val-split", default="data/processed/val.parquet")
    ap.add_argument("--test-split", default="data/processed/test.parquet")
    args = ap.parse_args()

    config = load_config(args.config)
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}")

    preprocessor_path = config.get("paths.preprocessor", "artifacts/preprocessor.joblib")

    # Load validation set for threshold tuning
    print(f"Loading validation set for threshold tuning: {args.val_split}")
    val_ds = NIDSDataset(args.val_split, preprocessor_path)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Load test set for reporting
    print(f"Loading test set: {args.test_split}")
    test_ds = NIDSDataset(args.test_split, preprocessor_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(config, test_ds.n_features).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Run inference on both splits
    print("Running inference on validation set...")
    val_probs, val_labels = _inference(model, val_loader, device)
    print("Running inference on test set...")
    test_probs, test_labels = _inference(model, test_loader, device)

    # Find F1-optimal threshold on validation set
    best_threshold, best_val_f1 = _find_best_threshold(val_labels, val_probs)
    print(f"\nOptimal threshold (from val): {best_threshold:.3f}  (val F1={best_val_f1:.4f})")

    # Compute metrics on test set at both thresholds
    default_metrics = _metrics_at_threshold(test_labels, test_probs, 0.5)
    tuned_metrics = _metrics_at_threshold(test_labels, test_probs, best_threshold)
    roc_auc = float(roc_auc_score(test_labels, test_probs))
    fpr_99, thr_99 = _fpr_at_recall(test_labels, test_probs, target_recall=0.99)

    # Primary metrics = tuned threshold (this is what the API will use)
    metrics = {
        "n_test": int(len(test_labels)),
        "n_features": int(test_ds.n_features),
        "optimal_threshold": best_threshold,
        "accuracy": tuned_metrics["accuracy"],
        "f1_binary": tuned_metrics["f1_binary"],
        "precision": tuned_metrics["precision"],
        "recall": tuned_metrics["recall"],
        "roc_auc": roc_auc,
        "fpr_at_99_recall": fpr_99,
        "threshold_for_99_recall": thr_99,
        "confusion_matrix": tuned_metrics["confusion_matrix"],
        "at_default_threshold_0.5": {
            "f1_binary": default_metrics["f1_binary"],
            "precision": default_metrics["precision"],
            "recall": default_metrics["recall"],
            "confusion_matrix": default_metrics["confusion_matrix"],
        },
    }

    out = Path("artifacts/metrics/table_A_clean.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print("\n=== CLEAN TEST PERFORMANCE (Table A) ===")
    print(f"  optimal_threshold   : {best_threshold:.3f}  (from validation set)")
    print(f"  accuracy            : {metrics['accuracy']:.4f}")
    print(f"  f1_binary           : {metrics['f1_binary']:.4f}    <- at tuned threshold")
    print(f"  precision           : {metrics['precision']:.4f}")
    print(f"  recall              : {metrics['recall']:.4f}")
    print(f"  roc_auc             : {metrics['roc_auc']:.4f}")
    print(f"  fpr_at_99_recall    : {metrics['fpr_at_99_recall']:.4f}    (threshold={thr_99:.3f})")
    print(f"  confusion_matrix    : {metrics['confusion_matrix']}")
    print(f"\n  (for reference, at default threshold 0.5)")
    print(f"    f1_binary         : {default_metrics['f1_binary']:.4f}")
    print(f"    precision         : {default_metrics['precision']:.4f}")
    print(f"    recall            : {default_metrics['recall']:.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()