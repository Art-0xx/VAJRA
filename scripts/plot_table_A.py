# Generate Table A figures (Confusion Matrix and ROC Curve) for the thesis.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from vajra.config import load_config
from vajra.data.dataset import NIDSDataset
from vajra.models.registry import get_model

FIG_DIR = Path("artifacts/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _inference(model, loader, device):
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def plot_confusion_matrix(cm, threshold, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_roc(labels, probs, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2.5, color="darkorange", label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", lw=1.5, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (Clean Test Set)", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_path = config.get("paths.preprocessor", "artifacts/preprocessor.joblib")

    ds = NIDSDataset("data/processed/test.parquet", pre_path)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    model = get_model(config, ds.n_features).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    probs, labels = _inference(model, loader, device)

    metrics = json.loads(Path("artifacts/metrics/table_A_clean.json").read_text())
    cm = metrics["confusion_matrix"]
    cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    threshold = metrics["optimal_threshold"]

    plot_confusion_matrix(cm_array, threshold, FIG_DIR / "confusion_matrix_clean.png")
    plot_roc(labels, probs, FIG_DIR / "roc_curve_clean.png")

    print(f"Saved: {FIG_DIR / 'confusion_matrix_clean.png'}")
    print(f"Saved: {FIG_DIR / 'roc_curve_clean.png'}")

if __name__ == "__main__":
    main()