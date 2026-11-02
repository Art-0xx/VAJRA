# Single-flow FGSM demo for VAJRA.
#
# Loads the trained baseline model, samples correctly-classified attack
# flows from the test set, attacks them with FGSM at several epsilon
# values, and reports how many stop being detected.
#
# This is the first concrete VAJRA finding: a tiny perturbation makes
# the NIDS miss attacks it was correctly catching a moment ago.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from vajra.attacks.fgsm import fgsm_attack
from vajra.config import load_config
from vajra.models.registry import get_model


def _predict_probs(model, x, device):
    with torch.no_grad():
        logits = model(x.to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu()


def main() -> None:
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load preprocessor + model
    pre = joblib.load("artifacts/preprocessor.joblib")
    model = get_model(config, len(pre.feature_columns)).to(device)
    model.load_state_dict(torch.load(
        "artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    # Load tuned threshold from Table A
    metrics_path = Path("artifacts/metrics/table_A_clean.json")
    if metrics_path.exists():
        threshold = float(json.loads(metrics_path.read_text())["optimal_threshold"])
    else:
        threshold = 0.880
    print(f"Decision threshold: {threshold:.3f}")

    # Sample 200 attack flows from the test set
    df = pd.read_parquet("data/processed/test.parquet")
    attack_df = df[df["binary_label"] == 1].sample(n=200, random_state=42)
    X = pre.transform(attack_df).astype(np.float32)
    x = torch.from_numpy(X).to(device)
    y = torch.ones(len(X), dtype=torch.long, device=device)

    # Filter to flows the model correctly detects at threshold
    clean_probs = _predict_probs(model, x, device)
    detected_mask = clean_probs > threshold
    x_detected = x[detected_mask]
    y_detected = y[detected_mask]
    n_detected = int(detected_mask.sum().item())
    print(f"\nSample: {len(x)} attack flows, {n_detected} correctly detected")
    print(f"Mean P(attack) on clean correctly-detected: "
          f"{clean_probs[detected_mask].mean().item():.4f}\n")

    if n_detected == 0:
        print("No correctly-detected samples to attack; abort.")
        return

    # Attack at each epsilon
    print(f"{'eps':>6} | {'mean P(attack)':>15} | {'detected':>9} | {'ASR':>6}")
    print("-" * 50)
    for eps in [0.005, 0.01, 0.03, 0.05, 0.10]:
        result = fgsm_attack(model, x_detected, y_detected, epsilon=eps)
        adv_probs = torch.softmax(result.adversarial_logits, dim=1)[:, 1].cpu()
        still_detected = (adv_probs > threshold).sum().item()
        asr = 1.0 - (still_detected / n_detected)
        print(
            f"{eps:>6.3f} | {adv_probs.mean().item():>15.4f} | "
            f"{still_detected:>4}/{n_detected:<4} | {asr*100:>4.0f}%"
        )

    print(f"\nThreshold for detection is {threshold:.3f}.")
    print("ASR = fraction of originally-detected attacks that become undetected.")


if __name__ == "__main__":
    main()