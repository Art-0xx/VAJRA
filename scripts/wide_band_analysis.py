# Wide-band constrained attack analysis.
#
# Restrict FGSM and noise to only the 43 features with 99% band >= 0.10.
# The other 22 constant features (99% band < 0.001) are excluded to
# eliminate the out-of-distribution artifact that inflates noise ASR.
#
# If FGSM still dominates noise on wide-band-only perturbations, the
# vulnerability is genuine, not a preprocessing artifact.
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vajra.config import load_config
from vajra.data.dataset import NIDSDataset
from vajra.models.registry import get_model


def main() -> None:
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pre = joblib.load("artifacts/preprocessor.joblib")
    model = get_model(config, len(pre.feature_columns)).to(device)
    model.load_state_dict(torch.load(
        "artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    # Wide-band mask (99% band >= 0.10)
    test_df = pd.read_parquet("data/processed/test.parquet")
    X_all = pre.transform(test_df)
    p99 = np.percentile(X_all, 99, axis=0)
    p01 = np.percentile(X_all, 1, axis=0)
    wide_mask_np = (p99 - p01) >= 0.10
    wide_mask = torch.tensor(wide_mask_np, dtype=torch.bool).to(device)
    print(f"Wide-band features: {wide_mask_np.sum()}/{len(wide_mask_np)}")

    # Sample 20K from test set for stability
    ds = NIDSDataset("data/processed/test.parquet", "artifacts/preprocessor.joblib")
    loader = DataLoader(ds, batch_size=5000, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        clean_p = torch.softmax(model(x), dim=1)[:, 1].cpu()
    mask = (y.cpu() == 1) & (clean_p > 0.88)
    n_detected = mask.sum().item()
    n_total_attack = int((y.cpu() == 1).sum())
    print(f"Detected clean attacks: {n_detected}/{n_total_attack}\n")

    print(f"{'eps':>8} | {'FGSM WB':>10} | {'Noise WB':>10} | {'ratio':>6} | {'FGSM all':>10} | {'Noise all':>10}")
    print("-" * 75)

    for eps in [0.005, 0.01, 0.05]:
        # --- Wide-band constrained FGSM ---
        x_req = x.clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x_req), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grad = x_req.grad
            pert = eps * grad.sign() * wide_mask.float()
            x_adv_wb = (x + pert).clamp(0, 1)
            adv_p_wb = torch.softmax(model(x_adv_wb), dim=1)[:, 1].cpu()
        asr_fgsm_wb = (adv_p_wb[mask] <= 0.88).float().mean().item()

        # --- Wide-band constrained noise ---
        torch.manual_seed(0)
        noise_wb = (torch.rand_like(x) * 2 - 1) * eps * wide_mask.float()
        x_noise_wb = (x + noise_wb).clamp(0, 1)
        with torch.no_grad():
            adv_p_noise_wb = torch.softmax(model(x_noise_wb), dim=1)[:, 1].cpu()
        asr_noise_wb = (adv_p_noise_wb[mask] <= 0.88).float().mean().item()

        # --- Unconstrained (all features) ---
        with torch.no_grad():
            pert_full = eps * grad.sign()
            x_adv_full = (x + pert_full).clamp(0, 1)
            adv_p_full = torch.softmax(model(x_adv_full), dim=1)[:, 1].cpu()
        asr_fgsm_full = (adv_p_full[mask] <= 0.88).float().mean().item()

        torch.manual_seed(0)
        noise_full = (torch.rand_like(x) * 2 - 1) * eps
        x_noise_full = (x + noise_full).clamp(0, 1)
        with torch.no_grad():
            adv_p_noise_full = torch.softmax(model(x_noise_full), dim=1)[:, 1].cpu()
        asr_noise_full = (adv_p_noise_full[mask] <= 0.88).float().mean().item()

        ratio = asr_fgsm_wb / max(asr_noise_wb, 1e-6)
        print(f"{eps:>8.3f} | {asr_fgsm_wb*100:>9.2f}% | {asr_noise_wb*100:>9.2f}% | "
              f"{ratio:>5.1f}x | {asr_fgsm_full*100:>9.2f}% | {asr_noise_full*100:>9.2f}%")

    print("\nKey metric: 'ratio' = FGSM ASR / Noise ASR on wide-band features.")
    print("If ratio > 2 at eps=0.05, the vulnerability is genuine (not OOD artifact).")


if __name__ == "__main__":
    main()