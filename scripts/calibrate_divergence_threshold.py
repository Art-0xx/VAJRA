#!/usr/bin/env python
# VAJRA - scripts/calibrate_divergence_threshold.py
#
# The JSD gate threshold (configs/default.yaml -> divergence.threshold) was
# a placeholder (0.050) never calibrated against this project's real
# reference profile. This script computes the actual JSD distribution for
# a large sample of real CIC-IDS2017 flows (reusing the same sampling
# approach as build_reference_profile.py) against the SAME reference
# profile nids_console.py loads at runtime, and reports percentiles so a
# threshold can be chosen from a target false-positive rate on clean
# traffic, instead of guessed.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from vajra.config import load_config
from vajra.console_helpers import load_default_config
from vajra.models.registry import get_model
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution
from vajra.justification.divergence import compute_divergence

SAMPLES_PER_FILE_PER_BUCKET = 100  # smaller than the profile-build sample -- this is a held-out-ish check
DATA_DIR = Path("data/raw/MachineLearningCVE")

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def load_sampled_rows(feature_columns: List[str]) -> pd.DataFrame:
    frames = []
    needed_cols = feature_columns + ["Label"]

    for fname in CSV_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue
        df.columns = df.columns.str.strip()
        if any(c not in df.columns for c in needed_cols):
            continue

        is_benign = df["Label"].astype(str).str.upper().str.strip() == "BENIGN"
        benign_rows, attack_rows = df[is_benign], df[~is_benign]
        n_b = min(SAMPLES_PER_FILE_PER_BUCKET, len(benign_rows))
        n_a = min(SAMPLES_PER_FILE_PER_BUCKET, len(attack_rows))
        if n_b > 0:
            frames.append(benign_rows.sample(n=n_b, random_state=7))
        if n_a > 0:
            frames.append(attack_rows.sample(n=n_a, random_state=7))
        print(f"[*] {fname}: sampled {n_b} benign-labeled + {n_a} attack-labeled")

    if not frames:
        raise RuntimeError("No usable rows loaded — check DATA_DIR/CSV_FILES.")
    return pd.concat(frames, ignore_index=True)


def drop_non_finite(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    sub = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(sub.to_numpy()).all(axis=1)
    return df[mask].reset_index(drop=True)


def main():
    default_cfg = load_default_config()
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(pre.feature_columns)

    ref_path = Path("artifacts/reference_profile.npy")
    if not ref_path.exists():
        raise RuntimeError(
            "artifacts/reference_profile.npy not found. Run scripts/build_reference_profile.py first — "
            "this calibration must use the SAME reference profile nids_console.py will load at runtime."
        )
    ref_profiles_raw = np.load(ref_path, allow_pickle=True).item()
    ref_profiles = {int(k): np.asarray(v, dtype=np.float32) for k, v in ref_profiles_raw.items()}
    print(f"[*] Loaded reference profile — classes: {list(ref_profiles.keys())}")

    print("[*] Loading baseline_model.pt ...")
    model = get_model(config, num_features, store_attention=True).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    print("[*] Sampling real flows for JSD calibration ...")
    df = load_sampled_rows(pre.feature_columns)
    df = drop_non_finite(df, pre.feature_columns)
    print(f"[*] Usable rows: {len(df)}")

    is_benign_true = df["Label"].astype(str).str.upper().str.strip() == "BENIGN"

    jsd_benign_true: List[float] = []   # JSD on rows whose TRUE label is benign
    jsd_attack_true: List[float] = []   # JSD on rows whose TRUE label is an attack
    skipped = 0

    for i, row in df.iterrows():
        try:
            feats = row[pre.feature_columns].to_dict()
            x_df = pd.DataFrame([feats])[pre.feature_columns]
            x = torch.tensor(pre.transform(x_df), dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                predicted_class = 1 if prob > 0.5 else 0
                maps = model.get_attention_maps()
                if not maps:
                    skipped += 1
                    continue
                rollout = compute_attention_rollout(maps)
                attr = get_feature_attribution(rollout).cpu().numpy().flatten()

            ref = ref_profiles.get(predicted_class, ref_profiles[0])
            jsd = compute_divergence(ref, attr)

            if is_benign_true.iloc[i]:
                jsd_benign_true.append(jsd)
            else:
                jsd_attack_true.append(jsd)

        except Exception as e:
            skipped += 1
            continue

        if (i + 1) % 200 == 0:
            print(f"    ...{i + 1}/{len(df)} processed (skipped={skipped})")

    jsd_benign_true = np.array(jsd_benign_true)
    jsd_attack_true = np.array(jsd_attack_true)

    print(f"\n[*] Done. benign-labeled rows scored: {len(jsd_benign_true)}, "
          f"attack-labeled rows scored: {len(jsd_attack_true)}, skipped: {skipped}\n")

    def describe(name: str, arr: np.ndarray):
        if len(arr) == 0:
            print(f"  {name}: NO DATA")
            return
        pct = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
        print(f"  {name} (n={len(arr)}): min={arr.min():.4f} "
              f"p5={pct[0]:.4f} p25={pct[1]:.4f} p50={pct[2]:.4f} p75={pct[3]:.4f} "
              f"p90={pct[4]:.4f} p95={pct[5]:.4f} p99={pct[6]:.4f} max={arr.max():.4f}")

    print("=== JSD distribution by TRUE label ===")
    describe("Benign-labeled", jsd_benign_true)
    describe("Attack-labeled", jsd_attack_true)

    if len(jsd_benign_true) > 0:
        print("\n=== Threshold suggestions (gate fires when JSD > threshold) ===")
        for target_fpr, pct in [(0.10, 90), (0.05, 95), (0.01, 99)]:
            thresh = np.percentile(jsd_benign_true, pct)
            print(f"  Target {target_fpr*100:.0f}% false-positive rate on benign-labeled flows "
                  f"-> threshold = {thresh:.4f} (p{pct} of benign JSD)")

    if len(jsd_benign_true) > 0 and len(jsd_attack_true) > 0:
        overlap_note = (
            "Benign and attack JSD ranges appear SEPARABLE at some threshold above."
            if np.percentile(jsd_attack_true, 25) > np.percentile(jsd_benign_true, 75)
            else "WARNING: benign and attack JSD distributions substantially OVERLAP. "
                 "This attention-rollout signal may not cleanly separate clean from "
                 "adversarial/attack traffic at any single threshold -- consider this "
                 "a real, reportable limitation of the divergence-gate approach for "
                 "this model, rather than a calibration failure."
        )
        print(f"\n[*] {overlap_note}")

    print("\n[*] Update configs/default.yaml -> divergence.threshold with the chosen value, "
          "or pass it explicitly if nids_console.py supports a CLI override.")


if __name__ == "__main__":
    main()