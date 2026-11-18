#!/usr/bin/env python
# VAJRA - scripts/build_reference_profile.py
#
# Builds artifacts/reference_profile.npy: the per-PREDICTED-class mean
# attention-rollout attribution vector, computed over real CIC-IDS2017
# flows. This is what nids_console.py's divergence gate compares live
# flows against (see ModelHolder.ref_profiles / handle_completed_flow in
# nids_console.py). Without this file, the gate falls back to a uniform
# vector and every JSD reading is meaningless (the exact problem flagged
# in the project's own debug log, section 4).
#
# IMPORTANT: buckets are keyed by the model's OWN PREDICTED class
# (0=benign, 1=attack), not the CSV's ground-truth label. This matches
# exactly how handle_completed_flow() looks up ref_profiles[predicted_class]
# at runtime -- the reference must be built the same way it's consumed.
from __future__ import annotations

import sys
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

# How many real flows to sample per CSV file, per rough label bucket
# (benign-labeled vs attack-labeled in the CSV). This is a SAMPLING target
# for diversity across files, not the final predicted-class bucket -- the
# actual bucketing happens after the model scores each flow.
SAMPLES_PER_FILE_PER_BUCKET = 150

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
    """Loads a diverse sample of real flows across all available CIC-IDS2017
    CSVs, split roughly by benign-vs-attack ground-truth label purely to
    ensure both kinds of traffic are represented in the sample -- the
    model's own prediction (not this label) decides the final bucket."""
    frames = []
    needed_cols = feature_columns + ["Label"]

    for fname in CSV_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[!] Skipping missing file: {fpath}")
            continue

        print(f"[*] Reading {fname} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"[!] Failed to read {fname}: {e}")
            continue

        df.columns = df.columns.str.strip()
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            print(f"[!] {fname} missing columns {missing[:5]}{'...' if len(missing) > 5 else ''} — skipping file")
            continue

        is_benign = df["Label"].astype(str).str.upper().str.strip() == "BENIGN"
        benign_rows = df[is_benign]
        attack_rows = df[~is_benign]

        n_benign = min(SAMPLES_PER_FILE_PER_BUCKET, len(benign_rows))
        n_attack = min(SAMPLES_PER_FILE_PER_BUCKET, len(attack_rows))

        if n_benign > 0:
            frames.append(benign_rows.sample(n=n_benign, random_state=42))
        if n_attack > 0:
            frames.append(attack_rows.sample(n=n_attack, random_state=42))

        print(f"    sampled {n_benign} benign-labeled + {n_attack} attack-labeled rows")

    if not frames:
        raise RuntimeError(
            "No usable rows loaded from any CSV. Check DATA_DIR and CSV_FILES "
            "paths in this script against your actual data/raw/MachineLearningCVE contents."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


def drop_non_finite(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Drops rows with inf/NaN in any feature column. The real CIC-IDS2017
    CSVs contain inf values (e.g. Flow Bytes/s when duration rounds to 0 —
    confirmed in this project's own exploratory stats on DoS Hulk rows).
    Silently feeding these through preprocessor.transform() can corrupt the
    profile with inf/NaN contributions."""
    sub = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(sub.to_numpy()).all(axis=1)
    n_dropped = (~finite_mask).sum()
    if n_dropped > 0:
        print(f"[*] Dropping {n_dropped} rows with non-finite feature values")
    return df[finite_mask].reset_index(drop=True)


def main():
    default_cfg = load_default_config()
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = len(pre.feature_columns)
    print(f"[*] Loading baseline_model.pt on {str(device).upper()} ({num_features} features)...")
    model = get_model(config, num_features, store_attention=True).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    print("[*] Sampling real flows from CIC-IDS2017 CSVs...")
    df = load_sampled_rows(pre.feature_columns)
    df = drop_non_finite(df, pre.feature_columns)
    print(f"[*] Total usable rows after cleaning: {len(df)}")

    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    skipped = 0

    print("[*] Scoring flows and accumulating attention attribution per predicted class...")
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

            if predicted_class not in sums:
                sums[predicted_class] = np.zeros_like(attr, dtype=np.float64)
                counts[predicted_class] = 0
            sums[predicted_class] += attr
            counts[predicted_class] += 1

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"[!] Row {i} failed: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"    ...{i + 1}/{len(df)} rows processed "
                  f"(class0={counts.get(0, 0)}, class1={counts.get(1, 0)}, skipped={skipped})")

    print(f"[*] Done. class0={counts.get(0, 0)}, class1={counts.get(1, 0)}, skipped={skipped}")

    if 0 not in counts or 1 not in counts or counts[0] == 0 or counts[1] == 0:
        print("[!] WARNING: one or both predicted classes have zero samples. The reference "
              "profile will be incomplete. Check model calibration / sample size before "
              "proceeding — nids_console.py will fall back to a uniform vector for any "
              "missing class, same as the current no-file case.")

    profiles: Dict[int, np.ndarray] = {}
    for k, s in sums.items():
        v = s / max(counts[k], 1)
        total = v.sum()
        if total > 0:
            v = v / total  # normalize to a probability distribution for Jensen-Shannon divergence
        profiles[k] = v.astype(np.float32)

    out_path = Path("artifacts/reference_profile.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, profiles, allow_pickle=True)
    print(f"[+] Saved {out_path} — classes present: {list(profiles.keys())}, "
          f"counts: {dict(counts)}")


if __name__ == "__main__":
    main()