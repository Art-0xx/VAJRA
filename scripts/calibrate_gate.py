#!/usr/bin/env python
"""Offline calibration of the VAJRA JSD gate on real CIC-IDS2017 flows.

Mirrors nids_console.handle_completed_flow:
  predicted_class = 1 if P(attack) > 0.5 else 0
  jsd = compute_divergence(ref_profiles[predicted_class], attention_attribution)

Flows used to BUILD reference_profile.npy (random_state=42, 150/bucket/file,
same logic as build_reference_profile.py) are excluded. Note: they may still
have been in the model's own training split.

Run from /mnt/d/VAJRA, no sudo, consoles closed:
  python scripts/calibrate_gate.py
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from vajra.config import load_config
from vajra.console_helpers import load_default_config
from vajra.models.registry import get_model
from vajra.justification.divergence import compute_divergence
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

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
LIVE = {"benign campaign": 0.094, "Hulk (live)": 0.088, "SYN flows (live)": 0.109}


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank().to_numpy()
    n_p, n_n = len(pos), len(neg)
    return (r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


def load_heldout(feature_cols, per_bucket, seed):
    frames = []
    for fname in CSV_FILES:
        fp = DATA_DIR / fname
        if not fp.exists():
            print(f"[!] missing {fp}")
            continue
        df = pd.read_csv(fp, low_memory=False)
        df.columns = df.columns.str.strip()
        if any(c not in df.columns for c in feature_cols + ["Label"]):
            print(f"[!] columns missing in {fname}, skipping")
            continue
        is_b = df["Label"].astype(str).str.upper().str.strip() == "BENIGN"
        used = []
        for sub in (df[is_b], df[~is_b]):
            if len(sub):
                used.append(sub.sample(n=min(150, len(sub)), random_state=42).index)  # profile rows
        used = used[0].append(used[1:]) if len(used) > 1 else used[0]
        rest = df.drop(index=used)
        rb, ra = rest[is_b.drop(index=used)], rest[~is_b.drop(index=used)]
        for sub in (rb, ra):
            if len(sub):
                frames.append(sub.sample(n=min(per_bucket, len(sub)), random_state=seed))
        print(f"[*] {fname}: {min(per_bucket,len(rb))} benign + {min(per_bucket,len(ra))} attack")
    out = pd.concat(frames, ignore_index=True)
    out["Label"] = out["Label"].astype(str).str.strip()
    sub = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(sub.to_numpy()).all(axis=1)
    print(f"[*] Dropping {(~ok).sum()} non-finite rows")
    out = out[ok].reset_index(drop=True)
    out[feature_cols] = sub[ok].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=400, help="rows per file per benign/attack bucket")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    cfg0 = load_default_config()
    config = load_config(cfg0.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(cfg0.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, len(pre.feature_columns), store_attention=True).to(device)
    model.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model.eval()

    prof = np.load("artifacts/reference_profile.npy", allow_pickle=True).item()
    prof = {int(k): np.asarray(v, dtype=np.float32) for k, v in prof.items()}

    df = load_heldout(list(pre.feature_columns), args.per_bucket, args.seed)
    y = (df["Label"].str.upper() != "BENIGN").astype(int).to_numpy()
    print(f"[*] Held-out set: {len(df)} flows ({(y==0).sum()} benign, {(y==1).sum()} attack)")

    X = torch.tensor(np.asarray(pre.transform(df[pre.feature_columns])), dtype=torch.float32)
    probs, jsds = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = X[i:i + args.batch].to(device)
            p = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            attr = get_feature_attribution(compute_attention_rollout(model.get_attention_maps()))
            attr = attr.cpu().numpy().reshape(len(xb), -1)
            for pi, a in zip(p, attr):
                jsds.append(float(compute_divergence(prof.get(1 if pi > 0.5 else 0, prof[0]), a)))
            probs.extend(p.tolist())
            if (i // args.batch) % 10 == 0:
                print(f"    ...{min(i+args.batch, len(X))}/{len(X)}")

    df["p_attack"], df["jsd"], df["y"] = probs, jsds, y
    j = df["jsd"].to_numpy()
    ben, att = j[y == 0], j[y == 1]

    print("\n=== JSD by TRUE label ===")
    for n, a in (("benign", ben), ("attack", att)):
        q = np.percentile(a, [5, 25, 50, 75, 95, 99])
        print(f"{n:7s} n={len(a):5d} mean={a.mean():.4f}  p5/25/50/75/95/99 = " + " / ".join(f"{v:.4f}" for v in q))

    a_j = auc(att, ben)
    a_p = auc(df.p_attack[y == 1].to_numpy(), df.p_attack[y == 0].to_numpy())
    print(f"\nAUC of JSD as attack detector:        {a_j:.3f}  (0.5 = useless, <0.5 = inverted)")
    print(f"AUC of model P(attack), for reference: {a_p:.3f}")

    # Does JSD help where the model is unsure or wrong?
    miss = (df.y == 1) & (df.p_attack <= 0.5)
    print(f"Attacks the model misses (P<=0.5): {int(miss.sum())}; mean JSD missed={df.jsd[miss].mean():.4f} "
          f"vs caught={df.jsd[(df.y==1)&~miss].mean():.4f}") if miss.any() else None

    out = {"auc_jsd": a_j, "auc_prob": a_p, "thresholds": {}}
    print("\n=== Candidate thresholds from benign JSD percentiles ===")
    for pct in (90, 95, 99):
        t = float(np.percentile(ben, pct))
        print(f"p{pct}: tau={t:.4f}  benign flagged={(ben>t).mean():.1%}  attack flagged={(att>t).mean():.1%}")
        out["thresholds"][f"p{pct}"] = {"tau": t, "fpr": float((ben > t).mean()), "tpr": float((att > t).mean())}
    print(f"current 0.050: benign flagged={(ben>0.05).mean():.1%}  attack flagged={(att>0.05).mean():.1%}")

    print("\n=== Mean JSD per Label ===")
    g = df.groupby("Label")["jsd"].agg(["count", "mean", "std"]).sort_values("mean")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    out["per_label"] = g.reset_index().to_dict(orient="records")

    print("\n=== Live-run values vs benign distribution ===")
    for k, v in LIVE.items():
        print(f"{k:18s} {v:.3f} -> benign percentile {(ben < v).mean():.0%}")

    res = Path("artifacts/results"); res.mkdir(parents=True, exist_ok=True)
    (res / "gate_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = np.linspace(0, max(j.max(), 0.2), 60)
        plt.figure(figsize=(7, 4))
        plt.hist(ben, bins, alpha=.6, label="benign", density=True)
        plt.hist(att, bins, alpha=.6, label="attack", density=True)
        plt.axvline(0.05, color="k", ls="--", label="gate 0.05")
        plt.xlabel("JSD (base 2)"); plt.ylabel("density"); plt.legend()
        plt.title(f"JSD by true label (AUC={a_j:.3f})"); plt.tight_layout()
        plt.savefig(res / "gate_jsd_hist.png", dpi=150)
        print(f"[+] Saved {res/'gate_jsd_hist.png'}")
    except Exception as e:
        print(f"[!] Histogram skipped: {e}")
    print(f"[+] Saved {res/'gate_calibration.json'}")


if __name__ == "__main__":
    main()