#!/usr/bin/env python
"""VAJRA live-demo script: PGD attack on baseline, then defended model.

For real CIC-IDS2017 attack rows the baseline DETECTS, it runs white-box
constrained PGD (L-inf, eps in the preprocessor's scaled space, locked
ports/flags, clipped to the observed feature range) and reports:
  - baseline evasion rate under attack
  - defended-model detection of the same adversarial examples (transfer)
  - defended-model evasion rate under its OWN white-box PGD
and prints a 4-column demo table for a few example flows.

Run from /mnt/d/VAJRA (no sudo):  python scripts/demo_attack_defense.py
NOTE: this is an independent PGD implementation, so numbers may differ from
the slide numbers (different sample, steps, bounds). Use it for the demo, and
if they differ a lot, say the demo uses its own settings.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vajra.config import load_config
from vajra.console_helpers import load_default_config
from vajra.models.registry import get_model

DATA_DIR = Path("data/raw/MachineLearningCVE")
CSV_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
LOCK_KEYS = ("port", "flag", "protocol")  # CIC: Destination Port + 4 PSH/URG Flags + 8 *Flag Count = 13


def load_attack_rows(feature_cols, per_file, seed):
    frames = []
    for f in CSV_FILES:
        fp = DATA_DIR / f
        if not fp.exists():
            print(f"[!] missing {fp}")
            continue
        df = pd.read_csv(fp, low_memory=False)
        df.columns = df.columns.str.strip()
        if any(c not in df.columns for c in feature_cols + ["Label"]):
            print(f"[!] columns missing in {f}")
            continue
        df = df[df["Label"].astype(str).str.strip().str.upper() != "BENIGN"]
        if len(df):
            frames.append(df.sample(n=min(per_file, len(df)), random_state=seed))
    out = pd.concat(frames, ignore_index=True)
    out["Label"] = out["Label"].astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
    num = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(num.to_numpy()).all(axis=1)
    out = out[ok].reset_index(drop=True)
    out[feature_cols] = num[ok].reset_index(drop=True)
    return out


def pgd(model, x, eps, alpha, steps, mask, lo, hi):
    """White-box L-inf PGD pushing true-attack rows toward 'benign'."""
    x0 = x.detach()
    y = torch.ones(len(x0), dtype=torch.long, device=x0.device)
    delta = (torch.rand_like(x0) * 2 - 1) * eps * mask
    for _ in range(steps):
        d = delta.clone().requires_grad_(True)
        xa = torch.min(torch.max(x0 + d, lo), hi)
        loss = F.cross_entropy(model(xa), y)
        g = torch.autograd.grad(loss, d)[0]
        delta = (delta + alpha * g.sign() * mask).clamp(-eps, eps).detach()
    return torch.min(torch.max(x0 + delta, lo), hi).detach()


@torch.no_grad()
def p_attack(model, x, bs=512):
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.softmax(model(x[i:i + bs]), dim=1)[:, 1])
    return torch.cat(out)


def attack_batched(model, x, bs=256, **kw):
    return torch.cat([pgd(model, x[i:i + bs], **kw) for i in range(0, len(x), bs)])


def load_model(config, n, name, device):
    m = get_model(config, n).to(device)
    m.load_state_dict(torch.load(f"artifacts/checkpoints/{name}", map_location=device))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--per-file", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--unconstrained", action="store_true", help="do not lock ports/flags")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg0 = load_default_config()
    config = load_config(cfg0.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(cfg0.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    tau = cfg0.get("thresholds", {"baseline": 0.880, "defended": 0.500})
    tau_b, tau_d = tau.get("baseline", 0.88), tau.get("defended", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = list(pre.feature_columns)

    base = load_model(config, len(cols), "baseline_model.pt", device)
    defn = load_model(config, len(cols), "defended_model.pt", device)

    df = load_attack_rows(cols, args.per_file, args.seed)
    X = torch.tensor(np.asarray(pre.transform(df[cols])), dtype=torch.float32, device=device)
    lo, hi = X.min(0).values, X.max(0).values  # observed feature range = feasibility box

    locked = [i for i, c in enumerate(cols) if (not args.unconstrained) and any(k in c.lower() for k in LOCK_KEYS)]
    mask = torch.ones(len(cols), device=device)
    mask[locked] = 0
    print(f"[*] {len(df)} real attack rows | eps={args.eps} steps={args.steps} | locked features: {len(locked)}")
    if locked:
        print("    locked: " + ", ".join(cols[i] for i in locked))
    kw = dict(eps=args.eps, alpha=args.eps / 10, steps=args.steps, mask=mask, lo=lo, hi=hi)

    pb = p_attack(base, X)
    det_b = pb > tau_b
    Xd = X[det_b]
    lab = df.loc[det_b.cpu().numpy(), "Label"].reset_index(drop=True)
    print(f"[*] Baseline detects {int(det_b.sum())}/{len(X)} attack rows (tau={tau_b})")

    # 1) white-box PGD vs baseline
    Xa_b = attack_batched(base, Xd, **kw)
    pb_adv = p_attack(base, Xa_b)
    evaded_b = pb_adv <= tau_b
    # 2) same adversarial examples vs defended (transfer)
    pd_clean = p_attack(defn, Xd)
    pd_transfer = p_attack(defn, Xa_b)
    # 3) white-box PGD vs defended (only rows defended detects clean)
    det_d = pd_clean > tau_d
    Xa_d = attack_batched(defn, Xd, **kw)
    pd_own = p_attack(defn, Xa_d)
    evaded_d = (pd_own <= tau_d) & det_d

    n = len(Xd)
    res = {
        "n_attack_rows": len(X), "n_detected_baseline": n, "eps": args.eps, "steps": args.steps,
        "locked_features": [cols[i] for i in locked],
        "baseline_evasion_rate": float(evaded_b.float().mean()),
        "defended_detects_clean": float(det_d.float().mean()),
        "defended_holds_on_baseline_adv": float((pd_transfer > tau_d).float().mean()),
        "defended_evasion_rate_own_pgd": float(evaded_d.sum() / max(int(det_d.sum()), 1)),
    }
    print("\n================ RESULTS (real CIC-IDS2017 attack rows) ================")
    print(f"Baseline   | clean detected: {n}/{len(X)} | evasion under PGD: {res['baseline_evasion_rate']:.1%}")
    print(f"Defended   | detects {res['defended_detects_clean']:.1%} of those clean rows")
    print(f"           | still flags {res['defended_holds_on_baseline_adv']:.1%} of the baseline-attack examples (transfer)")
    print(f"           | evasion under its OWN white-box PGD: {res['defended_evasion_rate_own_pgd']:.1%}")

    # per-class
    pc = pd.DataFrame({"Label": lab, "base_evaded": evaded_b.cpu().numpy(),
                       "def_held_transfer": (pd_transfer > tau_d).cpu().numpy(),
                       "def_clean": det_d.cpu().numpy()})
    g = pc.groupby("Label").agg(n=("Label", "size"), baseline_evaded=("base_evaded", "mean"),
                                defended_clean=("def_clean", "mean"),
                                defended_holds=("def_held_transfer", "mean")).sort_values("n", ascending=False)
    print("\nPer attack class:")
    print(g.to_string(float_format=lambda v: f"{v:.2f}" if v <= 1 else f"{v:.0f}"))

    # demo table: first example per class, prefer ones that actually evaded the baseline
    print("\n=== DEMO TABLE (P(attack): >tau_b flagged, baseline tau=%.2f, defended tau=%.2f) ===" % (tau_b, tau_d))
    print(f"{'Flow':<22}{'Base clean':>11}{'Base+PGD':>10}{'Def clean':>11}{'Def (same adv)':>16}{'Def+own PGD':>13}  Base fooled / Def holds own")
    ev = evaded_b.cpu().numpy()
    rows = []
    for cls in list(g.index[:4]):
        idx = np.where((lab == cls).to_numpy() & ev)[0]
        if len(idx) == 0:
            idx = np.where((lab == cls).to_numpy())[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        r = dict(flow=cls[:21], base_clean=float(pb[det_b][i]), base_pgd=float(pb_adv[i]),
                 def_clean=float(pd_clean[i]), def_transfer=float(pd_transfer[i]), def_own=float(pd_own[i]),
                 evaded=bool(ev[i]), def_holds_own=bool(pd_own[i] > tau_d))
        rows.append(r)
        tag = f"  {'yes' if r['evaded'] else 'no'} / {'yes' if r['def_holds_own'] else 'NO'}"
        print(f"{r['flow']:<22}{r['base_clean']:>10.1%}{r['base_pgd']:>10.1%}{r['def_clean']:>11.1%}{r['def_transfer']:>16.1%}{r['def_own']:>13.1%}{tag}")
    res["examples"] = rows
    res["per_class"] = g.reset_index().to_dict(orient="records")

    out = Path("artifacts/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "pgd_demo.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[+] Saved {out/'pgd_demo.json'}")
    print("[*] Feature-space attack: perturbed vectors are not guaranteed to map to sendable packets.")


if __name__ == "__main__":
    main()