# Security curve: plot ASR / robust accuracy vs epsilon for each attack.
#
# The security curve identifies the operational robustness limit — the
# largest epsilon at which the model retains more than some target robust
# accuracy. This is the standard visualization in the adversarial NIDS
# literature (Kheddar, 2025; hybrid CNN-LSTM-Transformer, 2026).
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

TABLE_B = Path("artifacts/metrics/table_B.csv")
FIG_DIR = Path("artifacts/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_table_b():
    if not TABLE_B.exists():
        raise FileNotFoundError(f"{TABLE_B} not found — run scripts/run_attack.py first")
    with TABLE_B.open() as f:
        return list(csv.DictReader(f))


def plot_security_curve(out_path: Path | None = None):
    rows = _load_table_b()
    by_attack = defaultdict(list)
    for r in rows:
        by_attack[r["attack"]].append(
            (float(r["epsilon"]), float(r["robust_accuracy"]))
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"fgsm": "o", "pgd": "s", "apgd": "^", "transfer": "D"}
    colors = {"fgsm": "#1f77b4", "pgd": "#d62728",
              "apgd": "#9467bd", "transfer": "#2ca02c"}

    for attack in ["fgsm", "pgd", "apgd", "transfer"]:
        if attack not in by_attack:
            continue
        pts = sorted(by_attack[attack])
        eps, ra = zip(*pts)
        ax.plot(eps, ra, marker=markers.get(attack, "o"),
                color=colors.get(attack, "gray"),
                label=attack.upper(), lw=2, markersize=8)

    # Operational threshold line at 0.90
    ax.axhline(0.90, linestyle="--", color="gray", lw=1, label="90% target")

    ax.set_xlabel("Perturbation budget ε (L∞)")
    ax.set_ylabel("Robust accuracy")
    ax.set_title("Security Curve — Baseline Transformer NIDS")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    out = out_path or FIG_DIR / "security_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_asr_curve(out_path: Path | None = None):
    rows = _load_table_b()
    by_attack = defaultdict(list)
    for r in rows:
        by_attack[r["attack"]].append((float(r["epsilon"]), float(r["asr"])))

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"fgsm": "o", "pgd": "s", "apgd": "^", "transfer": "D"}
    colors = {"fgsm": "#1f77b4", "pgd": "#d62728",
              "apgd": "#9467bd", "transfer": "#2ca02c"}

    for attack in ["fgsm", "pgd", "apgd", "transfer"]:
        if attack not in by_attack:
            continue
        pts = sorted(by_attack[attack])
        eps, asr = zip(*pts)
        ax.plot(eps, [a * 100 for a in asr],
                marker=markers.get(attack, "o"),
                color=colors.get(attack, "gray"),
                label=attack.upper(), lw=2, markersize=8)

    ax.set_xlabel("Perturbation budget ε (L∞)")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title("Attack Success Rate vs ε")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    out = out_path or FIG_DIR / "asr_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_security_curve()
    plot_asr_curve()