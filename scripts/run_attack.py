# Run an attack evaluation and append to Table B.
#
# Examples:
#   python scripts/run_attack.py --attack fgsm --eps 0.01 0.03 0.05 0.10
#   python scripts/run_attack.py --attack pgd --eps 0.01 0.03 0.05 0.10
#   python scripts/run_attack.py --attack apgd --eps 0.05 0.10 --apgd-iters 100
#   python scripts/run_attack.py --attack transfer --eps 0.05 0.10
#
# Each invocation appends rows to artifacts/metrics/table_B.csv. Rerun the
# same command to update rows with the same (attack, epsilon) key.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from vajra.attacks.autoattack import apgd_ce_attack
from vajra.attacks.evaluate_attack import AttackMetrics, evaluate_attack, metrics_to_dict
from vajra.attacks.fgsm import fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.attacks.transfer import load_or_train_surrogate, transfer_attack
from vajra.config import load_config
from vajra.data.dataset import NIDSDataset
from vajra.models.registry import get_model

TABLE_B_PATH = Path("artifacts/metrics/table_B.csv")
TABLE_B_FIELDS = [
    "model", "attack", "epsilon", "n_total", "n_attacked", "n_successful",
    "asr", "clean_f1", "robust_f1", "delta_f1", "clean_accuracy", "robust_accuracy",
]


def _load_model(config, preprocessor, checkpoint_path, device, store_attention=False):
    model = get_model(config, len(preprocessor.feature_columns),
                      store_attention=store_attention).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _make_attack_fn(attack_name, epsilon, args, surrogate_model, device):
    """Return a closure (model, x, y) -> adversarial_x."""
    if attack_name == "fgsm":
        def _fgsm_fn(m, x, y):
            return fgsm_attack(m, x, y, epsilon=epsilon).adversarial
        return _fgsm_fn

    if attack_name == "pgd":
        def _pgd_fn(m, x, y):
            return pgd_attack(m, x, y, epsilon=epsilon,
                              num_iter=args.pgd_iters).adversarial
        return _pgd_fn

    if attack_name == "apgd":
        def _apgd_fn(m, x, y):
            return apgd_ce_attack(m, x, y, epsilon=epsilon,
                                  num_iter=args.apgd_iters).adversarial
        return _apgd_fn

    if attack_name == "transfer":
        assert surrogate_model is not None
        def _transfer_fn(m, x, y):
            return transfer_attack(surrogate_model, x, y,
                                   epsilon=epsilon, num_iter=args.pgd_iters)
        return _transfer_fn

    if attack_name == "noise":
        def _noise_fn(m, x, y):
            noise = (torch.rand_like(x) * 2 - 1) * epsilon
            return (x + noise).clamp(0.0, 1.0)
        return _noise_fn

    raise ValueError(f"Unknown attack: {attack_name}")

def _append_table_b(metrics: AttackMetrics, model_name: str):
    TABLE_B_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name, **metrics_to_dict(metrics)}

    # Load existing rows, replace matching (model, attack, epsilon)
    rows = []
    if TABLE_B_PATH.exists():
        with TABLE_B_PATH.open() as f:
            rows = [r for r in csv.DictReader(f)]
        rows = [
            r for r in rows
            if not (
                r["model"] == model_name
                and r["attack"] == metrics.attack
                and abs(float(r["epsilon"]) - metrics.epsilon) < 1e-9
            )
        ]
    rows.append(row)

    with TABLE_B_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_B_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TABLE_B_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True,
                    choices=["fgsm", "pgd", "apgd", "transfer", "noise"])
    ap.add_argument("--eps", nargs="+", type=float, required=True)
    ap.add_argument("--config", default="configs/model_transformer.yaml")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline_model.pt")
    ap.add_argument("--model-name", default="baseline")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pgd-iters", type=int, default=10)
    ap.add_argument("--apgd-iters", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit evaluation to first N test samples (default: full)")
    ap.add_argument("--surrogate-checkpoint",
                    default="artifacts/checkpoints/surrogate_cnn.pt")
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    threshold = 0.880
    metrics_path = Path("artifacts/metrics/table_A_clean.json")
    if metrics_path.exists():
        threshold = float(json.loads(metrics_path.read_text())["optimal_threshold"])
    print(f"Threshold: {threshold:.3f}")

    # Target model
    model = _load_model(config, preprocessor, args.checkpoint, device)

    # Surrogate (only for transfer attack)
    surrogate_model = None
    if args.attack == "transfer":
        train_ds = NIDSDataset("data/processed/train.parquet", "artifacts/preprocessor.joblib")
        val_ds = NIDSDataset("data/processed/val.parquet", "artifacts/preprocessor.joblib")
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        label_arr = train_ds.df["binary_label"].to_numpy()
        counts = torch.bincount(torch.from_numpy(label_arr), minlength=2).float()
        class_weights = (len(label_arr) / (2 * counts)).to(device)
        surrogate_model = load_or_train_surrogate(
            train_loader, val_loader, train_ds.n_features, device,
            Path(args.surrogate_checkpoint), class_weights=class_weights,
        )

    # Test set
    test_ds = NIDSDataset("data/processed/test.parquet", "artifacts/preprocessor.joblib")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Run each epsilon
    for eps in args.eps:
        print(f"\n=== {args.attack.upper()} eps={eps} ===")
        attack_fn = _make_attack_fn(args.attack, eps, args, surrogate_model, device)
        metrics = evaluate_attack(
            model, test_loader, attack_fn,
            attack_name=args.attack, epsilon=eps,
            threshold=threshold, device=device,
            max_samples=args.max_samples,
        )
        print(f"  ASR             : {metrics.asr*100:.2f}%  "
              f"({metrics.n_successful}/{metrics.n_attacked})")
        print(f"  Robust accuracy : {metrics.robust_accuracy:.4f}")
        print(f"  Robust F1       : {metrics.robust_f1:.4f}")
        print(f"  Delta F1        : {metrics.delta_f1:.4f}")
        _append_table_b(metrics, args.model_name)

    print(f"\nTable B updated: {TABLE_B_PATH}")


if __name__ == "__main__":
    main()