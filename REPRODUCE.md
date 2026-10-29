# Reproducing VAJRA Results

## Data Preparation

    python scripts/prepare_data.py --dataset cicids2017

## Table A - Clean Performance

    python scripts/evaluate_clean.py --checkpoint artifacts/checkpoints/baseline_model.pt

## Table B - Robustness Profile

    python scripts/run_attack.py --attack fgsm --eps 0.01 0.03 0.05 0.1 --seeds 1 2 3
    python scripts/run_attack.py --attack pgd  --eps 0.01 0.03 0.05 0.1 --seeds 1 2 3
