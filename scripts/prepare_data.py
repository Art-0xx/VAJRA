# Prepare CIC-IDS2017 for VAJRA: clean, split, fit preprocessor, save.
import argparse
from pathlib import Path

import pandas as pd

from vajra.data.preprocess import (
    add_binary_label,
    clean_dataframe,
    fit_preprocessor,
    load_cicids_csvs,
    save_preprocessor,
    split_and_save,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017", choices=["cicids2017"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--preprocessor-out", default="artifacts/preprocessor.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.dataset} from {args.raw_dir}")
    df = load_cicids_csvs(args.raw_dir)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("Cleaning...")
    df, report = clean_dataframe(df)
    print(f"  Dropped NaN rows      : {report['n_nan_rows']:,}")
    print(f"  Dropped duplicate rows: {report['n_duplicate_rows']:,}")
    print(f"  Remaining rows        : {report['n_remaining']:,}")

    df = add_binary_label(df)
    counts = df["binary_label"].value_counts().to_dict()
    print(f"  Binary label counts   : {counts}")

    print("Splitting 70/15/15 (stratified)...")
    paths = split_and_save(df, Path(args.out_dir), seed=args.seed)
    for name, p in paths.items():
        size_mb = p.stat().st_size / 1e6
        print(f"  {name:<5}: {p}  ({size_mb:.1f} MB)")

    print("Fitting preprocessor on train split only...")
    train_df = pd.read_parquet(paths["train"])
    pre = fit_preprocessor(train_df)
    save_preprocessor(pre, args.preprocessor_out)
    print(f"  Saved: {args.preprocessor_out}")
    print(f"  Features: {len(pre.feature_columns)}")

    print("\nDone. Train/val/test parquet files and preprocessor are ready.")


if __name__ == "__main__":
    main()