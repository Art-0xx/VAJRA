# Cleaning, scaling, and splitting for CIC-IDS2017.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LABEL_COLUMN = "Label"
BINARY_LABEL = "binary_label"


@dataclass
class Preprocessor:
    feature_columns: list[str]
    scaler: MinMaxScaler

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        return self.scaler.transform(X).astype(np.float32)


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names without tripping pandas stubs."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_cicids_csvs(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df = _strip_columns(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = _strip_columns(df)

    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])

    df = df.replace([np.inf, -np.inf], np.nan)

    report: dict = {}
    report["n_nan_rows"] = int(df.isna().any(axis=1).sum())
    df = df.dropna().reset_index(drop=True)

    report["n_duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    report["n_remaining"] = int(len(df))
    return df, report


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df[LABEL_COLUMN].to_numpy(dtype=str)
    upper = np.char.upper(raw)
    df[BINARY_LABEL] = (upper != "BENIGN").astype(np.int64)
    return df


def fit_preprocessor(train_df: pd.DataFrame) -> Preprocessor:
    feature_columns = [
        c for c in train_df.columns if c not in {LABEL_COLUMN, BINARY_LABEL}
    ]
    X = train_df[feature_columns].to_numpy(dtype=np.float32)
    scaler = MinMaxScaler()
    scaler.fit(X)
    return Preprocessor(feature_columns=feature_columns, scaler=scaler)


def split_and_save(
    df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df[BINARY_LABEL], random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size),
        stratify=temp_df[BINARY_LABEL], random_state=seed,
    )

    paths = {
        "train": out / "train.parquet",
        "val": out / "val.parquet",
        "test": out / "test.parquet",
    }
    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)
    return paths


def save_preprocessor(pre: Preprocessor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, path)


def load_preprocessor(path: str | Path) -> Preprocessor:
    return joblib.load(path)