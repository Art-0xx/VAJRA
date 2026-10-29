# PyTorch Dataset wrapper for VAJRA flow data.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from vajra.data.preprocess import Preprocessor, load_preprocessor


class NIDSDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        preprocessor_path: str | Path,
        label_column: str = "binary_label",
    ) -> None:
        self.df = pd.read_parquet(parquet_path)
        self.pre: Preprocessor = load_preprocessor(preprocessor_path)
        self.X: np.ndarray = self.pre.transform(self.df)
        self.y: np.ndarray = self.df[label_column].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def build_loader(
    parquet_path: str | Path,
    preprocessor_path: str | Path,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    ds = NIDSDataset(parquet_path, preprocessor_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)