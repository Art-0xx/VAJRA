# Tests for the preprocessing pipeline.
import numpy as np
import pandas as pd

from vajra.data.preprocess import (
    BINARY_LABEL,
    LABEL_COLUMN,
    add_binary_label,
    clean_dataframe,
    fit_preprocessor,
)


def make_fake_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Feature A": rng.normal(size=n),
        "Feature B": rng.uniform(size=n),
        LABEL_COLUMN: ["BENIGN"] * (n // 2) + ["DoS Hulk"] * (n - n // 2),
    })


def test_cleaning_drops_inf_and_duplicates():
    df = make_fake_df(50)
    df.loc[0, "Feature A"] = np.inf
    df.loc[1, "Feature B"] = -np.inf
    df = pd.concat([df, df.iloc[[2, 3]]], ignore_index=True)

    cleaned, report = clean_dataframe(df)

    assert not cleaned.isna().any().any()
    assert np.isfinite(cleaned["Feature A"]).all()
    assert np.isfinite(cleaned["Feature B"]).all()
    assert cleaned.duplicated().sum() == 0
    assert report["n_nan_rows"] == 2
    assert report["n_duplicate_rows"] >= 2


def test_binary_label_mapping():
    df = make_fake_df(10)
    df = add_binary_label(df)
    benign = df[LABEL_COLUMN].str.upper() == "BENIGN"
    assert (df.loc[benign, BINARY_LABEL] == 0).all()
    assert (df.loc[~benign, BINARY_LABEL] == 1).all()


def test_scaler_fit_on_train_only():
    train_df = make_fake_df(200)
    train_df["Feature A"] = train_df["Feature A"].abs() + 1.0

    pre = fit_preprocessor(train_df)
    X_train = pre.transform(train_df)

    outlier = train_df.iloc[[0]].copy()
    outlier["Feature A"] = train_df["Feature A"].max() * 10
    X_out = pre.transform(outlier)

    a_idx = pre.feature_columns.index("Feature A")
    assert X_train[:, a_idx].max() <= 1.0 + 1e-6
    assert X_out[0, a_idx] > 1.0


def test_transform_is_deterministic():
    df = make_fake_df(50)
    pre = fit_preprocessor(df)
    X1 = pre.transform(df)
    X2 = pre.transform(df)
    assert np.allclose(X1, X2)
    assert X1.dtype == np.float32
    assert X1.shape == (50, 2)