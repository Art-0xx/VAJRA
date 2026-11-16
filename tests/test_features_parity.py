# Feature Parity Tests (Blueprint Section 5.2)
import pytest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from vajra.flows.extractor import PcapFlowExtractor

PCAP_PATH = "live_sample.pcap"


def test_flow_extractor_produces_77_cic_columns():
    pre = joblib.load("artifacts/preprocessor.joblib")
    extractor = PcapFlowExtractor(feature_columns=pre.feature_columns, scaler=pre.scaler)

    if not Path(PCAP_PATH).exists():
        pytest.skip(f"PCAP {PCAP_PATH} not found.")

    df = extractor.extract_from_pcap(PCAP_PATH)
    assert len(df) > 0
    assert list(df.columns) == pre.feature_columns
    assert df.shape[1] == 77

    # Check key columns have non-negative values
    assert (df["Flow Duration"] >= 0).all()
    assert (df["Total Fwd Packets"] >= 1).all()

    # Verify scaling produces 77-dimensional numerical array
    scaled = extractor.transform_to_model_input(df)
    assert scaled.shape == (len(df), 77)
    assert not np.isnan(scaled).any()

