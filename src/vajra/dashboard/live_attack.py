# VAJRA Live Attack Console — Level 1 demo.
import os
import random

import joblib
import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="VAJRA Live Attack", layout="wide")
st.title("VAJRA — Live Attack Console")
st.caption("Real CIC-IDS2017 flows. Real Transformer. Real FGSM attack.")

@st.cache_data
def load_test():
    df = pd.read_parquet("data/processed/test.parquet")
    return df[df["binary_label"] == 1].reset_index(drop=True)

try:
    attacks = load_test()
    pre = joblib.load("artifacts/preprocessor.joblib")
except Exception as e:
    st.error(f"Missing artifacts: {e}")
    st.stop()

with st.sidebar:
    st.header("Controls")
    idx = st.number_input(
        "Attack flow index",
        min_value=0, max_value=len(attacks) - 1,
        value=random.randint(0, len(attacks) - 1),
        step=1,
    )
    if st.button("Random attack flow"):
        idx = random.randint(0, len(attacks) - 1)
    eps = st.slider("FGSM ε", 0.001, 0.10, 0.05, 0.001)
    run = st.button("Run Attack", type="primary")

row = attacks.iloc[int(idx)]
features = pre.transform(attacks.iloc[[int(idx)]]).flatten().tolist()

st.subheader(f"Flow #{int(idx)} (true label: ATTACK)")

try:
    h = requests.get(f"{API_BASE}/health", timeout=3).json()
    st.caption(f"API: {h['status']} | model: {h['is_model_loaded']}")
except Exception as e:
    st.error(f"API unreachable: {e}")
    st.stop()

if run:
    clean = requests.post(
        f"{API_BASE}/predict",
        json={"features": features},
        timeout=10,
    ).json()

    adv = requests.post(
        f"{API_BASE}/predict/adversarial",
        json={"features": features, "attack": "fgsm", "epsilon": eps},
        timeout=30,
    ).json()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Clean")
        st.metric("Label", clean["label"].upper())
        st.metric("Confidence", f"{clean['confidence']:.3f}")
    with col2:
        st.markdown("### After FGSM")
        st.metric("Label", adv["adversarial"]["label"].upper())
        st.metric("Confidence", f"{adv['adversarial']['confidence']:.3f}")

    st.divider()
    if adv["label_flipped"]:
        st.error(f"⚠ ATTACK EVADED — label flipped from {clean['label']} to {adv['adversarial']['label']}")
    else:
        st.success(f"✓ Attack failed — model still predicts {adv['adversarial']['label']}")

    st.caption(
        f"Perturbation budget: ε = {eps}. "
        f"Each feature changed by at most ±{eps} in [0,1] feature space. "
        f"Original prediction confidence {clean['confidence']:.3f} → adversarial {adv['adversarial']['confidence']:.3f}."
    )

st.divider()
st.caption("Dashboard calls /predict and /predict/adversarial on the real trained model.")