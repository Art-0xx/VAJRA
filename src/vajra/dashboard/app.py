# VAJRA Streamlit dashboard (P0 placeholder implementation).
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEMO_FLOW = [0.03, 0.11, 0.42, 0.0, 0.88, 0.01, 0.07, 0.19] + [0.05] * 70

st.set_page_config(page_title="VAJRA NIDS Console", layout="wide")
st.title("VAJRA - Robust NIDS Console")

with st.sidebar:
    st.header("Controls")
    model_choice = st.selectbox("Model", ["Undefended", "PGD-AT", "TRADES"])
    attack_choice = st.selectbox("Attack", ["None", "FGSM", "PGD", "AutoAttack"])
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    run = st.button("Run Attack", type="primary")
if attack_choice is None:
    st.error("Attack selection unavailable - no option selected.")
    st.stop()
    
try:
    health = requests.get(API_BASE_URL + "/health", timeout=3).json()
    st.caption(
        "API status: **" + health["status"] + "** | "
        + "model_loaded: `" + str(health["is_model_loaded"]) + "` | "
        + "preprocessor_loaded: `" + str(health["is_preprocessor_loaded"]) + "`"
    )
except Exception as exc:
    st.error("API unreachable at " + API_BASE_URL + ": " + str(exc))
    st.stop()

if run:
    try:
        if attack_choice == "None":
            resp = requests.post(
                API_BASE_URL + "/predict",
                json={"features": DEMO_FLOW},
                timeout=10,
            ).json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", resp["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["confidence"]))
            with col2:
                st.info("No attack selected. Switch attack to FGSM/PGD to test.")
        else:
            resp = requests.post(
                API_BASE_URL + "/predict/adversarial",
                json={
                    "features": DEMO_FLOW,
                    "attack": attack_choice.lower(),
                    "epsilon": epsilon,
                },
                timeout=30,
            ).json()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Clean")
                st.metric("Label", resp["clean"]["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["clean"]["confidence"]))
            with col2:
                st.subheader("Adversarial")
                st.metric("Label", resp["adversarial"]["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["adversarial"]["confidence"]))
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Label flipped", str(resp["label_flipped"]))
            c2.metric("ACS", "{:.2f}".format(resp["acs"]))
            if resp["flagged"]:
                st.error("Adversarial input detected by attention divergence")
            else:
                st.success("Attention divergence within normal range")
    except Exception as exc:
        st.error("Request failed: " + str(exc))

st.divider()
st.caption("P0 skeleton - predictions are placeholders until Week 3.")
