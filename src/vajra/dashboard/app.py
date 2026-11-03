# VAJRA - Unified SOC Dashboard
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch

from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.attacks.fgsm import fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

# Page Config
st.set_page_config(page_title="VAJRA SOC", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Mode / Hacker Look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #00ffcc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-size: 20px; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom-color: #00ffcc !important; }
    h1, h2, h3 { color: #00ffcc !important; }
    .metric-box { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc; text-align: center; }
    .metric-value { font-size: 36px; font-weight: bold; margin: 0; }
    .metric-label { font-size: 16px; color: #888; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VAJRA: Robust NIDS Console")

# Load Models & Data
@st.cache_resource
def load_resources():
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre = joblib.load("artifacts/preprocessor.joblib")
    
    # Load Baseline
    model_base = get_model(config, len(pre.feature_columns), store_attention=True).to(device)
    model_base.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model_base.eval()
    
    # Load Defended
    model_def = get_model(config, len(pre.feature_columns), store_attention=True).to(device)
    model_def.load_state_dict(torch.load("artifacts/checkpoints/defended_model.pt", map_location=device))
    model_def.eval()
    
    # Load Data
    df = pd.read_parquet("data/processed/test.parquet")
    attacks_df = df[df["binary_label"] == 1].reset_index(drop=True)
    benign_df = df[df["binary_label"] == 0].reset_index(drop=True)
    
    return pre, device, model_base, model_def, attacks_df, benign_df

pre, device, model_base, model_def, attacks_df, benign_df = load_resources()
threshold = 0.880

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📡 Live Traffic Monitor", "⚔️ Attack Simulator", "🧠 Explainability & Defense"])

# --- TAB 1: LIVE TRAFFIC ---
with tab1:
    st.header("Live Network Traffic")
    st.write("Streaming simulated CIC-IDS2017 flows into the NIDS.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("▶️ Start Stream", type="primary")
        stop_btn = st.button("⏹️ Stop Stream")
    
    log_placeholder = st.empty()
    
    if start_btn:
        logs = []
        # Generate a mixed stream of attacks and benign
        stream_indices = [random.choice(attacks_df.index) for _ in range(15)] + [random.choice(benign_df.index) for _ in range(35)]
        random.shuffle(stream_indices)
        
        for i in range(50):
            time.sleep(0.1)  # Simulate network delay
            is_attack = (i % 3 == 0)
            
            if is_attack:
                flow = attacks_df.iloc[[i % len(attacks_df)]]
                true_label = "ATTACK"
            else:
                flow = benign_df.iloc[[i % len(benign_df)]]
                true_label = "BENIGN"
                
            x = torch.tensor(pre.transform(flow), dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model_base(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                
            pred_label = "ATTACK" if prob > threshold else "BENIGN"
            conf = prob if pred_label == "ATTACK" else (1 - prob)
            
            status = "🚨 ALERT" if pred_label == "ATTACK" else "✅ OK"
            color = "red" if pred_label == "ATTACK" else "green"
            
            log_str = f"<span style='color:{color}'><b>[{status}]</b> Time: {time.strftime('%H:%M:%S')} | Pred: {pred_label} ({conf:.3f}) | True: {true_label}</span>"
            logs.insert(0, log_str)
            
            # Render as HTML
            html = "<div style='font-family: monospace; font-size: 14px; background: #1a1c24; padding: 10px; border-radius: 5px;'>"
            html += "<br>".join(logs[:15])
            html += "</div>"
            log_placeholder.markdown(html, unsafe_allow_html=True)

# --- TAB 2: ATTACK SIMULATOR ---
with tab2:
    st.header("Adversarial Attack Console")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Attack Configuration")
        attack_type = st.selectbox("Select Attack Algorithm", ["FGSM", "PGD"])
        eps = st.slider("Perturbation Budget (ε)", 0.01, 0.15, 0.05, 0.01)
        
        if st.button("🚀 Execute Attack", type="primary"):
            # Pick a random attack
            flow = attacks_df.iloc[[random.randint(0, len(attacks_df)-1)]]
            x_clean = torch.tensor(pre.transform(flow), dtype=torch.float32).to(device)
            y = torch.tensor([1], dtype=torch.long).to(device)
            
            # Clean pred
            with torch.no_grad():
                clean_prob = torch.softmax(model_base(x_clean), dim=1)[0, 1].item()
                
            # Attack
            if attack_type == "FGSM":
                result = fgsm_attack(model_base, x_clean, y, epsilon=eps)
            else:
                result = pgd_attack(model_base, x_clean, y, epsilon=eps, num_iter=10)
                
            x_adv = result.adversarial
            
            with torch.no_grad():
                adv_prob = torch.softmax(model_base(x_adv), dim=1)[0, 1].item()
                
            clean_label = "ATTACK" if clean_prob > threshold else "BENIGN"
            adv_label = "ATTACK" if adv_prob > threshold else "BENIGN"
            
            # Save to session state for Tab 3
            st.session_state["x_clean"] = x_clean
            st.session_state["x_adv"] = x_adv
            st.session_state["clean_label"] = clean_label
            st.session_state["adv_label"] = adv_label
            st.session_state["clean_prob"] = clean_prob
            st.session_state["adv_prob"] = adv_prob
            
    with col2:
        if "clean_label" in st.session_state:
            st.subheader("Attack Results")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f"<div class='metric-box'><p class='metric-label'>Clean Prediction</p><p class='metric-value' style='color: {'red' if st.session_state['clean_label'] == 'ATTACK' else 'green'};'>{st.session_state['clean_label']}</p></div>", unsafe_allow_html=True)
                st.caption(f"Confidence: {st.session_state['clean_prob']:.3f}")
            with c2:
                st.markdown("<div class='metric-box' style='border:none; padding-top: 30px;'><h2 style='font-size: 40px;'>&#10132;</h2></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-box'><p class='metric-label'>Adversarial Prediction</p><p class='metric-value' style='color: {'red' if st.session_state['adv_label'] == 'ATTACK' else 'green'};'>{st.session_state['adv_label']}</p></div>", unsafe_allow_html=True)
                st.caption(f"Confidence: {st.session_state['adv_prob']:.3f}")
                
            if st.session_state['clean_label'] != st.session_state['adv_label']:
                st.error("⚠️ ATTACK SUCCESSFUL — Label Flipped!")
            else:
                st.success("🛡️ ATTACK FAILED — Model robust.")
        else:
            st.info("Configure and execute an attack to see results.")

# --- TAB 3: EXPLAINABILITY & DEFENSE ---
with tab3:
    st.header("Attention Justification & Defense")
    
    defense_on = st.toggle("Activate PGD Adversarial Training (Defended Model)", value=False)
    active_model = model_def if defense_on else model_base
    
    if "x_clean" in st.session_state and "x_adv" in st.session_state:
        x_clean = st.session_state["x_clean"]
        x_adv = st.session_state["x_adv"]
        
        # Get attentions
        with torch.no_grad():
            active_model(x_clean)
        clean_rollout = compute_attention_rollout(active_model.get_attention_maps())
        clean_attn = get_feature_attribution(clean_rollout)[0].cpu().numpy()
        
        with torch.no_grad():
            active_model(x_adv)
        adv_rollout = compute_attention_rollout(active_model.get_attention_maps())
        adv_attn = get_feature_attribution(adv_rollout)[0].cpu().numpy()
        
        # Model predictions
        with torch.no_grad():
            clean_prob_new = torch.softmax(active_model(x_clean), dim=1)[0, 1].item()
            adv_prob_new = torch.softmax(active_model(x_adv), dim=1)[0, 1].item()
            
        # Show defense result
        st.subheader("Defense Evaluation")
        if defense_on:
            st.success("Defended Model Active. Trained with PGD-AT.")
            new_label = "ATTACK" if adv_prob_new > threshold else "BENIGN"
            if new_label == "ATTACK":
                st.success(f"🛡️ DEFENSE SUCCESSFUL: Model correctly identified the attack despite perturbation. (P={adv_prob_new:.3f})")
            else:
                st.warning(f"⚠️ Attack bypassed the defense. (P={adv_prob_new:.3f})")
        else:
            st.warning("Baseline Model Active. Vulnerable to perturbation.")
            
        # Plotting
        st.markdown("---")
        st.subheader("Attention Hijacking Analysis")
        st.write("These charts show which features the model is focusing on.")
        
        features = pre.feature_columns
        top_clean_idx = np.argsort(clean_attn)[-7:][::-1]
        top_adv_idx = np.argsort(adv_attn)[-7:][::-1]
        
        c1, c2 = st.columns(2)
        
        with c1:
            fig1 = px.bar(
                x=[clean_attn[i] for i in top_clean_idx],
                y=[features[i] for i in top_clean_idx],
                orientation='h',
                title="Clean Flow Attention",
                labels={'x': 'Attention Weight', 'y': 'Feature'},
                color_discrete_sequence=['#00ffcc']
            )
            fig1.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            fig2 = px.bar(
                x=[adv_attn[i] for i in top_adv_idx],
                y=[features[i] for i in top_adv_idx],
                orientation='h',
                title="Adversarial Flow Attention",
                labels={'x': 'Attention Weight', 'y': 'Feature'},
                color_discrete_sequence=['#ff4b4b']
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.info("Go to the 'Attack Simulator' tab and run an attack first to see the explainability charts.")