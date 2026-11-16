#!/usr/bin/env python
"""
VAJRA v2.0 - Headless Tactical CLI Evaluation Harness
Run attacks, defense validations, and attention divergence directly in the terminal without UI.

Usage:
    python scripts/test_cli.py --attack constrained --eps 0.05
    python scripts/test_cli.py --attack pgd --eps 0.05 --defend
    python scripts/test_cli.py --attack fgsm --eps 0.03 --flow-idx 12
    python scripts/test_cli.py --pcap live_sample.pcap
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.attacks.fgsm import fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.attacks.autoattack import apgd_ce_attack
from vajra.attacks.constrained_pgd import constrained_pgd_attack
from vajra.defenses.input_validator import InputValidator
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution
from vajra.justification.divergence import compute_divergence
from vajra.response.engine import ResponseEngine

# ANSI Palette
R = "\033[91m"
G = "\033[92m"
C = "\033[96m"
Y = "\033[93m"
B = "\033[1m"
X = "\033[0m"

TAU_BASELINE = 0.880
TAU_DEFENDED = 0.500
DIVERGENCE_TRIGGER = 0.150

def parse_args():
    parser = argparse.ArgumentParser(description="VAJRA Headless Threat CLI")
    parser.add_argument("--attack", choices=["constrained", "pgd", "fgsm", "apgd", "none"], default="constrained",
                        help="Adversarial attack algorithm (default: constrained)")
    parser.add_argument("--eps", type=float, default=0.050, help="Perturbation budget epsilon (default: 0.050)")
    parser.add_argument("--defend", action="store_true", help="Load defended_model.pt (PGD-AT) instead of baseline")
    parser.add_argument("--flow-idx", type=int, default=0, help="Row index from test set to attack (default: 0)")
    parser.add_argument("--pcap", type=str, default=None, help="Optional PCAP file to extract flows and classify")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n{C}{B}======================================================================{X}")
    print(f"{C}{B} VAJRA // HEADLESS CLI THREAT EVALUATION HARNESS{X}")
    print(f"{C}{B}======================================================================{X}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {B}{str(device).upper()}{X}")

    # 1. Load Preprocessor & Models
    pre = joblib.load("artifacts/preprocessor.joblib")
    config = load_config("configs/model_transformer.yaml")
    num_features = len(pre.feature_columns)

    model = get_model(config, num_features, store_attention=True).to(device)
    if args.defend:
        ckpt_path = "artifacts/checkpoints/defended_model.pt"
        active_tau = TAU_DEFENDED
        model_name = "defended_model.pt (PGD-AT Hardened)"
    else:
        ckpt_path = "artifacts/checkpoints/baseline_model.pt"
        active_tau = TAU_BASELINE
        model_name = "baseline_model.pt (Unhardened Baseline)"

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[*] Mounted Backbone: {G if args.defend else Y}{model_name}{X}")
    print(f"[*] Calibrated Decision Threshold (Tau): {B}{active_tau:.3f}{X}")

    # 2. Input Validation Layer
    validator = InputValidator(clip=True, reject_nan=True, clip_min=0.0, clip_max=1.0, feature_names=pre.feature_columns)

    # 3. Handle PCAP or Test Parquet
    if args.pcap:
        from vajra.flows.extractor import PcapFlowExtractor
        print(f"\n{Y}[*] Ingesting and assembling packets from {args.pcap}...{X}")
        extractor = PcapFlowExtractor(feature_columns=pre.feature_columns, scaler=pre.scaler)
        pcap_df = extractor.extract_from_pcap(args.pcap)
        print(f"[+] Reconstructed {len(pcap_df)} bidirectional flows from PCAP.")
        raw_df = pcap_df.iloc[[0]]
    else:
        print(f"\n[*] Slicing target flow row from test set...")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile("data/processed/test.parquet")
        # Read small batch
        batch = pf.read_row_group(0)
        df_slice = batch.to_pandas()
        if "Label" in df_slice.columns:
            attacks_df = df_slice[~df_slice["Label"].isin([0, "BENIGN", "benign"])]
            if len(attacks_df) == 0:
                attacks_df = df_slice
        else:
            attacks_df = df_slice
        sample_row = attacks_df.iloc[[args.flow_idx % len(attacks_df)]]
        raw_df = sample_row

    # Transform
    x_raw = pre.transform(raw_df)
    v_report = validator.validate(x_raw)
    print(f"[*] Input Validator: Valid={v_report.is_valid} | Clipped={v_report.clipped} | Out-of-bounds={v_report.out_of_bounds_count}")

    x_clean = v_report.sanitized_tensor.to(device)
    y_target = torch.tensor([1], dtype=torch.long).to(device)

    # 4. Clean Inference
    with torch.no_grad():
        clean_logits = model(x_clean)
        clean_prob = torch.softmax(clean_logits, dim=1)[0, 1].item()
    clean_verdict = "ATTACK" if clean_prob > active_tau else "BENIGN"

    print(f"\n{B}----------------------------------------------------------------------{X}")
    print(f"{B} 1. CLEAN FLOW INFERENCE TELEMETRY{X}")
    print(f"{B}----------------------------------------------------------------------{X}")
    print(f"  P(ATTACK): {clean_prob*100:5.2f}% | Threshold: {active_tau*100:5.2f}%")
    print(f"  Verdict:   {R if clean_verdict == 'ATTACK' else G}{B}{clean_verdict}{X}")

    if args.attack == "none":
        print(f"\n[*] Attack mode is NONE. Terminating.")
        return

    # 5. Adversarial Perturbation
    print(f"\n{B}----------------------------------------------------------------------{X}")
    print(f"{B} 2. ADVERSARIAL ATTACK PIPELINE (Vector: {args.attack.upper()}, eps={args.eps:.3f}){X}")
    print(f"{B}----------------------------------------------------------------------{X}")
    
    t0 = time.perf_counter()
    if args.attack == "constrained":
        res = constrained_pgd_attack(model, x_clean, y_target, epsilon=args.eps,
                                     feature_names=pre.feature_columns, num_iter=10)
    elif args.attack == "pgd":
        res = pgd_attack(model, x_clean, y_target, epsilon=args.eps, num_iter=10)
    elif args.attack == "fgsm":
        res = fgsm_attack(model, x_clean, y_target, epsilon=args.eps)
    elif args.attack == "apgd":
        res = apgd_ce_attack(model, x_clean, y_target, epsilon=args.eps, num_iter=15)
    t_delta = (time.perf_counter() - t0) * 1000

    x_adv = res.adversarial
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_prob = torch.softmax(adv_logits, dim=1)[0, 1].item()
    adv_verdict = "ATTACK" if adv_prob > active_tau else "BENIGN"

    l_inf = torch.max(torch.abs(x_adv - x_clean)).item()
    l_2 = torch.norm(x_adv - x_clean).item()
    is_evaded = (clean_verdict == "ATTACK" and adv_verdict == "BENIGN")

    print(f"  Execution Time: {t_delta:.2f} ms")
  
    print(f"  Perturbation:   L_inf={l_inf:.4f} | L_2={l_2:.4f}")
    print(f"  Adversarial P(ATTACK): {adv_prob*100:5.2f}% (was {clean_prob*100:5.2f}%)")
    print(f"  Adversarial Verdict:   {R if adv_verdict == 'ATTACK' else G}{B}{adv_verdict}{X}")

    if is_evaded:
        print(f"\n  {R}{B}[!] STATUS: EXPLOITED // NIDS EVADED (RED TEAM VICTORY){X}")
        print(f"      Model attention mechanism blinded; malicious flow classified as BENIGN.")
    else:
        print(f"\n  {G}{B}[+] STATUS: DEFENDED // THREAT HELD (BLUE TEAM RESISTED){X}")
        print(f"      Perturbation resisted; attack remained detected as MALICIOUS.")

    # 6. Attention Rollout & Divergence Analysis
    print(f"\n{B}----------------------------------------------------------------------{X}")
    print(f"{B} 3. ATTENTION ROLLOUT & JENSEN-SHANNON DIVERGENCE (JSD){X}")
    print(f"{B}----------------------------------------------------------------------{X}")

    # Clean rollout
    with torch.no_grad():
        model(x_clean)
        rollout_clean = compute_attention_rollout(model.get_attention_maps())
        attr_clean = get_feature_attribution(rollout_clean).cpu().numpy().flatten()

        model(x_adv)
        rollout_adv = compute_attention_rollout(model.get_attention_maps())
        attr_adv = get_feature_attribution(rollout_adv).cpu().numpy().flatten()

    jsd = compute_divergence(torch.tensor(attr_clean), torch.tensor(attr_adv))
    if hasattr(jsd, "item"):
        jsd = jsd.item()
    print(f"  Jensen-Shannon Divergence (JSD): {B}{jsd:.4f}{X} (Trigger Threshold: {DIVERGENCE_TRIGGER:.3f})")
    
    if jsd > DIVERGENCE_TRIGGER:
        print(f"  {R}[!] DIVERGENCE ALERT: Attention weights hijacked by perturbation!{X}")
    else:
        print(f"  {G}[*] ATTENTION STABLE: Attention manifold preserved under perturbation.{X}")

    # Top perturbed features
    delta_vec = torch.abs(x_adv - x_clean).cpu().numpy().flatten()
    top_indices = np.argsort(delta_vec)[-5:][::-1]
    print(f"\n  Top Attacked Feature Dimensions:")
    for rank, idx in enumerate(top_indices, 1):
        feat_name = pre.feature_columns[idx]
        print(f"    [{rank}] {feat_name:<32} Delta: {delta_vec[idx]:.4f} (Attn: {attr_adv[idx]:.4f})")

    # 7. Automated Response Engine Emulation
    print(f"\n{B}----------------------------------------------------------------------{X}")
    print(f"{B} 4. AUTOMATED INCIDENT RESPONSE ENGINE (IPTABLES){X}")
    print(f"{B}----------------------------------------------------------------------{X}")
    engine = ResponseEngine(mode="simulate", block_duration_sec=300)
    action = engine.handle_verdict(
        flow_id="0x7F4A",
        verdict=adv_verdict,
        confidence=adv_prob if not is_evaded else clean_prob,
        src_ip="192.168.10.50",
        dst_ip="172.16.0.1",
        dst_port=443
    )
    if action:
        print(f"  Decision: {B}{action.action}{X} | Rule: {action.rule}")
    else:
        print(f"  Decision: {B}ALLOW{X} | No firewall action triggered (Flow evaluated as {adv_verdict}).")
    print(f"{C}{B}======================================================================\n{X}")

if __name__ == "__main__":
    main()
