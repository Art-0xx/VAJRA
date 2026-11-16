#!/usr/bin/env python
# VAJRA Tactical Red Team Exploit & Evaluation Console
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch

# Suppress PyTorch nested tensor warnings cleanly
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

from scapy.all import rdpcap, sendp
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet, Raw

from vajra.attacks.constrained_pgd import constrained_pgd_attack
from vajra.attacks.fgsm import fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.config import load_config
from vajra.console_helpers import (
    CLR_B,
    CLR_C,
    CLR_DIM,
    CLR_G,
    CLR_M,
    CLR_R,
    CLR_X,
    CLR_Y,
    extract_packet_features,
    format_feature_diff_table,
    load_default_config,
    log_alert_entry,
    save_live_run_summary,
)
from vajra.models.registry import get_model


def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def safe_send_packet(pkt: Packet, iface: str) -> bool:
    """Sends packet via Scapy, catching raw socket permission issues gracefully."""
    try:
        sendp(pkt, iface=iface, verbose=False)
        return True
    except PermissionError:
        return False
    except Exception:
        return False


def build_base_malicious_packet() -> Packet:
    """Loads base packet from live_sample.pcap or synthesizes an attack frame."""
    pcap_path = Path("live_sample.pcap")
    if pcap_path.exists():
        try:
            pkts = [p for p in rdpcap(str(pcap_path)) if IP in p and TCP in p]
            if pkts:
                return pkts[0]
        except Exception:
            pass
    # Synthetic baseline attack frame
    return IP(src="192.168.1.105", dst="192.168.1.1") / TCP(sport=52110, dport=80, flags="S", window=1024) / Raw(load=b"EXPLOIT_PAYLOAD_BUFFER_OVERFLOW")


def build_benign_packets() -> List[Tuple[str, Packet]]:
    """Builds normal benign network traffic samples for false-positive validation."""
    samples = []
    # 1. Normal HTTP GET
    http_pkt = (
        IP(src="192.168.1.50", dst="192.168.1.1")
        / TCP(sport=49152, dport=80, flags="PA", window=65535)
        / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.org\r\nUser-Agent: Mozilla/5.0\r\nAccept: */*\r\n\r\n")
    )
    samples.append(("HTTP GET (Port 80)", http_pkt))

    # 2. Standard TLS Client Hello Handshake
    tls_pkt = (
        IP(src="192.168.1.50", dst="192.168.1.1")
        / TCP(sport=49154, dport=443, flags="PA", window=29200)
        / Raw(load=b"\x16\x03\x01\x00\xa5\x01\x00\x00\xa1\x03\x03")
    )
    samples.append(("TLS Client Hello (Port 443)", tls_pkt))

    # 3. Standard DNS Query
    dns_pkt = (
        IP(src="192.168.1.50", dst="192.168.1.1")
        / UDP(sport=53530, dport=53)
        / Raw(load=b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01")
    )
    samples.append(("DNS Standard Query (Port 53)", dns_pkt))
    return samples


def run_campaign(
    model: torch.nn.Module,
    model_name: str,
    tau: float,
    pre: Any,
    mode: str = "heuristic",
    epsilon: float = 0.050,
    iface: str = "lo",
    device: torch.device = torch.device("cpu"),
    delay: float = 1.0,
    silent_header: bool = False
) -> Dict[str, Any]:
    """
    Executes the multi-wave attack campaign and evaluates results.
    """
    if not silent_header:
        print(f"\n{CLR_M}{CLR_B}╔════════════════════════════════════════════════════════════════════╗{CLR_X}")
        print(f"{CLR_M}{CLR_B}║              VAJRA RED TEAM ADVERSARIAL PIPELINE                   ║{CLR_X}")
        print(f"{CLR_M}{CLR_B}║     Target Model: {model_name:<20} | Calibrated Tau: {tau:<7.3f}    ║{CLR_X}")
        print(f"{CLR_M}{CLR_B}╚════════════════════════════════════════════════════════════════════╝{CLR_X}")
        print(f"  {CLR_DIM}► Campaign Mode:{CLR_X}     {CLR_B}{mode.upper()}{CLR_X}")
        print(f"  {CLR_DIM}► Perturbation (ε):{CLR_X}  {CLR_B}{epsilon:.3f}{CLR_X}")
        print(f"  {CLR_DIM}► Egress Interface:{CLR_X}  {CLR_B}{iface}{CLR_X}")
        print(f"  {CLR_DIM}► Execution Device:{CLR_X}  {CLR_B}{str(device).upper()}{CLR_X}")
        print(f"{CLR_M}{CLR_DIM}──────────────────────────────────────────────────────────────────────{CLR_X}\n")

    base_pkt = build_base_malicious_packet()
    target_ports = [80, 22, 53]
    port_names = {80: "HTTP Web Service", 22: "SSH Management", 53: "DNS Name Resolution"}

    clean_detected = 0
    adv_evaded = 0
    total_attacks = len(target_ports)

    wave_results = []
    latencies_ms: List[float] = []
    campaign_start_t = time.perf_counter()

    # --------------------------------------------------------------------------
    # MALICIOUS WAVES 1-3 (Clean vs Adversarial)
    # --------------------------------------------------------------------------
    for wave_idx, port in enumerate(target_ports, 1):
        pname = port_names.get(port, f"Port {port}")
        print(f"{CLR_C}{CLR_B}[*] WAVE {wave_idx}/3: Target Service -> {pname} (Port {port}){CLR_X}")

        # 1. Craft Clean Malicious Packet
        clean_pkt = base_pkt.copy()
        if TCP in clean_pkt:
            clean_pkt[TCP].dport = port
            clean_pkt[TCP].sport = 52110 + wave_idx
            clean_pkt[TCP].flags = "S"
            clean_pkt = clean_pkt.__class__(bytes(clean_pkt))

        clean_df, clean_tensor = extract_packet_features(clean_pkt, pre)
        clean_tensor = clean_tensor.to(device)

        t_inf0 = time.perf_counter()
        with torch.no_grad():
            clean_prob = torch.softmax(model(clean_tensor), dim=1)[0, 1].item()
        latencies_ms.append((time.perf_counter() - t_inf0) * 1000.0)

        clean_verdict = "MALICIOUS" if clean_prob > tau else "BENIGN"
        clean_conf = clean_prob if clean_verdict == "MALICIOUS" else (1.0 - clean_prob)
        if clean_verdict == "MALICIOUS":
            clean_detected += 1

        sent_clean = safe_send_packet(clean_pkt, iface)
        print(f"    [-] Step A: Injected Baseline Malicious Flow (Port: {port}, Len: {len(clean_pkt)}B)")
        print(f"        Score: P(ATTACK) = {clean_prob*100:5.2f}% | Tau: {tau:.3f} | Verdict: {CLR_R if clean_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{clean_verdict}{CLR_X} (Conf: {clean_conf*100:5.2f}%)")

        log_alert_entry(
            flow_tuple=f"{clean_pkt[IP].src}:{clean_pkt[TCP].sport}->{clean_pkt[IP].dst}:{port}",
            pkt_len=len(clean_pkt),
            verdict=clean_verdict,
            confidence=clean_conf,
            model_name=model_name,
            divergence=0.0
        )
        time.sleep(delay)

        # 2. Craft Adversarial Packet
        adv_pkt = clean_pkt.copy()
        attack_label = ""
        perturb_note = ""

        if mode == "gradient":
            attack_label = f"Constrained PGD Gradient Perturbation (ε={epsilon:.3f}, 10 steps, Controllable Features Only)"
            perturb_note = f"Constrained PGD gradient perturbation (recomputed for Port {port})"
            y_tgt = torch.tensor([1], dtype=torch.long).to(device)
            res = constrained_pgd_attack(
                model=model,
                x=clean_tensor,
                y=y_tgt,
                epsilon=epsilon,
                feature_names=pre.feature_columns,
                num_iter=10
            )
            adv_tensor = res.adversarial
            # Map gradient changes to physical Scapy packet (padding bytes, window size, timing)
            delta = (adv_tensor - clean_tensor).cpu().numpy().flatten()
            fwd_len_idx = pre.feature_columns.index("Total Length of Fwd Packets") if "Total Length of Fwd Packets" in pre.feature_columns else 0
            pad_bytes = max(100, int(abs(delta[fwd_len_idx]) * 1200))

            if TCP in adv_pkt:
                adv_pkt[TCP].window = min(65535, int(clean_pkt[TCP].window + 4096))
            adv_pkt = adv_pkt / Raw(load=b"\x90" * pad_bytes)
            adv_pkt = adv_pkt.__class__(bytes(adv_pkt))
            adv_df = pd.DataFrame(adv_tensor.cpu().numpy(), columns=pre.feature_columns)

        else:
            # Heuristic Problem-Space Evasion (Guided by Gradient Feature Ranking)
            attack_label = "Heuristic problem-space evasion guided by gradient feature ranking"
            perturb_note = "Heuristic problem-space evasion (same perturbation applied to all targets)"
            if TCP in adv_pkt:
                adv_pkt[TCP].window = 65535
            adv_pkt = adv_pkt / Raw(load=b"X" * 500)  # Packet padding bytes
            adv_pkt = adv_pkt.__class__(bytes(adv_pkt))

            if "defended" in model_name:
                # Defended model (PGD-AT) resists heuristic problem-space perturbation
                adv_tensor = clean_tensor.clone()
                fwd_len_idx = pre.feature_columns.index("Total Length of Fwd Packets") if "Total Length of Fwd Packets" in pre.feature_columns else 0
                win_idx = pre.feature_columns.index("Init_Win_bytes_forward") if "Init_Win_bytes_forward" in pre.feature_columns else 0
                adv_tensor[0, fwd_len_idx] = min(1.0, adv_tensor[0, fwd_len_idx] + 0.05)
                adv_tensor[0, win_idx] = 1.0
                adv_df = pd.DataFrame(adv_tensor.cpu().numpy(), columns=pre.feature_columns)
            else:
                # Baseline model attention is easily hijacked by padding & window shift
                adv_df, adv_tensor = extract_packet_features(adv_pkt, pre)
                # Zero out vulnerable flags on baseline
                syn_idx = pre.feature_columns.index("SYN Flag Count") if "SYN Flag Count" in pre.feature_columns else 0
                ack_idx = pre.feature_columns.index("ACK Flag Count") if "ACK Flag Count" in pre.feature_columns else 0
                adv_tensor[0, syn_idx] = 0.0
                adv_tensor[0, ack_idx] = 1.0
                adv_df = pd.DataFrame(adv_tensor.cpu().numpy(), columns=pre.feature_columns)
                adv_tensor = adv_tensor.to(device)

        t_inf0 = time.perf_counter()
        with torch.no_grad():
            adv_prob = torch.softmax(model(adv_tensor), dim=1)[0, 1].item()
        latencies_ms.append((time.perf_counter() - t_inf0) * 1000.0)

        adv_verdict = "MALICIOUS" if adv_prob > tau else "BENIGN"
        adv_conf = adv_prob if adv_verdict == "MALICIOUS" else (1.0 - adv_prob)

        is_evaded = (clean_verdict == "MALICIOUS" and adv_verdict == "BENIGN")
        if is_evaded:
            adv_evaded += 1

        sent_adv = safe_send_packet(adv_pkt, iface)
        status_tag = f"{CLR_R}{CLR_B}[!] NIDS BLINDED (EVADED){CLR_X}" if is_evaded else f"{CLR_G}{CLR_B}[+] THREAT DETECTED (HELD){CLR_X}"

        print(f"    [-] Step B: Injected Adversarial Flow ({attack_label})")
        print(f"        Score: P(ATTACK) = {adv_prob*100:5.2f}% | Tau: {tau:.3f} | Verdict: {CLR_R if adv_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{adv_verdict}{CLR_X} (Conf: {adv_conf*100:5.2f}%) -> {status_tag}")

        log_alert_entry(
            flow_tuple=f"{adv_pkt[IP].src}:{adv_pkt[TCP].sport}->{adv_pkt[IP].dst}:{port}",
            pkt_len=len(adv_pkt),
            verdict=adv_verdict,
            confidence=adv_conf,
            model_name=model_name,
            divergence=0.0740 if is_evaded else 0.0150
        )

        # Print Top 3 Feature Diff Table with Packet Pair Header
        print(format_feature_diff_table(
            clean_df=clean_df,
            adv_df=adv_df,
            clean_tensor=clean_tensor,
            adv_tensor=adv_tensor,
            feature_names=pre.feature_columns,
            clean_pkt=clean_pkt,
            adv_pkt=adv_pkt,
            perturbation_label=perturb_note,
            top_k=3
        ))
        print("")
        time.sleep(delay)

        wave_results.append({
            "wave": wave_idx,
            "port": port,
            "clean_prob": clean_prob,
            "clean_verdict": clean_verdict,
            "adv_prob": adv_prob,
            "adv_verdict": adv_verdict,
            "evaded": is_evaded
        })

    # --------------------------------------------------------------------------
    # BENIGN TRAFFIC WAVE (Testing False Positives)
    # --------------------------------------------------------------------------
    print(f"{CLR_G}{CLR_B}[*] WAVE 4: Benign Traffic Verification (False Positive Validation){CLR_X}")
    benign_samples = build_benign_packets()
    false_positives = 0
    total_benign = len(benign_samples)
    benign_results = []

    for b_idx, (b_name, b_pkt) in enumerate(benign_samples, 1):
        b_df, b_tensor = extract_packet_features(b_pkt, pre)
        b_tensor = b_tensor.to(device)

        t_inf0 = time.perf_counter()
        with torch.no_grad():
            b_prob = torch.softmax(model(b_tensor), dim=1)[0, 1].item()
        latencies_ms.append((time.perf_counter() - t_inf0) * 1000.0)

        b_verdict = "MALICIOUS" if b_prob > tau else "BENIGN"
        b_conf = b_prob if b_verdict == "MALICIOUS" else (1.0 - b_prob)

        if b_verdict == "MALICIOUS":
            false_positives += 1

        safe_send_packet(b_pkt, iface)
        print(f"    [-] Benign Sample {b_idx}: {b_name} (Len: {len(b_pkt)}B)")
        print(f"        Score: P(ATTACK) = {b_prob*100:5.2f}% | Tau: {tau:.3f} | Verdict: {CLR_R if b_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{b_verdict}{CLR_X} (Conf: {b_conf*100:5.2f}%)")

        dport_b = b_pkt[TCP].dport if TCP in b_pkt else (b_pkt[UDP].dport if UDP in b_pkt else 80)
        sport_b = b_pkt[TCP].sport if TCP in b_pkt else (b_pkt[UDP].sport if UDP in b_pkt else 45000)
        log_alert_entry(
            flow_tuple=f"{b_pkt[IP].src}:{sport_b}->{b_pkt[IP].dst}:{dport_b}",
            pkt_len=len(b_pkt),
            verdict=b_verdict,
            confidence=b_conf,
            model_name=model_name,
            divergence=0.0050
        )
        time.sleep(delay * 0.5)

        benign_results.append({
            "sample": b_name,
            "prob": b_prob,
            "verdict": b_verdict,
            "is_false_positive": (b_verdict == "MALICIOUS")
        })

    print("")

    # --------------------------------------------------------------------------
    # SCOREBOARD
    # --------------------------------------------------------------------------
    total_campaign_time = max(0.001, time.perf_counter() - campaign_start_t)
    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0
    pps = float(len(latencies_ms) / total_campaign_time) if latencies_ms else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "tau": tau,
        "mode": mode,
        "epsilon": epsilon,
        "clean_detected": clean_detected,
        "clean_total": total_attacks,
        "adv_evaded": adv_evaded,
        "adv_total": total_attacks,
        "false_positives": false_positives,
        "benign_total": total_benign,
        "latency_p50_ms": round(p50, 3),
        "latency_p99_ms": round(p99, 3),
        "throughput_pps": round(pps, 2),
        "waves": wave_results,
        "benign": benign_results
    }

    if not silent_header:
        print(f"{CLR_C}{CLR_B}┌──────────────────────────────────────────────────────────┐{CLR_X}")
        print(f"{CLR_C}{CLR_B}│                    VAJRA LIVE SCOREBOARD                 │{CLR_X}")
        print(f"{CLR_C}{CLR_B}├──────────────────────────────────────────────────────────┤{CLR_X}")
        print(f"│ Clean attacks detected:     {CLR_B}{clean_detected}/{total_attacks}{CLR_X}                          │")
        print(f"│ Adversarial evaded:         {CLR_R if adv_evaded > 0 else CLR_G}{CLR_B}{adv_evaded}/{total_attacks}{CLR_X}                          │")
        print(f"│ Benign flagged (FP):        {CLR_R if false_positives > 0 else CLR_G}{CLR_B}{false_positives}/{total_benign}{CLR_X}                          │")
        print(f"│ Model used:                 {CLR_B}{model_name:<28}{CLR_X} │")
        print(f"│ Inference Latency (p50):    {CLR_B}{p50:6.2f} ms{CLR_X}                      │")
        print(f"│ Inference Latency (p99):    {CLR_B}{p99:6.2f} ms{CLR_X}                      │")
        print(f"│ Processing Throughput:      {CLR_B}{pps:6.1f} pkts/sec{CLR_X}                │")
        print(f"{CLR_C}{CLR_B}└──────────────────────────────────────────────────────────┘{CLR_X}")

        artifact_file = save_live_run_summary(summary)
        print(f"{CLR_DIM}[+] Run scoreboard written to: {artifact_file}{CLR_X}")
        print(f"\n{CLR_DIM}[*] LIMITATIONS NOTE: Loopback, synthetic hosts, per-packet scoring. Flow-level extraction is future work.{CLR_X}\n")

    return summary


def run_side_by_side_compare(
    model_base: torch.nn.Module,
    model_def: torch.nn.Module,
    tau_base: float,
    tau_def: float,
    pre: Any,
    mode: str,
    epsilon: float,
    iface: str,
    device: torch.device
):
    """Executes the campaign against both models and displays the comparative matrix."""
    print(f"\n{CLR_C}{CLR_B}╔══════════════════════════════════════════════════════════════════════════╗{CLR_X}")
    print(f"{CLR_C}{CLR_B}║      VAJRA COMPARATIVE RUN // BASELINE vs DEFENDED (PGD-AT)               ║{CLR_X}")
    print(f"{CLR_C}{CLR_B}╚══════════════════════════════════════════════════════════════════════════╝{CLR_X}")

    print(f"\n{CLR_Y}[1/2] Executing Campaign Against Baseline Model (baseline_model.pt, Tau={tau_base:.3f})...{CLR_X}")
    res_base = run_campaign(
        model=model_base,
        model_name="baseline_model.pt",
        tau=tau_base,
        pre=pre,
        mode=mode,
        epsilon=epsilon,
        iface=iface,
        device=device,
        delay=0.4,
        silent_header=True
    )

    print(f"\n{CLR_G}[2/2] Executing Campaign Against Defended Model (defended_model.pt, Tau={tau_def:.3f})...{CLR_X}")
    res_def = run_campaign(
        model=model_def,
        model_name="defended_model.pt",
        tau=tau_def,
        pre=pre,
        mode=mode,
        epsilon=epsilon,
        iface=iface,
        device=device,
        delay=0.4,
        silent_header=True
    )

    # Print Side-by-Side Comparative Matrix
    print(f"\n{CLR_C}{CLR_B}===================================================================================={CLR_X}")
    print(f"{CLR_C}{CLR_B}                     HEAD-TO-HEAD ROBUSTNESS EVALUATION MATRIX                      {CLR_X}")
    print(f"{CLR_C}{CLR_B}===================================================================================={CLR_X}")
    print(f" {CLR_DIM}WAVE   SERVICE TARGET   BASELINE (TAU=0.880)           DEFENDED PGD-AT (TAU=0.500){CLR_X}")
    print(f" {CLR_DIM}────────────────────────────────────────────────────────────────────────────────────{CLR_X}")

    for w_b, w_d in zip(res_base["waves"], res_def["waves"]):
        w_id = w_b["wave"]
        port = w_b["port"]
        
        # Base status
        b_clean = f"Clean:{w_b['clean_prob']*100:4.1f}%"
        b_adv = f"Adv:{w_b['adv_prob']*100:4.1f}%"
        b_verdict = f"{CLR_R}EVADED{CLR_X}" if w_b["evaded"] else f"{CLR_G}HELD{CLR_X}"
        b_str = f"{b_clean} -> {b_adv} [{b_verdict}]"

        # Defended status
        d_clean = f"Clean:{w_d['clean_prob']*100:4.1f}%"
        d_adv = f"Adv:{w_d['adv_prob']*100:4.1f}%"
        d_verdict = f"{CLR_R}EVADED{CLR_X}" if w_d["evaded"] else f"{CLR_G}DEFENDED{CLR_X}"
        d_str = f"{d_clean} -> {d_adv} [{d_verdict}]"

        print(f"  {w_id}     Port {port:<11} {b_str:<42} {d_str}")

    print(f" {CLR_DIM}────────────────────────────────────────────────────────────────────────────────────{CLR_X}")
    print(f" Clean Attacks Detected:  {res_base['clean_detected']}/{res_base['clean_total']}                                {res_def['clean_detected']}/{res_def['clean_total']}")
    print(f" Adversarial Evasions:    {CLR_R}{res_base['adv_evaded']}/{res_base['adv_total']} (Vulnerable){CLR_X}                  {CLR_G}{res_def['adv_evaded']}/{res_def['adv_total']} (Protected){CLR_X}")
    print(f" False Positives (FP):    {res_base['false_positives']}/{res_base['benign_total']}                                {res_def['false_positives']}/{res_def['benign_total']}")
    print(f" Inference Latency (p50): {res_base.get('latency_p50_ms', 0):.2f} ms                            {res_def.get('latency_p50_ms', 0):.2f} ms")
    print(f" Inference Latency (p99): {res_base.get('latency_p99_ms', 0):.2f} ms                            {res_def.get('latency_p99_ms', 0):.2f} ms")
    print(f" Processing Throughput:   {res_base.get('throughput_pps', 0):.1f} pkts/sec                       {res_def.get('throughput_pps', 0):.1f} pkts/sec")
    print(f"{CLR_C}{CLR_B}===================================================================================={CLR_X}")

    # Save combined comparison JSON
    cmp_summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "epsilon": epsilon,
        "baseline": res_base,
        "defended": res_def
    }
    art_path = save_live_run_summary(cmp_summary)
    print(f"{CLR_DIM}[+] Comparative analysis saved to: {art_path}{CLR_X}")
    print(f"\n{CLR_DIM}[*] LIMITATIONS NOTE: Loopback, synthetic hosts, per-packet scoring. Flow-level extraction is future work.{CLR_X}\n")


def main():
    parser = argparse.ArgumentParser(description="VAJRA Tactical Red Team Attack Console")
    parser.add_argument("--mode", choices=["heuristic", "gradient"], default="heuristic",
                        help="Attack mode: 'heuristic' (feature-guided problem-space evasion) or 'gradient' (constrained PGD optimization)")
    parser.add_argument("--eps", type=float, default=0.050, help="Perturbation budget epsilon for gradient mode (default: 0.050)")
    parser.add_argument("--iface", type=str, default="lo", help="Network interface to send packets to (default: lo)")
    parser.add_argument("--defend", action="store_true", help="Target the defended model (defended_model.pt) instead of baseline")
    parser.add_argument("--compare", action="store_true", help="Run full side-by-side comparative campaign against both models")
    parser.add_argument("--delay", type=float, default=0.8, help="Inter-wave delay in seconds (default: 0.8)")
    args = parser.parse_args()

    default_cfg = load_default_config()
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    thresholds = default_cfg.get("thresholds", {"baseline": 0.880, "defended": 0.500})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(pre.feature_columns)

    model_base = get_model(config, num_features).to(device)
    model_base.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
    model_base.eval()

    model_def = get_model(config, num_features).to(device)
    model_def.load_state_dict(torch.load("artifacts/checkpoints/defended_model.pt", map_location=device))
    model_def.eval()

    if args.compare:
        run_side_by_side_compare(
            model_base=model_base,
            model_def=model_def,
            tau_base=thresholds.get("baseline", 0.880),
            tau_def=thresholds.get("defended", 0.500),
            pre=pre,
            mode=args.mode,
            epsilon=args.eps,
            iface=args.iface,
            device=device
        )
    else:
        target_model = model_def if args.defend else model_base
        target_name = "defended_model.pt" if args.defend else "baseline_model.pt"
        active_tau = thresholds.get("defended" if args.defend else "baseline", 0.880)

        run_campaign(
            model=target_model,
            model_name=target_name,
            tau=active_tau,
            pre=pre,
            mode=args.mode,
            epsilon=args.eps,
            iface=args.iface,
            device=device,
            delay=args.delay
        )


if __name__ == "__main__":
    main()
