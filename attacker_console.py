#!/usr/bin/env python
# VAJRA Tactical Red Team Exploit & Evaluation Console
# REWRITTEN: sends real multi-packet FLOWS (not single packets), perturbs
# packets on the wire (never a tensor after the fact), and scores through
# the exact same FlowBuilder + score_completed_flow path as nids_console.py.
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

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

from scapy.all import sendp
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Packet, Raw

from vajra.config import load_config

# Same single source of truth the NIDS console uses. No local reimplementation.
from vajra.console_helpers import (
    CLR_B, CLR_C, CLR_DIM, CLR_G, CLR_M, CLR_R, CLR_X, CLR_Y,
    load_default_config,
    log_alert_entry,
    save_live_run_summary,
    scapy_to_packet_info,
    score_completed_flow,
)
from vajra.flows.builder import FlowBuilder
from vajra.models.registry import get_model


def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def safe_send_packet(pkt: Packet, iface: str) -> bool:
    """sendp() operates at Layer 2 and needs an Ethernet header. Without one,
    the raw IP bytes get misread as a bogus L2 frame by anything sniffing
    with link-type EN10MB (which 'lo' reports as) — this was silently
    dropping every packet before it ever reached process_packet()."""
    try:
        if Ether not in pkt:
            pkt = Ether(dst="ff:ff:ff:ff:ff:ff", src="00:00:00:00:00:00", type=0x0800) / pkt
        sendp(pkt, iface=iface, verbose=False)
        return True
    except PermissionError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------
# Flow builders: produce a LIST of real packets forming one bidirectional
# flow. These are the only place "what does an attack look like on the
# wire" is decided — everything downstream just sends and scores packets.
# ---------------------------------------------------------------------
def build_syn_flood_flow(
    src_ip: str, dst_ip: str, sport: int, dport: int,
    n_packets: int = 6, inter_packet_delay: float = 0.02,
) -> List[Tuple[float, Packet]]:
    """A realistic one-way SYN flood burst: several SYNs, no replies,
    tight timing. Mirrors the shape of CIC-IDS2017 DoS/DDoS flows (many
    forward packets, ~0 backward packets, short inter-arrival times)."""
    packets = []
    t = 0.0
    for i in range(n_packets):
        pkt = (
            IP(src=src_ip, dst=dst_ip)
            / TCP(sport=sport, dport=dport, flags="S", window=1024, seq=1000 + i)
        )
        packets.append((t, pkt))
        t += inter_packet_delay
    return packets


def build_benign_flow(
    src_ip: str, dst_ip: str, sport: int, dport: int,
) -> List[Tuple[float, Packet]]:
    """A short, realistic bidirectional HTTP-style exchange: SYN, SYN-ACK,
    ACK, PSH-ACK (request), PSH-ACK (response), FIN. Gives the flow both
    forward AND backward packets, unlike a one-way flood."""
    packets = []
    packets.append((0.00, IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="S", window=29200, seq=100)))
    packets.append((0.01, IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="SA", window=29200, seq=500, ack=101)))
    packets.append((0.02, IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="A", window=29200, seq=101, ack=501)))
    packets.append((0.03, IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA", window=29200, seq=101, ack=501)
                     / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.org\r\n\r\n")))
    packets.append((0.05, IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="PA", window=29200, seq=501, ack=150)
                     / Raw(load=b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello world!")))
    packets.append((0.06, IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="FA", window=29200, seq=150, ack=562)))
    return packets


def apply_wire_perturbation(
    packets: List[Tuple[float, Packet]],
    mode: str,
    epsilon: float,
) -> List[Tuple[float, Packet]]:
    """Perturbs the ACTUAL packets of a flow — padding, timing, window size —
    before they are sent. This is the ONLY place the attack is modified.
    Nothing downstream is allowed to touch a tensor to fake an effect.

    Both modes keep the attack functionally a SYN flood: SYN flags are
    preserved, destination stays the same. Only attacker-controllable
    wire properties change (payload padding, inter-packet delay, window).
    """
    out = []
    for i, (t, pkt) in enumerate(packets):
        new_pkt = pkt.copy()

        if mode == "gradient":
            # Scale padding/timing with epsilon so larger budgets perturb more.
            pad_bytes = max(0, int(epsilon * 2000))  # e.g. eps=0.05 -> 100B padding
            delay_jitter = epsilon * 0.5  # seconds of extra spacing
            if TCP in new_pkt:
                new_pkt[TCP].window = min(65535, int(new_pkt[TCP].window + epsilon * 60000))
            if pad_bytes > 0:
                new_pkt = new_pkt / Raw(load=b"\x00" * pad_bytes)
            new_pkt = new_pkt.__class__(bytes(new_pkt))  # re-serialize so lengths are consistent
            out.append((t + i * delay_jitter, new_pkt))

        else:  # heuristic
            # Fixed-size padding + fixed window bump, same for every packet —
            # a simple, non-gradient "problem-space" perturbation.
            if TCP in new_pkt:
                new_pkt[TCP].window = 65535
            new_pkt = new_pkt / Raw(load=b"\x90" * 80)
            new_pkt = new_pkt.__class__(bytes(new_pkt))
            out.append((t, new_pkt))

    return out


def replay_flow_and_score(
    packets: List[Tuple[float, Packet]],
    flow_builder: FlowBuilder,
    preprocessor: Any,
    model: torch.nn.Module,
    device: torch.device,
    iface: str,
    send_packets: bool = True,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[torch.Tensor], Optional[float]]:
    """Sends (optionally) each packet of a flow with real relative timing,
    feeds every one through scapy_to_packet_info + FlowBuilder exactly as
    the NIDS console would see them off the wire, and scores the flow only
    once it completes. Returns (None, ...) if the flow never completed
    within this call (e.g. no FIN/RST and timeout not yet reached) —
    caller must flush explicitly in that case.
    """
    start = time.perf_counter()
    completed_feats = None
    last_flow_id = None

    for t_offset, pkt in packets:
        target_time = start + t_offset
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        if send_packets:
            safe_send_packet(pkt, iface)

        # Re-parse the packet exactly as the NIDS console's sniffer would.
        # This guarantees the attacker and the NIDS agree on what was sent.
        p_info = scapy_to_packet_info(pkt)
        if p_info is None:
            continue
        last_flow_id = f"{p_info.src_ip}:{p_info.src_port}->{p_info.dst_ip}:{p_info.dst_port}"
        result = flow_builder.process_packet(p_info)
        if result is not None:
            completed_feats = result

    if completed_feats is None:
        return None, None, None, None

    df, x_tensor, prob = score_completed_flow(completed_feats, preprocessor, model, device)
    return completed_feats, df, x_tensor, prob


def top_feature_diff_table(
    clean_df: pd.DataFrame, adv_df: pd.DataFrame, feature_names: List[str], top_k: int = 5,
) -> str:
    """Diffs two REAL (re-extracted, not fabricated) flow feature vectors
    in raw units, sorted by absolute change."""
    c = clean_df.iloc[0]
    a = adv_df.iloc[0]
    deltas = (a - c).abs().sort_values(ascending=False)

    lines = []
    lines.append(f"  {CLR_DIM}┌──────────────────────────────┬────────────┬────────────┬────────────┐{CLR_X}")
    lines.append(f"  {CLR_DIM}│{CLR_X} {CLR_B}FEATURE (RAW UNITS){CLR_X}           {CLR_DIM}│{CLR_X} {CLR_B}CLEAN{CLR_X}      {CLR_DIM}│{CLR_X} {CLR_B}ADVERSAL{CLR_X}   {CLR_DIM}│{CLR_X} {CLR_B}DELTA{CLR_X}      {CLR_DIM}│{CLR_X}")
    lines.append(f"  {CLR_DIM}├──────────────────────────────┼────────────┼────────────┼────────────┤{CLR_X}")
    for fname in deltas.index[:top_k]:
        c_val, a_val, d_val = c[fname], a[fname], deltas[fname]
        fname_trunc = (fname[:29] + "..") if len(fname) > 29 else fname
        lines.append(f"  {CLR_DIM}│{CLR_X} {fname_trunc:<30} {CLR_DIM}│{CLR_X} {c_val:10.2f} {CLR_DIM}│{CLR_X} {a_val:10.2f} {CLR_DIM}│{CLR_X} {CLR_M}{d_val:10.2f}{CLR_X} {CLR_DIM}│{CLR_X}")
    lines.append(f"  {CLR_DIM}└──────────────────────────────┴────────────┴────────────┴────────────┘{CLR_X}")
    return "\n".join(lines)


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
    silent_header: bool = False,
    flow_timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    if not silent_header:
        print(f"\n{CLR_M}{CLR_B}╔════════════════════════════════════════════════════════════════════╗{CLR_X}")
        print(f"{CLR_M}{CLR_B}║              VAJRA RED TEAM ADVERSARIAL PIPELINE (flow-based)      ║{CLR_X}")
        print(f"{CLR_M}{CLR_B}║     Target Model: {model_name:<20} | Calibrated Tau: {tau:<7.3f}    ║{CLR_X}")
        print(f"{CLR_M}{CLR_B}╚════════════════════════════════════════════════════════════════════╝{CLR_X}")
        print(f"  {CLR_DIM}► Campaign Mode:{CLR_X}     {CLR_B}{mode.upper()}{CLR_X}")
        print(f"  {CLR_DIM}► Perturbation (ε):{CLR_X}  {CLR_B}{epsilon:.3f}{CLR_X}")
        print(f"  {CLR_DIM}► Egress Interface:{CLR_X}  {CLR_B}{iface}{CLR_X}")
        print(f"  {CLR_DIM}► Execution Device:{CLR_X}  {CLR_B}{str(device).upper()}{CLR_X}")
        print(f"{CLR_M}{CLR_DIM}──────────────────────────────────────────────────────────────────────{CLR_X}\n")

    target_ports = [80, 22, 53]
    port_names = {80: "HTTP Web Service", 22: "SSH Management", 53: "DNS Name Resolution"}

    clean_detected = 0
    adv_evaded = 0
    total_attacks = len(target_ports)
    wave_results = []
    latencies_ms: List[float] = []
    campaign_start_t = time.perf_counter()

    for wave_idx, port in enumerate(target_ports, 1):
        pname = port_names.get(port, f"Port {port}")
        print(f"{CLR_C}{CLR_B}[*] WAVE {wave_idx}/3: Target Service -> {pname} (Port {port}){CLR_X}")

        src_ip, dst_ip = "192.168.1.105", "192.168.1.1"
        sport = 52100 + wave_idx

        # --- CLEAN FLOW: real multi-packet SYN flood burst ---
        clean_packets = build_syn_flood_flow(src_ip, dst_ip, sport, port)
        clean_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

        t0 = time.perf_counter()
        feats, clean_df, clean_tensor, clean_prob = replay_flow_and_score(
            clean_packets, clean_builder, pre, model, device, iface
        )
        if feats is None:
            # Flow never completed naturally (no FIN/RST in a flood) — flush it.
            expired = clean_builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
            if expired:
                _, feats = expired[0]
                clean_df, clean_tensor, clean_prob = score_completed_flow(feats, pre, model, device)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if clean_prob is None:
            print(f"    {CLR_R}[!] Clean flow did not complete — skipping wave.{CLR_X}")
            continue

        clean_verdict = "MALICIOUS" if clean_prob > tau else "BENIGN"
        clean_conf = clean_prob if clean_verdict == "MALICIOUS" else (1.0 - clean_prob)
        if clean_verdict == "MALICIOUS":
            clean_detected += 1

        print(f"    [-] Step A: Sent clean SYN-flood flow ({len(clean_packets)} packets, port {port})")
        print(f"        Score: P(ATTACK) = {clean_prob*100:5.2f}% | Tau: {tau:.3f} | "
              f"Verdict: {CLR_R if clean_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{clean_verdict}{CLR_X} (Conf: {clean_conf*100:5.2f}%)")

        log_alert_entry(
            flow_tuple=f"{src_ip}:{sport}->{dst_ip}:{port}", pkt_len=int(feats.get("Total Length of Fwd Packets", 0)),
            verdict=clean_verdict, confidence=clean_conf, model_name=model_name, divergence=0.0,
        )
        time.sleep(delay)

        # --- ADVERSARIAL FLOW: SAME attack, perturbed on the wire ---
        adv_sport = sport + 1000  # distinct 5-tuple so it's a fresh flow
        base_adv_packets = build_syn_flood_flow(src_ip, dst_ip, adv_sport, port)
        adv_packets = apply_wire_perturbation(base_adv_packets, mode=mode, epsilon=epsilon)
        adv_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

        attack_label = (
            f"Constrained-style wire perturbation (ε={epsilon:.3f}): padding + timing + window"
            if mode == "gradient"
            else "Heuristic wire perturbation: fixed padding (80B) + window bump"
        )

        t0 = time.perf_counter()
        adv_feats, adv_df, adv_tensor, adv_prob = replay_flow_and_score(
            adv_packets, adv_builder, pre, model, device, iface
        )
        if adv_feats is None:
            expired = adv_builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
            if expired:
                _, adv_feats = expired[0]
                adv_df, adv_tensor, adv_prob = score_completed_flow(adv_feats, pre, model, device)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if adv_prob is None:
            print(f"    {CLR_R}[!] Adversarial flow did not complete — skipping diff.{CLR_X}")
            continue

        adv_verdict = "MALICIOUS" if adv_prob > tau else "BENIGN"
        adv_conf = adv_prob if adv_verdict == "MALICIOUS" else (1.0 - adv_prob)

        # Honest evasion check: ONLY true if clean was caught AND adversarial was missed.
        is_evaded = (clean_verdict == "MALICIOUS" and adv_verdict == "BENIGN")
        if is_evaded:
            adv_evaded += 1

        status_tag = (
            f"{CLR_R}{CLR_B}[!] NIDS BLINDED (EVADED){CLR_X}" if is_evaded
            else f"{CLR_G}{CLR_B}[+] THREAT DETECTED (HELD){CLR_X}"
        )

        print(f"    [-] Step B: Sent adversarial flow ({attack_label})")
        print(f"        Score: P(ATTACK) = {adv_prob*100:5.2f}% | Tau: {tau:.3f} | "
              f"Verdict: {CLR_R if adv_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{adv_verdict}{CLR_X} "
              f"(Conf: {adv_conf*100:5.2f}%) -> {status_tag}")

        log_alert_entry(
            flow_tuple=f"{src_ip}:{adv_sport}->{dst_ip}:{port}", pkt_len=int(adv_feats.get("Total Length of Fwd Packets", 0)),
            verdict=adv_verdict, confidence=adv_conf, model_name=model_name,
            divergence=0.0,  # divergence is computed in the NIDS console, not here
        )

        print(top_feature_diff_table(clean_df, adv_df, pre.feature_columns, top_k=5))
        print("")
        time.sleep(delay)

        wave_results.append({
            "wave": wave_idx, "port": port,
            "clean_prob": clean_prob, "clean_verdict": clean_verdict,
            "adv_prob": adv_prob, "adv_verdict": adv_verdict,
            "evaded": is_evaded,
        })

    # --- BENIGN WAVE: real bidirectional flows, for false-positive check ---
    print(f"{CLR_G}{CLR_B}[*] WAVE 4: Benign Traffic Verification (False Positive Validation){CLR_X}")
    benign_results = []
    false_positives = 0
    benign_specs = [
        ("HTTP-style exchange (Port 80)", 80),
        ("HTTPS-style exchange (Port 443)", 443),
        ("DNS-style exchange (Port 53)", 53),
    ]
    for b_idx, (b_name, b_port) in enumerate(benign_specs, 1):
        b_sport = 60000 + b_idx
        b_packets = build_benign_flow("192.168.1.50", "192.168.1.1", b_sport, b_port)
        b_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

        t0 = time.perf_counter()
        b_feats, b_df, b_tensor, b_prob = replay_flow_and_score(b_packets, b_builder, pre, model, device, iface)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if b_prob is None:
            print(f"    {CLR_R}[!] Benign flow {b_idx} did not complete (check FIN handling).{CLR_X}")
            continue

        b_verdict = "MALICIOUS" if b_prob > tau else "BENIGN"
        b_conf = b_prob if b_verdict == "MALICIOUS" else (1.0 - b_prob)
        if b_verdict == "MALICIOUS":
            false_positives += 1

        print(f"    [-] Benign Sample {b_idx}: {b_name}")
        print(f"        Score: P(ATTACK) = {b_prob*100:5.2f}% | Tau: {tau:.3f} | "
              f"Verdict: {CLR_R if b_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{b_verdict}{CLR_X} (Conf: {b_conf*100:5.2f}%)")

        log_alert_entry(
            flow_tuple=f"192.168.1.50:{b_sport}->192.168.1.1:{b_port}",
            pkt_len=int(b_feats.get("Total Length of Fwd Packets", 0)),
            verdict=b_verdict, confidence=b_conf, model_name=model_name, divergence=0.0,
        )
        benign_results.append({"sample": b_name, "prob": b_prob, "verdict": b_verdict, "is_false_positive": b_verdict == "MALICIOUS"})
        time.sleep(delay * 0.5)

    print("")

    total_campaign_time = max(0.001, time.perf_counter() - campaign_start_t)
    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0
    flows_per_sec = float(len(latencies_ms) / total_campaign_time) if latencies_ms else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(), "model": model_name, "tau": tau,
        "mode": mode, "epsilon": epsilon,
        "clean_detected": clean_detected, "clean_total": total_attacks,
        "adv_evaded": adv_evaded, "adv_total": total_attacks,
        "false_positives": false_positives, "benign_total": len(benign_specs),
        "latency_p50_ms": round(p50, 3), "latency_p99_ms": round(p99, 3),
        "flow_throughput_per_sec": round(flows_per_sec, 2),
        "waves": wave_results, "benign": benign_results,
    }

    if not silent_header:
        print(f"{CLR_C}{CLR_B}┌──────────────────────────────────────────────────────────┐{CLR_X}")
        print(f"{CLR_C}{CLR_B}│                    VAJRA LIVE SCOREBOARD (flow-based)     │{CLR_X}")
        print(f"{CLR_C}{CLR_B}├──────────────────────────────────────────────────────────┤{CLR_X}")
        print(f"│ Clean attacks detected:     {CLR_B}{clean_detected}/{total_attacks}{CLR_X}")
        print(f"│ Adversarial evaded:         {CLR_R if adv_evaded > 0 else CLR_G}{CLR_B}{adv_evaded}/{total_attacks}{CLR_X}")
        print(f"│ Benign flagged (FP):        {CLR_R if false_positives > 0 else CLR_G}{CLR_B}{false_positives}/{len(benign_specs)}{CLR_X}")
        print(f"│ Model used:                 {CLR_B}{model_name}{CLR_X}")
        print(f"│ Flow Latency (p50):         {CLR_B}{p50:6.2f} ms{CLR_X}")
        print(f"│ Flow Latency (p99):         {CLR_B}{p99:6.2f} ms{CLR_X}")
        print(f"│ Flow Throughput:            {CLR_B}{flows_per_sec:6.2f} flows/sec{CLR_X}")
        print(f"{CLR_C}{CLR_B}└──────────────────────────────────────────────────────────┘{CLR_X}")

        artifact_file = save_live_run_summary(summary)
        print(f"{CLR_DIM}[+] Run scoreboard written to: {artifact_file}{CLR_X}")
        print(f"\n{CLR_DIM}[*] NOTE: Scoring is per-FLOW (multi-packet, CIC-IDS2017 semantics), "
              f"not per-packet. Perturbations are applied to real packets on the wire, "
              f"then re-extracted — scores are never fabricated.{CLR_X}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="VAJRA Tactical Red Team Attack Console")
    parser.add_argument("--mode", choices=["heuristic", "gradient"], default="heuristic")
    parser.add_argument("--eps", type=float, default=0.050)
    parser.add_argument("--iface", type=str, default="lo")
    parser.add_argument("--defend", action="store_true")
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--flow-timeout", type=float, default=3.0)
    args = parser.parse_args()

    default_cfg = load_default_config()
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))
    thresholds = default_cfg.get("thresholds", {"baseline": 0.880, "defended": 0.500})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(pre.feature_columns)

    target_name = "defended_model.pt" if args.defend else "baseline_model.pt"
    model = get_model(config, num_features).to(device)
    model.load_state_dict(torch.load(f"artifacts/checkpoints/{target_name}", map_location=device))
    model.eval()
    active_tau = thresholds.get("defended" if args.defend else "baseline", 0.880)

    run_campaign(
        model=model, model_name=target_name, tau=active_tau, pre=pre,
        mode=args.mode, epsilon=args.eps, iface=args.iface, device=device,
        delay=args.delay, flow_timeout_sec=args.flow_timeout,
    )


if __name__ == "__main__":
    main()
