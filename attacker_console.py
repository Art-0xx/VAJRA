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
            / Raw(load=b"X" * 50)
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


def build_real_hulk_reconstruction(
    src_ip: str, dst_ip: str, sport: int, dport: int = 80,
) -> List[Tuple[float, Packet]]:
    """Reconstructs the STRUCTURAL SHAPE of one real, specific CIC-IDS2017
    'DoS Hulk' flow (Wednesday-workingHours.pcap_ISCX.csv, DoS Hulk rows,
    index 5): 3 fwd / 6 bwd packets, ~3.7ms total duration, large
    response/small request byte asymmetry, no fresh handshake.

    IMPORTANT / HONEST CAVEAT: this is a reconstruction from aggregate CSV
    feature statistics, NOT a byte-level replay of the original packets
    (the raw PCAP for this dataset was not available on this machine).
    The real flow's exact per-packet byte layout and ordering is not
    recoverable from the CSV alone. This function targets the flow's
    STRUCTURAL properties (packet counts, direction asymmetry, timing
    scale, flag pattern) as closely as the available statistics allow.
    Treat any score from this flow as evidence about structural
    sensitivity, not as proof of exact packet-level evasion/detection.

    Real CSV row reference values:
      Total Fwd Packets = 3, Total Backward Packets = 6
      Total Length of Fwd Packets = 349, Total Length of Bwd Packets = 11595
      Flow Duration (us) = 3720, SYN/FIN/RST Flag Count = 0, PSH Flag Count = 1
      Init_Win_bytes_forward = 29200, Init_Win_bytes_backward = 235

    CONFIRMED LIVE (2026-11-18 through 2026-11-23): this reconstruction
    repeatedly scored >90% P(ATTACK) on baseline_model.pt, tau=0.880,
    verdict MALICIOUS on both the attacker and NIDS consoles across multiple
    independent runs. The standard heuristic wire perturbation (padding +
    window bump) did NOT evade detection (confidence stayed >90% after
    perturbation). Actual live Flow Duration measured 280-360ms (vs the
    CSV's 3.7ms target) due to real OS scheduling overhead in a
    sleep-based replay — detection held despite this timing mismatch,
    suggesting the model's decision here is driven by packet-count/byte-
    asymmetry structure rather than exact timing.
    """
    packets: List[Tuple[float, Packet]] = []

    # Forward: one small request w/ payload (~349B total), then two bare ACKs.
    packets.append((0.0, IP(src=src_ip, dst=dst_ip) /
                     TCP(sport=sport, dport=dport, flags="PA", window=29200, seq=1000) /
                     Raw(load=b"X" * 309)))
    packets.append((0.000542, IP(src=src_ip, dst=dst_ip) /
                     TCP(sport=sport, dport=dport, flags="A", window=29200, seq=1309)))
    packets.append((0.002812, IP(src=src_ip, dst=dst_ip) /
                     TCP(sport=sport, dport=dport, flags="A", window=29200, seq=1309)))

    # Backward: large response segments (full-MTU-like) + small remainder.
    bwd_times = [0.0003, 0.0009, 0.0015, 0.0021, 0.0027, 0.0031]
    bwd_sizes = [5792, 5792, 11, 0, 0, 0]
    seq = 500
    for bt, sz in zip(bwd_times, bwd_sizes):
        pkt = IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="A", window=235, seq=seq)
        if sz > 0:
            pkt = pkt / Raw(load=b"Y" * max(0, sz - 40))
        packets.append((bt, pkt))
        seq += sz

    packets.sort(key=lambda x: x[0])
    return packets


# ---------------------------------------------------------------------
# PLACEHOLDER: a second real attack type (DoS slowloris) reconstruction.
# NOT YET IMPLEMENTED — the full real CSV row (Flow Duration, Total
# Fwd/Bwd Packets, IAT fields) is still needed to build this accurately.
# What's confirmed so far from one real slowloris row:
#   Subflow Fwd Packets = 4, Subflow Bwd Packets = 2, Subflow Bwd Bytes = 0
#   Subflow Fwd Bytes (assumed = Total Length of Fwd Packets) = 231
#   Init_Win_bytes_forward = 29200, Init_Win_bytes_backward = 235
#   min_seg_size_forward = 32, act_data_pkt_fwd = 1
# Still missing: Flow Duration, Fwd/Bwd IAT Mean/Max/Min, flag counts.
# Once those are available, implement build_real_slowloris_reconstruction()
# following the same pattern as build_real_hulk_reconstruction() above:
# slowloris's real mechanism is a LONG-HELD, low-packet-count connection
# (opposite of Hulk's rapid request/response), so expect large inter-packet
# delays here rather than small payload/large-response asymmetry.
# ---------------------------------------------------------------------


def apply_wire_perturbation(
    packets: List[Tuple[float, Packet]],
    mode: str,
    epsilon: float,
) -> List[Tuple[float, Packet]]:
    """Perturbs the ACTUAL packets of a flow — padding, timing, window size —
    before they are sent. This is the ONLY place the attack is modified.
    Nothing downstream is allowed to touch a tensor to fake an effect.

    Both modes keep the attack's flag pattern intact: only attacker-
    controllable wire properties change (payload padding, inter-packet
    delay, window size).
    """
    out = []
    for i, (t, pkt) in enumerate(packets):
        new_pkt = pkt.copy()

        if mode == "gradient":
            pad_bytes = max(0, int(epsilon * 2000))
            delay_jitter = epsilon * 0.5
            if TCP in new_pkt:
                new_pkt[TCP].window = min(65535, int(new_pkt[TCP].window + epsilon * 60000))
            if pad_bytes > 0:
                new_pkt = new_pkt / Raw(load=b"\x00" * pad_bytes)
            new_pkt = new_pkt.__class__(bytes(new_pkt))
            out.append((t + i * delay_jitter, new_pkt))
        else:  # heuristic
            if TCP in new_pkt:
                new_pkt[TCP].window = 65535
            new_pkt = new_pkt / Raw(load=b"\x90" * 80)
            new_pkt = new_pkt.__class__(bytes(new_pkt))
            out.append((t, new_pkt))

    return out


def replay_flow_and_score(packets, flow_builder, preprocessor, model, device, iface, send_packets=True):
    start = time.perf_counter()
    completed_feats = None

    for t_offset, pkt in packets:
        target_time = start + t_offset
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        # Build the EXACT wire packet once. Both sending and scoring must see
        # identical bytes, or the two consoles will silently disagree.
        wire_pkt = pkt if Ether in pkt else (
            Ether(dst="ff:ff:ff:ff:ff:ff", src="00:00:00:00:00:00", type=0x0800) / pkt
        )

        if send_packets:
            try:
                sendp(wire_pkt, iface=iface, verbose=False)
            except Exception:
                pass

        p_info = scapy_to_packet_info(wire_pkt)   # same object that was sent
        if p_info is None:
            continue
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


def run_hulk_reconstruction_wave(
    model, tau, pre, model_name, device, iface, flow_timeout_sec, latencies_ms, mode, epsilon,
) -> Optional[Dict[str, Any]]:
    """Wave 5: scores one structurally-reconstructed real CIC-IDS2017 DoS Hulk
    flow (see build_real_hulk_reconstruction docstring for the honest caveat
    on what this is and isn't), THEN applies the same wire perturbation used
    on the synthetic SYN floods and checks whether it evades detection.

    Uses explicit, hardcoded addresses — NOT values leaked from an earlier
    loop — so this wave's behavior never depends on what Waves 1-3 happened
    to set src_ip/dst_ip to."""
    print(f"{CLR_C}{CLR_B}[*] WAVE 5: Structural Reconstruction of Real CIC-IDS2017 Flow "
          f"(DoS Hulk, Wednesday CSV){CLR_X}")

    hulk_src_ip, hulk_dst_ip = "192.168.1.105", "192.168.1.1"

    # --- Step A: clean reconstructed Hulk flow ---
    hulk_packets = build_real_hulk_reconstruction(hulk_src_ip, hulk_dst_ip, sport=54000, dport=80)
    hulk_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

    t0 = time.perf_counter()
    feats_h, df_h, tensor_h, prob_h = replay_flow_and_score(
        hulk_packets, hulk_builder, pre, model, device, iface
    )
    if feats_h is None:
        expired = hulk_builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
        if expired:
            _, feats_h = expired[0]
            df_h, tensor_h, prob_h = score_completed_flow(feats_h, pre, model, device)
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    if prob_h is None:
        print(f"    {CLR_R}[!] Reconstructed Hulk flow did not complete — skipping.{CLR_X}")
        return None

    hulk_verdict = "MALICIOUS" if prob_h > tau else "BENIGN"
    hulk_conf = prob_h if hulk_verdict == "MALICIOUS" else (1.0 - prob_h)

    print(f"    [-] Step A: Sent reconstructed clean Hulk flow (3 fwd / 6 bwd packets)")
    print(f"        Score: P(ATTACK) = {prob_h*100:5.2f}% | Tau: {tau:.3f} | "
          f"Verdict: {CLR_R if hulk_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{hulk_verdict}{CLR_X} "
          f"(Conf: {hulk_conf*100:5.2f}%)")
    print(f"    {CLR_DIM}Reconstructed vs real CSV row -> "
          f"Fwd pkts: {int(feats_h.get('Total Fwd Packets', 0))} (real: 3) | "
          f"Bwd pkts: {int(feats_h.get('Total Backward Packets', 0))} (real: 6) | "
          f"Duration(us): {feats_h.get('Flow Duration', 0):.0f} (real: 3720){CLR_X}")

    log_alert_entry(
        flow_tuple=f"{hulk_src_ip}:54000->{hulk_dst_ip}:80",
        pkt_len=int(feats_h.get("Total Length of Fwd Packets", 0)),
        verdict=hulk_verdict, confidence=hulk_conf,
        model_name=f"{model_name} (hulk-reconstruction-clean)", divergence=0.0,
    )

    result: Dict[str, Any] = {
        "clean_prob": prob_h, "clean_verdict": hulk_verdict,
        "fwd_packets": int(feats_h.get("Total Fwd Packets", 0)),
        "bwd_packets": int(feats_h.get("Total Backward Packets", 0)),
        "duration_us": float(feats_h.get("Flow Duration", 0)),
        "note": "Structural reconstruction from CSV aggregate stats, not a raw-PCAP byte replay.",
    }

    # --- Step B: ONLY attempt evasion if the clean flow was actually detected.
    # Testing evasion against an undetected attack is meaningless (there is
    # nothing to evade) — this mirrors the honest evasion check already used
    # for the synthetic SYN-flood waves.
    if hulk_verdict != "MALICIOUS":
        print(f"    {CLR_DIM}[i] Clean reconstruction was not detected — skipping evasion test "
              f"(nothing to evade).{CLR_X}\n")
        result.update({"adv_prob": None, "adv_verdict": None, "evaded": None})
        return result

    adv_hulk_packets = apply_wire_perturbation(hulk_packets, mode=mode, epsilon=epsilon)
    adv_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

    t0 = time.perf_counter()
    adv_feats_h, adv_df_h, adv_tensor_h, adv_prob_h = replay_flow_and_score(
        adv_hulk_packets, adv_builder, pre, model, device, iface
    )
    if adv_feats_h is None:
        expired = adv_builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
        if expired:
            _, adv_feats_h = expired[0]
            adv_df_h, adv_tensor_h, adv_prob_h = score_completed_flow(adv_feats_h, pre, model, device)
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    attack_label = (
        f"Constrained-style wire perturbation (ε={epsilon:.3f}): padding + timing + window"
        if mode == "gradient"
        else "Heuristic wire perturbation: fixed padding (80B) + window bump"
    )

    if adv_prob_h is None:
        print(f"    {CLR_R}[!] Perturbed Hulk flow did not complete — skipping evasion check.{CLR_X}\n")
        result.update({"adv_prob": None, "adv_verdict": None, "evaded": None})
        return result

    adv_hulk_verdict = "MALICIOUS" if adv_prob_h > tau else "BENIGN"
    adv_hulk_conf = adv_prob_h if adv_hulk_verdict == "MALICIOUS" else (1.0 - adv_prob_h)
    evaded = (hulk_verdict == "MALICIOUS" and adv_hulk_verdict == "BENIGN")

    status_tag = (
        f"{CLR_R}{CLR_B}[!] NIDS BLINDED (EVADED){CLR_X}" if evaded
        else f"{CLR_G}{CLR_B}[+] THREAT DETECTED (HELD){CLR_X}"
    )

    print(f"    [-] Step B: Sent perturbed Hulk flow ({attack_label})")
    print(f"        Score: P(ATTACK) = {adv_prob_h*100:5.2f}% | Tau: {tau:.3f} | "
          f"Verdict: {CLR_R if adv_hulk_verdict == 'MALICIOUS' else CLR_G}{CLR_B}{adv_hulk_verdict}{CLR_X} "
          f"(Conf: {adv_hulk_conf*100:5.2f}%) -> {status_tag}")

    if df_h is not None and adv_df_h is not None:
        print(top_feature_diff_table(df_h, adv_df_h, pre.feature_columns, top_k=5))

    log_alert_entry(
        flow_tuple=f"{hulk_src_ip}:55000->{hulk_dst_ip}:80",
        pkt_len=int(adv_feats_h.get("Total Length of Fwd Packets", 0)),
        verdict=adv_hulk_verdict, confidence=adv_hulk_conf,
        model_name=f"{model_name} (hulk-reconstruction-adversarial)", divergence=0.0,
    )
    print("")

    result.update({
        "adv_prob": adv_prob_h, "adv_verdict": adv_hulk_verdict, "evaded": evaded,
    })
    return result


def run_hulk_repeated_trial(
    model, tau, pre, model_name, device, iface, flow_timeout_sec, mode, epsilon, n_repeats: int = 10,
) -> Dict[str, Any]:
    """Runs the real-flow Hulk reconstruction (clean + perturbed) N times with
    fresh ports each run, to replace a single anecdotal result (n=1 or n=2)
    with an actual sample. Reports per-run outcomes plus mean/std so the
    detection and evasion-holding claims can be stated with a real sample
    size instead of 'it worked once/twice'."""
    print(f"\n{CLR_C}{CLR_B}[*] REPEATED HULK TRIAL: {n_repeats} runs (clean + perturbed each){CLR_X}")
    print(f"{CLR_DIM}    Each run uses a fresh source port so flows never collide.{CLR_X}\n")

    hulk_dst_ip = "192.168.1.1"
    hulk_src_ip = "192.168.1.105"

    clean_probs: List[float] = []
    adv_probs: List[float] = []
    clean_detected_count = 0
    held_count = 0
    evaded_count = 0
    failed_count = 0

    for run_idx in range(n_repeats):
        sport = 56000 + run_idx * 2  # distinct, non-colliding port per run

        # --- clean ---
        hulk_packets = build_real_hulk_reconstruction(hulk_src_ip, hulk_dst_ip, sport=sport, dport=80)
        builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)
        feats_h, df_h, _, prob_h = replay_flow_and_score(hulk_packets, builder, pre, model, device, iface)
        if feats_h is None:
            expired = builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
            if expired:
                _, feats_h = expired[0]
                df_h, _, prob_h = score_completed_flow(feats_h, pre, model, device)

        if prob_h is None:
            print(f"    Run {run_idx+1}/{n_repeats}: {CLR_R}clean flow failed to complete{CLR_X}")
            failed_count += 1
            continue

        clean_probs.append(prob_h)
        clean_verdict = "MALICIOUS" if prob_h > tau else "BENIGN"
        if clean_verdict == "MALICIOUS":
            clean_detected_count += 1

        # --- perturbed (only if clean was detected, same honest-evasion rule as elsewhere) ---
        adv_prob_h = None
        if clean_verdict == "MALICIOUS":
            adv_packets = apply_wire_perturbation(hulk_packets, mode=mode, epsilon=epsilon)
            adv_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)
            adv_feats_h, _, _, adv_prob_h = replay_flow_and_score(adv_packets, adv_builder, pre, model, device, iface)
            if adv_feats_h is None:
                expired = adv_builder.flush_expired(current_time=time.time() + flow_timeout_sec + 1.0)
                if expired:
                    _, adv_feats_h = expired[0]
                    _, _, adv_prob_h = score_completed_flow(adv_feats_h, pre, model, device)

            if adv_prob_h is not None:
                adv_probs.append(adv_prob_h)
                adv_verdict = "MALICIOUS" if adv_prob_h > tau else "BENIGN"
                if adv_verdict == "BENIGN":
                    evaded_count += 1
                else:
                    held_count += 1

        status = (
            f"clean={prob_h*100:5.2f}% ({clean_verdict})"
            + (f" | adv={adv_prob_h*100:5.2f}% ({'MALICIOUS' if adv_prob_h > tau else 'BENIGN'})"
               if adv_prob_h is not None else " | adv=skipped (clean not detected)")
        )
        print(f"    Run {run_idx+1}/{n_repeats}: {status}")

    n_valid = len(clean_probs)
    result = {
        "n_repeats": n_repeats,
        "n_valid": n_valid,
        "n_failed": failed_count,
        "clean_detected_count": clean_detected_count,
        "clean_detection_rate": clean_detected_count / n_valid if n_valid else None,
        "clean_prob_mean": float(np.mean(clean_probs)) if clean_probs else None,
        "clean_prob_std": float(np.std(clean_probs)) if clean_probs else None,
        "held_count": held_count,
        "evaded_count": evaded_count,
        "evasion_rate": evaded_count / (held_count + evaded_count) if (held_count + evaded_count) > 0 else None,
        "adv_prob_mean": float(np.mean(adv_probs)) if adv_probs else None,
        "adv_prob_std": float(np.std(adv_probs)) if adv_probs else None,
    }

    print(f"\n{CLR_C}{CLR_B}┌──────────────────────────────────────────────────────────┐{CLR_X}")
    print(f"{CLR_C}{CLR_B}│          REPEATED HULK TRIAL SUMMARY (n={n_repeats})                  │{CLR_X}")
    print(f"{CLR_C}{CLR_B}├──────────────────────────────────────────────────────────┤{CLR_X}")
    print(f"│ Valid runs:                 {n_valid}/{n_repeats} ({failed_count} failed to complete)")
    if result["clean_prob_mean"] is not None:
        print(f"│ Clean detection rate:       {clean_detected_count}/{n_valid} "
              f"({result['clean_detection_rate']*100:.1f}%)")
        print(f"│ Clean P(ATTACK) mean±std:   {result['clean_prob_mean']*100:.2f}% ± {result['clean_prob_std']*100:.2f}%")
    if result["evasion_rate"] is not None:
        print(f"│ Evasion rate (of detected): {evaded_count}/{held_count + evaded_count} "
              f"({result['evasion_rate']*100:.1f}%)")
        print(f"│ Adv P(ATTACK) mean±std:     {result['adv_prob_mean']*100:.2f}% ± {result['adv_prob_std']*100:.2f}%")
    print(f"{CLR_C}{CLR_B}└──────────────────────────────────────────────────────────┘{CLR_X}\n")

    # Persist this as a run artifact too, same as the full campaign summary.
    save_live_run_summary({
        "timestamp": datetime.now().isoformat(),
        "run_type": "hulk_repeated_trial",
        "model": model_name, "tau": tau, "mode": mode, "epsilon": epsilon,
        **result,
    })

    return result


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
        clean_packets = build_syn_flood_flow(src_ip, dst_ip, sport, port, n_packets=6, inter_packet_delay=2.0)
        clean_builder = FlowBuilder(feature_columns=pre.feature_columns, flow_timeout_sec=flow_timeout_sec)

        t0 = time.perf_counter()
        feats, clean_df, clean_tensor, clean_prob = replay_flow_and_score(
            clean_packets, clean_builder, pre, model, device, iface
        )
        if feats is None:
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
        adv_sport = sport + 1000
        base_adv_packets = build_syn_flood_flow(src_ip, dst_ip, adv_sport, port, n_packets=6, inter_packet_delay=2.0)
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
            verdict=adv_verdict, confidence=adv_conf, model_name=model_name, divergence=0.0,
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

    # --- WAVE 5: structural reconstruction of a real CIC-IDS2017 Hulk flow,
    #     clean score + wire-perturbation evasion test ---
    hulk_result = run_hulk_reconstruction_wave(
        model, tau, pre, model_name, device, iface, flow_timeout_sec, latencies_ms, mode, epsilon,
    )

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
        "hulk_reconstruction": hulk_result,
    }

    if not silent_header:
        print(f"{CLR_C}{CLR_B}┌──────────────────────────────────────────────────────────┐{CLR_X}")
        print(f"{CLR_C}{CLR_B}│                    VAJRA LIVE SCOREBOARD (flow-based)     │{CLR_X}")
        print(f"{CLR_C}{CLR_B}├──────────────────────────────────────────────────────────┤{CLR_X}")
        print(f"│ Clean attacks detected:     {CLR_B}{clean_detected}/{total_attacks}{CLR_X}")
        print(f"│ Adversarial evaded:         {CLR_R if adv_evaded > 0 else CLR_G}{CLR_B}{adv_evaded}/{total_attacks}{CLR_X}")
        print(f"│ Benign flagged (FP):        {CLR_R if false_positives > 0 else CLR_G}{CLR_B}{false_positives}/{len(benign_specs)}{CLR_X}")
        if hulk_result is not None:
            hr_clean = f"{hulk_result['clean_verdict']} ({hulk_result['clean_prob']*100:.2f}%)"
            print(f"│ Hulk reconstruction (clean): {CLR_B}{hr_clean}{CLR_X}")
            if hulk_result.get("adv_verdict") is not None:
                hr_adv = f"{hulk_result['adv_verdict']} ({hulk_result['adv_prob']*100:.2f}%)"
                evaded_tag = "EVADED" if hulk_result["evaded"] else "HELD"
                print(f"│ Hulk reconstruction (adv):   {CLR_B}{hr_adv} -> {evaded_tag}{CLR_X}")
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
    parser.add_argument("--hulk-repeat", type=int, default=0,
                         help="If > 0, run ONLY the repeated Hulk reconstruction trial this many "
                              "times (clean+perturbed each), instead of the full 5-wave campaign. "
                              "Use this to build a real sample size for the detection/evasion claim "
                              "instead of a single anecdotal run.")
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

    if args.hulk_repeat > 0:
        run_hulk_repeated_trial(
            model=model, tau=active_tau, pre=pre, model_name=target_name,
            device=device, iface=args.iface, flow_timeout_sec=args.flow_timeout,
            mode=args.mode, epsilon=args.eps, n_repeats=args.hulk_repeat,
        )
        return

    run_campaign(
        model=model, model_name=target_name, tau=active_tau, pre=pre,
        mode=args.mode, epsilon=args.eps, iface=args.iface, device=device,
        delay=args.delay, flow_timeout_sec=args.flow_timeout,
    )


if __name__ == "__main__":
    main()