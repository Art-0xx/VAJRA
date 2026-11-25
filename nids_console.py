#!/usr/bin/env python
# VAJRA Tactical Network Intrusion Detection Console
# REWRITTEN: real flow-based extraction, single codepath (no duplicate logic)
from __future__ import annotations

import argparse
import os
import sys
import threading
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

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

from vajra.config import load_config

# ---------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for packet parsing + scoring. Both this console
# and attacker_console.py MUST import these from console_helpers — never
# redefine them locally. That duplication is what caused the earlier
# 98.58% vs 99.98% mismatch between the two consoles.
# ---------------------------------------------------------------------
from vajra.console_helpers import (
    CLR_B, CLR_C, CLR_DIM, CLR_G, CLR_M, CLR_R, CLR_X, CLR_Y,
    load_default_config,
    log_alert_entry,
    scapy_to_packet_info,
    score_completed_flow,
)
from vajra.flows.builder import FlowBuilder, PacketInfo
from vajra.justification.divergence import compute_divergence
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution
from vajra.models.registry import get_model


class PacketDeduplicator:
    """Eliminates loopback (lo) echo packet capture duplicates."""

    def __init__(self, window_sec: float = 0.5, max_entries: int = 2000):
        self.window_sec = window_sec
        self.max_entries = max_entries
        self.seen: Dict[Tuple, float] = {}
        self.lock = threading.Lock()

    def is_duplicate(self, pkt: Any) -> bool:
        if not (IP in pkt and (TCP in pkt or UDP in pkt)):
            return False
        with self.lock:
            now = time.time()
            if len(self.seen) > self.max_entries:
                self.seen = {k: v for k, v in self.seen.items() if now - v < self.window_sec}

            proto = pkt[IP].proto
            src = pkt[IP].src
            dst = pkt[IP].dst
            ipid = getattr(pkt[IP], "id", 0)
            sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
            dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
            seq = getattr(pkt[TCP], "seq", 0) if TCP in pkt else 0
            payload_len = len(pkt[TCP].payload) if (TCP in pkt and pkt[TCP].payload) else 0

            sig = (src, dst, proto, sport, dport, ipid, seq, len(pkt), payload_len)
            if sig in self.seen and (now - self.seen[sig]) < self.window_sec:
                return True
            self.seen[sig] = now
            return False


def load_reference_profiles(num_features: int) -> Dict[int, np.ndarray]:
    """Loads the per-class attention reference profile built from real
    training-set flows (see scripts/build_reference_profile.py). Falls back
    to a uniform vector — and WARNS loudly — if the file is missing, since a
    uniform fallback makes every JSD reading downstream meaningless."""
    ref_path = Path("artifacts/reference_profile.npy")
    if ref_path.exists():
        profiles = np.load(ref_path, allow_pickle=True).item()
        print(f"{CLR_C}[*] Loaded reference attention profile from {ref_path} "
              f"(classes: {list(profiles.keys())}){CLR_X}")
        return {int(k): np.asarray(v, dtype=np.float32) for k, v in profiles.items()}
    else:
        print(f"{CLR_R}{CLR_B}[!] WARNING: artifacts/reference_profile.npy not found.{CLR_X}")
        print(f"{CLR_R}    Divergence gate will use a UNIFORM fallback and is NOT meaningful.{CLR_X}")
        print(f"{CLR_R}    Run scripts/build_reference_profile.py first.{CLR_X}")
        uniform = np.ones(num_features, dtype=np.float32) / num_features
        return {0: uniform.copy(), 1: uniform.copy()}


class ModelHolder:
    """Manages active Transformer backbone, runtime hot-swapping, and the
    per-flow FlowBuilder that turns raw packets into scored flows."""

    def __init__(self, initial_model: str, config: Dict[str, Any], preprocessor: Any,
                 device: torch.device, flow_timeout_sec: float = 3.0):
        self.lock = threading.Lock()
        self.device = device
        self.config = config
        self.pre = preprocessor
        self.num_features = len(preprocessor.feature_columns)

        default_cfg = load_default_config()
        self.thresholds = default_cfg.get("thresholds", {"baseline": 0.880, "defended": 0.500})
        # NOTE: this threshold is on the CORRECTED base-2 JSD (bounded [0,1]).
        # 0.150 was calibrated against the OLD natural-log JSD and is almost
        # certainly wrong now. Treat it as a placeholder until you run
        # scripts/calibrate_divergence_threshold.py against clean validation
        # flows and pick a value from the desired false-positive rate.
        self.div_thresh = float(default_cfg.get("divergence", {}).get("threshold", 0.150))

        print(f"{CLR_C}[*] Pre-mounting model weights in memory ({str(device).upper()})...{CLR_X}")
        self.model_base = get_model(config, self.num_features, store_attention=True).to(device)
        self.model_base.load_state_dict(torch.load("artifacts/checkpoints/baseline_model.pt", map_location=device))
        self.model_base.eval()

        self.model_def = get_model(config, self.num_features, store_attention=True).to(device)
        self.model_def.load_state_dict(torch.load("artifacts/checkpoints/defended_model.pt", map_location=device))
        self.model_def.eval()

        self.current_mode = initial_model.lower()
        self.active_model = self.model_def if self.current_mode == "defended" else self.model_base
        self.active_tau = self.thresholds.get(self.current_mode, 0.880)

        # Real per-class reference profiles (NOT a single synthetic packet).
        self.ref_profiles = load_reference_profiles(self.num_features)

        # One FlowBuilder accumulates real packets into real flows.
        self.flow_builder = FlowBuilder(
            feature_columns=preprocessor.feature_columns,
            flow_timeout_sec=flow_timeout_sec,
        )

    def toggle(self) -> str:
        with self.lock:
            if self.current_mode == "baseline":
                self.current_mode = "defended"
                self.active_model = self.model_def
                self.active_tau = self.thresholds.get("defended", 0.500)
            else:
                self.current_mode = "baseline"
                self.active_model = self.model_base
                self.active_tau = self.thresholds.get("baseline", 0.880)

            print(f"\n{CLR_B}{CLR_C}------------------------------------------------------------{CLR_X}")
            print(f"{CLR_B}{CLR_C}[HOT-SWAP] Model Backbone Switched: "
                  f"{CLR_G if self.current_mode == 'defended' else CLR_Y}{self.current_mode.upper()}"
                  f"{CLR_C} (Tau = {self.active_tau:.3f}){CLR_X}")
            print(f"{CLR_B}{CLR_C}------------------------------------------------------------{CLR_X}\n")
            return self.current_mode

    def get_active(self):
        with self.lock:
            return self.active_model, self.active_tau, self.current_mode


class NIDSCounters:
    """Tracks performance and live flow telemetry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_flows = 0
        self.malicious_detected = 0
        self.benign_detected = 0
        self.divergence_alerts = 0
        self.latencies_ms: List[float] = []
        self.start_time = time.time()

    def record(self, latency_ms: float, is_malicious: bool, is_divergent: bool):
        with self.lock:
            self.total_flows += 1
            if is_malicious:
                self.malicious_detected += 1
            else:
                self.benign_detected += 1
            if is_divergent:
                self.divergence_alerts += 1
            self.latencies_ms.append(latency_ms)


def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def handle_completed_flow(
    flow_id: str,
    flow_feats: Dict[str, float],
    model_holder: ModelHolder,
    counters: NIDSCounters,
):
    """Scores one COMPLETED flow (not a single packet). Only fires when
    FlowBuilder decides a flow has ended (FIN/RST) or timed out. This is the
    only place a verdict is produced — nids and attacker consoles both land
    here via the same score_completed_flow() helper."""
    t0 = time.perf_counter()

    active_model, active_tau, active_mode = model_holder.get_active()

    with torch.no_grad():
        df, x_tensor, prob_malicious = score_completed_flow(
            flow_feats, model_holder.pre, active_model, model_holder.device
        )
        print(f"{CLR_DIM}  [debug] fwd_pkts={flow_feats.get('Total Fwd Packets')} "
        f"bwd_pkts={flow_feats.get('Total Backward Packets')} "
        f"duration={flow_feats.get('Flow Duration')}{CLR_X}")
        maps = active_model.get_attention_maps()
        predicted_class = 1 if prob_malicious > 0.5 else 0
        if maps:
            rollout = compute_attention_rollout(maps)
            cur_attr = get_feature_attribution(rollout).cpu().numpy().flatten()
            ref = model_holder.ref_profiles.get(predicted_class, model_holder.ref_profiles[0])
            jsd = compute_divergence(ref, cur_attr)
        else:
            jsd = 0.0

    is_divergent = jsd > model_holder.div_thresh
    routed_to_defended = False

    if is_divergent and active_mode == "baseline":
        routed_to_defended = True
        with torch.no_grad():
            def_logits = model_holder.model_def(x_tensor)
            prob_malicious = torch.softmax(def_logits, dim=1)[0, 1].item()
        active_tau = model_holder.thresholds.get("defended", 0.500)

    is_malicious = prob_malicious > active_tau
    confidence = prob_malicious if is_malicious else (1.0 - prob_malicious)
    verdict = "MALICIOUS" if is_malicious else "BENIGN"
    dt_ms = (time.perf_counter() - t0) * 1000.0
    counters.record(dt_ms, is_malicious, is_divergent)

    total_bytes = int(
        flow_feats.get("Total Length of Fwd Packets", 0)
        + flow_feats.get("Total Length of Bwd Packets", 0)
    )

    log_alert_entry(
        flow_tuple=flow_id, pkt_len=total_bytes, verdict=verdict, confidence=confidence,
        model_name="defended (routed)" if routed_to_defended else active_mode,
        divergence=jsd,
    )

    t_str = get_timestamp()
    icon, color = ("🚨 [ALERT]", CLR_R) if is_malicious else ("✅ [INFO ]", CLR_G)
    print(f"\n[{t_str}] {color}{CLR_B}{icon} Flow: {flow_id} (Bytes: {total_bytes}B){CLR_X}")
    print(f"            Verdict:    {color}{CLR_B}{verdict}{CLR_X} | P(ATTACK): {prob_malicious*100:5.2f}% "
          f"| Tau: {active_tau:.3f} (Conf: {confidence*100:5.2f}%)")
    print(f"            Attention:  JSD = {jsd:.4f} (Gate: {model_holder.div_thresh:.3f})")
    if routed_to_defended:
        held = "THREAT HELD" if is_malicious else "no threat"
        print(f"            {CLR_Y}{CLR_B}[>] REROUTED TO DEFENDED MODEL (THREAT HELD){CLR_X}")
    elif is_divergent:
        print(f"            {CLR_Y}[?] SUSPICIOUS: attention shift detected (JSD={jsd:.4f} > {model_holder.div_thresh:.3f}){CLR_X}")


def process_packet(pkt: Any, model_holder: ModelHolder, counters: NIDSCounters, dedup: PacketDeduplicator):
    if not (IP in pkt and (TCP in pkt or UDP in pkt)):
        return
    if dedup.is_duplicate(pkt):
        return

    p_info = scapy_to_packet_info(pkt)
    if p_info is None:
        return

    # Accumulate into the flow. Only returns non-None when the flow COMPLETES
    # (FIN/RST seen, or a later packet triggers a prior flow's timeout).
    completed = model_holder.flow_builder.process_packet(p_info)
    if completed is not None:
        flow_id = f"{p_info.src_ip}:{p_info.src_port}->{p_info.dst_ip}:{p_info.dst_port}"
        handle_completed_flow(flow_id, completed, model_holder, counters)


def flush_idle_flows_periodically(model_holder: ModelHolder, counters: NIDSCounters, stop_event: threading.Event):
    """Background sweep so flows that go idle (no FIN/RST) still get scored
    instead of sitting in active_flows forever."""
    while not stop_event.is_set():
        time.sleep(1.0)
        expired = model_holder.flow_builder.flush_expired(current_time=time.time())
        for flow_id, feats in expired:
            handle_completed_flow(flow_id, feats, model_holder, counters)

def process_packet_safe(pkt, model_holder, counters, dedup):
    """TEMPORARY debug wrapper — Scapy swallows exceptions raised inside
    sniff()'s prn callback silently. This forces them to print so we can
    see why the console goes quiet with no errors."""
    try:
        process_packet(pkt, model_holder, counters, dedup)
    except Exception:
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="VAJRA Tactical NIDS Real-Time Console")
    parser.add_argument("--model", choices=["baseline", "defended"], default="baseline")
    parser.add_argument("--iface", type=str, default="lo")
    parser.add_argument("--defend", action="store_true")
    parser.add_argument("--flow-timeout", type=float, default=3.0,
                         help="Seconds of inactivity before a flow is scored (default: 3.0 for live demo; "
                              "CIC-IDS2017 itself uses much longer timeouts offline)")
    args = parser.parse_args()

    init_model = "defended" if args.defend else args.model.lower()

    default_cfg = load_default_config()
    version = default_cfg.get("version", "2.0.0")
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_holder = ModelHolder(initial_model=init_model, config=config, preprocessor=pre,
                                device=device, flow_timeout_sec=args.flow_timeout)
    counters = NIDSCounters()
    dedup = PacketDeduplicator(window_sec=0.5)

    _, active_tau, active_name = model_holder.get_active()

    print(f"\n{CLR_C}{CLR_B}╔════════════════════════════════════════════════════════════════════╗{CLR_X}")
    print(f"{CLR_C}{CLR_B}║        VAJRA TACTICAL NIDS SENSOR CONSOLE (flow-based)              ║{CLR_X}")
    print(f"{CLR_C}{CLR_B}║      Transformer Network Intrusion Detection & Defense v{version}      ║{CLR_X}")
    print(f"{CLR_C}{CLR_B}╚════════════════════════════════════════════════════════════════════╝{CLR_X}")
    print(f"  {CLR_DIM}► Initial Backbone:{CLR_X}  {CLR_B}{active_name.upper()}_MODEL.PT{CLR_X}")
    print(f"  {CLR_DIM}► Calibrated Tau:{CLR_X}    {CLR_B}{active_tau:.3f}{CLR_X}")
    print(f"  {CLR_DIM}► Divergence Gate:{CLR_X}   {CLR_B}JSD(base2) > {model_holder.div_thresh:.3f}{CLR_X} -> Defended Auto-Reroute")
    print(f"  {CLR_DIM}► Flow Timeout:{CLR_X}      {CLR_B}{args.flow_timeout:.1f}s{CLR_X} (verdicts arrive per-FLOW, not per-packet)")
    print(f"  {CLR_DIM}► Active Device:{CLR_X}     {CLR_B}{str(device).upper()}{CLR_X}")
    print(f"  {CLR_DIM}► Network Interface:{CLR_X} {CLR_B}{args.iface}{CLR_X}")
    print(f"  {CLR_DIM}► Hot-Swap Command:{CLR_X}  Type {CLR_B}'m' + Enter{CLR_X} anytime to toggle baseline <-> defended")
    print(f"{CLR_C}{CLR_DIM}──────────────────────────────────────────────────────────────────────{CLR_X}")
    print(f"{CLR_C}[*] Sniffing on '{args.iface}'... flows will appear once they complete or go idle.{CLR_X}\n")

    stop_event = threading.Event()

    def sniffer_thread():
        try:
            target_iface = args.iface
            from scapy.all import conf, get_if_list
            available = get_if_list()
            if target_iface not in available and target_iface == "lo" and os.name == "nt":
                target_iface = conf.iface
            sniff(
                iface=target_iface,
                prn=lambda pkt: process_packet_safe(pkt, model_holder, counters, dedup),
                store=False,
                stop_filter=lambda p: stop_event.is_set(),
            )
        except Exception as e:
            if not stop_event.is_set():
                print(f"{CLR_R}[-] Sniffer error: {e}{CLR_X}")
                print(f"{CLR_Y}[*] Tip: Ensure raw packet privileges (sudo) or check interface name.{CLR_X}")

    threading.Thread(target=sniffer_thread, daemon=True).start()
    threading.Thread(target=flush_idle_flows_periodically, args=(model_holder, counters, stop_event), daemon=True).start()

    try:
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            cmd = line.strip().lower()
            if cmd == "m":
                model_holder.toggle()
            elif cmd in ("q", "quit", "exit"):
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        total_time = max(0.001, time.time() - counters.start_time)
        print(f"\n\n{CLR_C}{CLR_B}======================================================================{CLR_X}")
        print(f"{CLR_C}{CLR_B}              VAJRA SENSOR PERFORMANCE SUMMARY                        {CLR_X}")
        print(f"{CLR_C}{CLR_B}======================================================================{CLR_X}")
        print(f"  Flows Scored:             {counters.total_flows}")
        print(f"  Malicious Alerts Fired:   {counters.malicious_detected}")
        print(f"  Benign Flows Verified:    {counters.benign_detected}")
        print(f"  Attention Shift Alerts:   {counters.divergence_alerts}")
        if counters.latencies_ms:
            p50 = np.percentile(counters.latencies_ms, 50)
            p99 = np.percentile(counters.latencies_ms, 99)
            fps = counters.total_flows / total_time
            print(f"  Inference Latency (p50):  {p50:.2f} ms  [per completed flow, not per packet]")
            print(f"  Inference Latency (p99):  {p99:.2f} ms")
            print(f"  Flow Throughput:          {fps:.2f} flows/sec")
        print(f"\n{CLR_DIM}[*] NOTE: Scoring is per-flow (CIC-IDS2017 semantics), not per-packet.{CLR_X}")
        print(f"{CLR_C}{CLR_B}======================================================================\n{CLR_X}")


if __name__ == "__main__":
    main()