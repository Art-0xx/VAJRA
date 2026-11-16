#!/usr/bin/env python
# VAJRA Tactical Network Intrusion Detection Console
from __future__ import annotations

import argparse
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw

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
    load_default_config,
    log_alert_entry,
)
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


class ModelHolder:
    """Manages active Transformer backbone and runtime hot-swapping."""

    def __init__(self, initial_model: str, config: Dict[str, Any], preprocessor: Any, device: torch.device):
        self.lock = threading.Lock()
        self.device = device
        self.config = config
        self.pre = preprocessor
        self.num_features = len(preprocessor.feature_columns)

        default_cfg = load_default_config()
        self.thresholds = default_cfg.get("thresholds", {"baseline": 0.880, "defended": 0.500})
        self.div_thresh = float(default_cfg.get("divergence", {}).get("threshold", 0.050))

        # Pre-load both baseline and defended weights for instantaneous zero-latency hot-swapping
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

        # Baseline nominal reference attention profile (clean unperturbed signature)
        pkt_ref = IP(src="192.168.1.105", dst="192.168.1.1") / TCP(sport=52110, dport=80, flags="S", window=1024) / Raw(b"EXPLOIT_PAYLOAD_BUFFER_OVERFLOW")
        _, x_ref = extract_packet_features(pkt_ref, preprocessor)
        with torch.no_grad():
            self.model_base(x_ref.to(device))
            ref_maps = self.model_base.get_attention_maps()
            if ref_maps:
                roll = compute_attention_rollout(ref_maps)
                self.ref_attention = get_feature_attribution(roll).cpu().numpy().flatten()
            else:
                self.ref_attention = np.ones(self.num_features, dtype=np.float32) / self.num_features

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
            print(f"{CLR_B}{CLR_C}[HOT-SWAP] Model Backbone Switched: {CLR_G if self.current_mode == 'defended' else CLR_Y}{self.current_mode.upper()}{CLR_C} (Tau = {self.active_tau:.3f}){CLR_X}")
            print(f"{CLR_B}{CLR_C}------------------------------------------------------------{CLR_X}\n")
            return self.current_mode

    def get_active(self):
        with self.lock:
            return self.active_model, self.active_tau, self.current_mode


class NIDSCounters:
    """Tracks performance and live packet telemetry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_packets = 0
        self.malicious_detected = 0
        self.benign_detected = 0
        self.divergence_alerts = 0
        self.latencies_ms: List[float] = []
        self.start_time = time.time()
        self.last_verdict: Optional[str] = None

    def record(self, latency_ms: float, is_malicious: bool, is_divergent: bool):
        with self.lock:
            self.total_packets += 1
            if is_malicious:
                self.malicious_detected += 1
            else:
                self.benign_detected += 1
            if is_divergent:
                self.divergence_alerts += 1
            self.latencies_ms.append(latency_ms)


def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def process_packet(
    pkt: Any,
    model_holder: ModelHolder,
    counters: NIDSCounters,
    pre: Any,
    dedup: PacketDeduplicator,
    verbose: bool = True
):
    if not (IP in pkt and (TCP in pkt or UDP in pkt)):
        return

    # Drop loopback / double-echo duplicate captures
    if dedup.is_duplicate(pkt):
        return

    t0 = time.perf_counter()
    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    pkt_len = len(pkt)
    flow_tuple = f"{src_ip}:{sport}->{dst_ip}:{dport}"

    # Extract 77 features and scale
    _, x_tensor = extract_packet_features(pkt, pre)
    x_tensor = x_tensor.to(model_holder.device)

    active_model, active_tau, active_mode = model_holder.get_active()

    # Model inference
    with torch.no_grad():
        logits = active_model(x_tensor)
        prob_malicious = torch.softmax(logits, dim=1)[0, 1].item()
        
        # Attention rollout and divergence
        maps = active_model.get_attention_maps()
        if maps:
            rollout = compute_attention_rollout(maps)
            cur_attr = get_feature_attribution(rollout).cpu().numpy().flatten()
            jsd = compute_divergence(model_holder.ref_attention, cur_attr)
        else:
            jsd = 0.0

    is_divergent = (jsd > model_holder.div_thresh)
    routed_to_defended = False

    # Runtime Attention Divergence Routing (Blueprint 5.6)
    if is_divergent and active_mode == "baseline":
        routed_to_defended = True
        with torch.no_grad():
            def_logits = model_holder.model_def(x_tensor)
            prob_def = torch.softmax(def_logits, dim=1)[0, 1].item()
        # Use defended prediction
        prob_malicious = prob_def
        active_tau = model_holder.thresholds.get("defended", 0.500)

    is_malicious = (prob_malicious > active_tau)
    confidence = prob_malicious if is_malicious else (1.0 - prob_malicious)
    verdict = "MALICIOUS" if is_malicious else "BENIGN"

    dt_ms = (time.perf_counter() - t0) * 1000.0
    counters.record(dt_ms, is_malicious, is_divergent)

    # Log alert to persistent JSONL (written per-flow)
    log_alert_entry(
        flow_tuple=flow_tuple,
        pkt_len=pkt_len,
        verdict=verdict,
        confidence=confidence,
        model_name="defended (routed)" if routed_to_defended else active_mode,
        divergence=jsd
    )

    # Print live detection stream
    t_str = get_timestamp()
    if is_malicious:
        print(f"\n[{t_str}] {CLR_R}{CLR_B}🚨 [ALERT] Flow: {flow_tuple} (Len: {pkt_len}B){CLR_X}")
        print(f"            Verdict:    {CLR_R}{CLR_B}MALICIOUS{CLR_X} | P(ATTACK): {prob_malicious*100:5.2f}% | Tau: {active_tau:.3f} (Conf: {confidence*100:5.2f}%)")
        print(f"            Attention:  JSD = {jsd:.4f} (Gate: {model_holder.div_thresh:.3f})")
        if routed_to_defended:
            print(f"            {CLR_Y}{CLR_B}[>] REROUTED TO DEFENDED MODEL (THREAT HELD){CLR_X}")
    else:
        print(f"\n[{t_str}] {CLR_G}{CLR_B}✅ [INFO ] Flow: {flow_tuple} (Len: {pkt_len}B){CLR_X}")
        print(f"            Verdict:    {CLR_G}{CLR_B}BENIGN{CLR_X} | P(ATTACK): {prob_malicious*100:5.2f}% | Tau: {active_tau:.3f} (Conf: {confidence*100:5.2f}%)")
        print(f"            Attention:  JSD = {jsd:.4f} (Gate: {model_holder.div_thresh:.3f})")
        if routed_to_defended:
            print(f"            {CLR_Y}{CLR_B}[>] REROUTED TO DEFENDED MODEL (THREAT HELD){CLR_X}")
        elif is_divergent:
            print(f"            {CLR_Y}[?] SUSPICIOUS: attention shift detected (JSD={jsd:.4f} > {model_holder.div_thresh:.3f}){CLR_X}")


def main():
    parser = argparse.ArgumentParser(description="VAJRA Tactical NIDS Real-Time Console")
    parser.add_argument("--model", choices=["baseline", "defended"], default="baseline",
                        help="Initial model backbone: baseline (baseline_model.pt) or defended (defended_model.pt)")
    parser.add_argument("--iface", type=str, default="lo", help="Network interface to monitor (default: lo)")
    parser.add_argument("--defend", action="store_true", help="Shortcut for --model defended")
    args = parser.parse_args()

    init_model = "defended" if args.defend else args.model.lower()

    default_cfg = load_default_config()
    version = default_cfg.get("version", "2.0.0")
    config = load_config(default_cfg.get("models", {}).get("config", "configs/model_transformer.yaml"))
    pre = joblib.load(default_cfg.get("models", {}).get("preprocessor", "artifacts/preprocessor.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_holder = ModelHolder(initial_model=init_model, config=config, preprocessor=pre, device=device)
    counters = NIDSCounters()
    dedup = PacketDeduplicator(window_sec=0.5)

    active_m, active_tau, active_name = model_holder.get_active()

    # Startup Banner
    print(f"\n{CLR_C}{CLR_B}╔════════════════════════════════════════════════════════════════════╗{CLR_X}")
    print(f"{CLR_C}{CLR_B}║              VAJRA TACTICAL NIDS SENSOR CONSOLE                    ║{CLR_X}")
    print(f"{CLR_C}{CLR_B}║      Transformer Network Intrusion Detection & Defense v{version}      ║{CLR_X}")
    print(f"{CLR_C}{CLR_B}╚════════════════════════════════════════════════════════════════════╝{CLR_X}")
    print(f"  {CLR_DIM}► Initial Backbone:{CLR_X}  {CLR_B}{active_name.upper()}_MODEL.PT{CLR_X} ({'PGD-AT Hardened' if active_name == 'defended' else 'Unhardened Baseline'})")
    print(f"  {CLR_DIM}► Calibrated Tau:{CLR_X}    {CLR_B}{active_tau:.3f}{CLR_X} (loaded from configs/default.yaml)")
    print(f"  {CLR_DIM}► Divergence Gate:{CLR_X}   {CLR_B}JSD > {model_holder.div_thresh:.3f}{CLR_X} -> Defended Auto-Reroute")
    print(f"  {CLR_DIM}► Active Device:{CLR_X}     {CLR_B}{str(device).upper()}{CLR_X}")
    print(f"  {CLR_DIM}► Network Interface:{CLR_X} {CLR_B}{args.iface}{CLR_X}")
    print(f"  {CLR_DIM}► Hot-Swap Command:{CLR_X}  Type {CLR_B}'m' + Enter{CLR_X} anytime to toggle baseline <-> defended")
    print(f"{CLR_C}{CLR_DIM}──────────────────────────────────────────────────────────────────────{CLR_X}")
    print(f"{CLR_C}[*] Sniffing live traffic on '{args.iface}'... Ready for incoming flows.{CLR_X}\n")

    # Start sniffing in background daemon thread
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
                prn=lambda pkt: process_packet(pkt, model_holder, counters, pre, dedup),
                store=False,
                stop_filter=lambda p: stop_event.is_set()
            )
        except Exception as e:
            if not stop_event.is_set():
                print(f"{CLR_R}[-] Sniffer error: {e}{CLR_X}")
                print(f"{CLR_Y}[*] Tip: Ensure raw packet privileges (e.g. sudo -v) or run in simulation mode.{CLR_X}")

    sniff_t = threading.Thread(target=sniffer_thread, daemon=True)
    sniff_t.start()

    # Interactive keyboard hot-swap handler
    try:
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            cmd = line.strip().lower()
            if cmd == "m":
                model_holder.toggle()
            elif cmd in ["q", "quit", "exit"]:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        total_time = max(0.001, time.time() - counters.start_time)
        print(f"\n\n{CLR_C}{CLR_B}======================================================================{CLR_X}")
        print(f"{CLR_C}{CLR_B}              VAJRA SENSOR PERFORMANCE SUMMARY                        {CLR_X}")
        print(f"{CLR_C}{CLR_B}======================================================================{CLR_X}")
        print(f"  Packets Processed:        {counters.total_packets}")
        print(f"  Malicious Alerts Fired:   {counters.malicious_detected}")
        print(f"  Benign Packets Verified:  {counters.benign_detected}")
        print(f"  Attention Shift Alerts:   {counters.divergence_alerts}")

        if counters.latencies_ms:
            p50 = np.percentile(counters.latencies_ms, 50)
            p99 = np.percentile(counters.latencies_ms, 99)
            pps = counters.total_packets / total_time
            print(f"  Inference Latency (p50):  {p50:.2f} ms")
            print(f"  Inference Latency (p99):  {p99:.2f} ms")
            print(f"  Throughput:               {pps:.1f} packets/sec")

        print(f"\n{CLR_DIM}[*] LIMITATIONS NOTE: Loopback, synthetic hosts, per-packet scoring. Flow-level extraction is future work.{CLR_X}")
        print(f"{CLR_C}{CLR_B}======================================================================\n{CLR_X}")


if __name__ == "__main__":
    main()
