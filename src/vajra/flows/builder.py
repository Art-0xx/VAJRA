# Bidirectional Network Flow Builder & 77-Feature Extractor (Blueprint Section 5.2)
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PacketInfo:
    """Parsed packet metadata for flow aggregation."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    header_length: int
    flags: str
    window_size: int
    payload_len: int


class BidirectionalFlow:
    """Tracks state and statistics for a single bidirectional TCP/UDP flow."""

    def __init__(self, first_packet: PacketInfo) -> None:
        self.flow_id = f"{first_packet.src_ip}:{first_packet.src_port}->{first_packet.dst_ip}:{first_packet.dst_port}"
        self.fwd_ip = first_packet.src_ip
        self.bwd_ip = first_packet.dst_ip
        self.fwd_port = first_packet.src_port
        self.bwd_port = first_packet.dst_port
        self.protocol = first_packet.protocol

        self.start_time = first_packet.timestamp
        self.last_time = first_packet.timestamp

        # Packet and length tracking
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []
        self.fwd_header_lengths: List[int] = []
        self.bwd_header_lengths: List[int] = []

        # Inter-arrival times
        self.all_timestamps: List[float] = []
        self.fwd_timestamps: List[float] = []
        self.bwd_timestamps: List[float] = []

        # Flags & TCP properties
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0
        self.cwe_count = 0
        self.ece_count = 0

        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0

        self.init_win_fwd = first_packet.window_size
        self.init_win_bwd = 0
        self.act_data_pkt_fwd = 0
        self.min_seg_size_forward = first_packet.header_length

        self.add_packet(first_packet)

    def is_forward(self, pkt: PacketInfo) -> bool:
        return pkt.src_ip == self.fwd_ip and pkt.src_port == self.fwd_port

    def add_packet(self, pkt: PacketInfo) -> None:
        self.last_time = pkt.timestamp
        self.all_timestamps.append(pkt.timestamp)
        is_fwd = self.is_forward(pkt)

        # Flag counts
        if "F" in pkt.flags:
            self.fin_count += 1
        if "S" in pkt.flags:
            self.syn_count += 1
        if "R" in pkt.flags:
            self.rst_count += 1
        if "P" in pkt.flags:
            self.psh_count += 1
            if is_fwd:
                self.fwd_psh += 1
            else:
                self.bwd_psh += 1
        if "A" in pkt.flags:
            self.ack_count += 1
        if "U" in pkt.flags:
            self.urg_count += 1
            if is_fwd:
                self.fwd_urg += 1
            else:
                self.bwd_urg += 1
        if "E" in pkt.flags:
            self.ece_count += 1
        if "C" in pkt.flags:
            self.cwe_count += 1

        if is_fwd:
            self.fwd_lengths.append(pkt.length)
            self.fwd_header_lengths.append(pkt.header_length)
            self.fwd_timestamps.append(pkt.timestamp)
            if pkt.payload_len > 0:
                self.act_data_pkt_fwd += 1
            self.min_seg_size_forward = min(self.min_seg_size_forward, pkt.header_length)
        else:
            self.bwd_lengths.append(pkt.length)
            self.bwd_header_lengths.append(pkt.header_length)
            self.bwd_timestamps.append(pkt.timestamp)
            if self.init_win_bwd == 0:
                self.init_win_bwd = pkt.window_size

    def to_cic_features(self, feature_columns: List[str]) -> Dict[str, float]:
        """Compute the full 77-feature dictionary matching CIC-IDS2017 schema."""
        duration_us = max((self.last_time - self.start_time) * 1_000_000.0, 1.0)
        duration_sec = duration_us / 1_000_000.0

        n_fwd = len(self.fwd_lengths)
        n_bwd = len(self.bwd_lengths)
        tot_fwd_len = sum(self.fwd_lengths)
        tot_bwd_len = sum(self.bwd_lengths)
        all_lengths = self.fwd_lengths + self.bwd_lengths

        # Packet Length Statistics
        fwd_max = max(self.fwd_lengths) if self.fwd_lengths else 0.0
        fwd_min = min(self.fwd_lengths) if self.fwd_lengths else 0.0
        fwd_mean = float(np.mean(self.fwd_lengths)) if self.fwd_lengths else 0.0
        fwd_std = float(np.std(self.fwd_lengths)) if len(self.fwd_lengths) > 1 else 0.0

        bwd_max = max(self.bwd_lengths) if self.bwd_lengths else 0.0
        bwd_min = min(self.bwd_lengths) if self.bwd_lengths else 0.0
        bwd_mean = float(np.mean(self.bwd_lengths)) if self.bwd_lengths else 0.0
        bwd_std = float(np.std(self.bwd_lengths)) if len(self.bwd_lengths) > 1 else 0.0

        pkt_min = min(all_lengths) if all_lengths else 0.0
        pkt_max = max(all_lengths) if all_lengths else 0.0
        pkt_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
        pkt_std = float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0
        pkt_var = float(np.var(all_lengths)) if len(all_lengths) > 1 else 0.0

        # Flow rates
        flow_bytes_s = (tot_fwd_len + tot_bwd_len) / duration_sec
        flow_pkts_s = (n_fwd + n_bwd) / duration_sec
        fwd_pkts_s = n_fwd / duration_sec
        bwd_pkts_s = n_bwd / duration_sec

        # Inter-arrival times (IATs) in microseconds
        def calc_iat(timestamps: List[float]) -> Tuple[float, float, float, float, float]:
            if len(timestamps) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            iats = [(timestamps[i] - timestamps[i - 1]) * 1_000_000.0 for i in range(1, len(timestamps))]
            return float(np.mean(iats)), float(np.std(iats)), float(max(iats)), float(min(iats)), float(sum(iats))

        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, _ = calc_iat(self.all_timestamps)
        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min, fwd_iat_tot = calc_iat(self.fwd_timestamps)
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, bwd_iat_tot = calc_iat(self.bwd_timestamps)

        # Build feature dictionary
        features: Dict[str, float] = {
            "Destination Port": float(self.bwd_port),
            "Flow Duration": float(duration_us),
            "Total Fwd Packets": float(n_fwd),
            "Total Backward Packets": float(n_bwd),
            "Total Length of Fwd Packets": float(tot_fwd_len),
            "Total Length of Bwd Packets": float(tot_bwd_len),
            "Fwd Packet Length Max": float(fwd_max),
            "Fwd Packet Length Min": float(fwd_min),
            "Fwd Packet Length Mean": float(fwd_mean),
            "Fwd Packet Length Std": float(fwd_std),
            "Bwd Packet Length Max": float(bwd_max),
            "Bwd Packet Length Min": float(bwd_min),
            "Bwd Packet Length Mean": float(bwd_mean),
            "Bwd Packet Length Std": float(bwd_std),
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Fwd IAT Total": float(fwd_iat_tot),
            "Fwd IAT Mean": float(fwd_iat_mean),
            "Fwd IAT Std": float(fwd_iat_std),
            "Fwd IAT Max": float(fwd_iat_max),
            "Fwd IAT Min": float(fwd_iat_min),
            "Bwd IAT Total": float(bwd_iat_tot),
            "Bwd IAT Mean": float(bwd_iat_mean),
            "Bwd IAT Std": float(bwd_iat_std),
            "Bwd IAT Max": float(bwd_iat_max),
            "Bwd IAT Min": float(bwd_iat_min),
            "Fwd PSH Flags": float(self.fwd_psh),
            "Bwd PSH Flags": float(self.bwd_psh),
            "Fwd URG Flags": float(self.fwd_urg),
            "Bwd URG Flags": float(self.bwd_urg),
            "Fwd Header Length": float(sum(self.fwd_header_lengths)),
            "Bwd Header Length": float(sum(self.bwd_header_lengths)),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": float(pkt_min),
            "Max Packet Length": float(pkt_max),
            "Packet Length Mean": float(pkt_mean),
            "Packet Length Std": float(pkt_std),
            "Packet Length Variance": float(pkt_var),
            "FIN Flag Count": float(self.fin_count),
            "SYN Flag Count": float(self.syn_count),
            "RST Flag Count": float(self.rst_count),
            "PSH Flag Count": float(self.psh_count),
            "ACK Flag Count": float(self.ack_count),
            "URG Flag Count": float(self.urg_count),
            "CWE Flag Count": float(self.cwe_count),
            "ECE Flag Count": float(self.ece_count),
            "Down/Up Ratio": float(n_bwd / max(n_fwd, 1)),
            "Average Packet Size": float(pkt_mean),
            "Avg Fwd Segment Size": float(fwd_mean),
            "Avg Bwd Segment Size": float(bwd_mean),
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": float(n_fwd),
            "Subflow Fwd Bytes": float(tot_fwd_len),
            "Subflow Bwd Packets": float(n_bwd),
            "Subflow Bwd Bytes": float(tot_bwd_len),
            "Init_Win_bytes_forward": float(self.init_win_fwd),
            "Init_Win_bytes_backward": float(self.init_win_bwd),
            "act_data_pkt_fwd": float(self.act_data_pkt_fwd),
            "min_seg_size_forward": float(self.min_seg_size_forward),
            "Active Mean": 0.0,
            "Active Std": 0.0,
            "Active Max": 0.0,
            "Active Min": 0.0,
            "Idle Mean": 0.0,
            "Idle Std": 0.0,
            "Idle Max": 0.0,
            "Idle Min": 0.0,
        }

        # Return vector ordered by exact expected feature_columns
        return {col: features.get(col, 0.0) for col in feature_columns}


class FlowBuilder:
    """Manages active network flows, groups packets by bidirectional 5-tuple,
    and yields completed flows upon timeout or termination.
    """

    def __init__(
        self,
        feature_columns: List[str],
        flow_timeout_sec: float = 15.0,
    ) -> None:
        self.feature_columns = feature_columns
        self.flow_timeout_sec = flow_timeout_sec
        self.active_flows: Dict[str, BidirectionalFlow] = {}

    def _get_key(self, pkt: PacketInfo) -> str:
        # Canonical bidirectional 5-tuple key
        endpoints = sorted([(pkt.src_ip, pkt.src_port), (pkt.dst_ip, pkt.dst_port)])
        return f"{endpoints[0][0]}:{endpoints[0][1]}<->{endpoints[1][0]}:{endpoints[1][1]}:{pkt.protocol}"

    def process_packet(self, pkt: PacketInfo) -> Optional[Dict[str, float]]:
        """Add a packet. If flow finishes (FIN/RST) or expires, returns its 77 features."""
        key = self._get_key(pkt)
        completed_features = None

        if key in self.active_flows:
            flow = self.active_flows[key]
            # Check timeout
            if pkt.timestamp - flow.last_time > self.flow_timeout_sec:
                completed_features = flow.to_cic_features(self.feature_columns)
                # Restart flow with current packet
                self.active_flows[key] = BidirectionalFlow(pkt)
            else:
                flow.add_packet(pkt)
                # If connection terminates with FIN or RST, flush flow
                if "F" in pkt.flags or "R" in pkt.flags:
                    completed_features = flow.to_cic_features(self.feature_columns)
                    del self.active_flows[key]
        else:
            self.active_flows[key] = BidirectionalFlow(pkt)

        return completed_features

    def flush_expired(self, current_time: float) -> List[Tuple[str, Dict[str, float]]]:
        """Flush and return all flows that have exceeded inactivity timeout."""
        expired = []
        keys_to_delete = []
        for key, flow in self.active_flows.items():
            if current_time - flow.last_time > self.flow_timeout_sec:
                expired.append((flow.flow_id, flow.to_cic_features(self.feature_columns)))
                keys_to_delete.append(key)
        for k in keys_to_delete:
            del self.active_flows[k]
        return expired

