# PCAP & Live Network Flow Feature Extractor (Blueprint Section 5.2)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, rdpcap, sniff

from vajra.flows.builder import FlowBuilder, PacketInfo

logger = logging.getLogger("vajra.flows")


class PcapFlowExtractor:
    """Extracts bidirectional CIC-IDS2017 77-feature flows from PCAP captures."""

    def __init__(self, feature_columns: List[str], scaler: Any = None) -> None:
        self.feature_columns = feature_columns
        self.scaler = scaler

    def _scapy_to_packet_info(self, pkt: Any) -> Optional[PacketInfo]:
        if IP not in pkt:
            return None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "OTHER")

        if TCP in pkt:
            tcp_layer = pkt[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            flags = str(tcp_layer.flags)
            win_size = tcp_layer.window
            hdr_len = len(tcp_layer)
            payload_len = len(tcp_layer.payload)
        elif UDP in pkt:
            udp_layer = pkt[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            flags = ""
            win_size = 0
            hdr_len = 8
            payload_len = len(udp_layer.payload)
        else:
            return None

        return PacketInfo(
            timestamp=float(pkt.time),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto,
            length=len(pkt),
            header_length=hdr_len,
            flags=flags,
            window_size=win_size,
            payload_len=payload_len,
        )

    def extract_from_pcap(
        self,
        pcap_path: str,
        timeout_sec: float = 15.0,
    ) -> pd.DataFrame:
        """Process a pcap file into a DataFrame matching the 77 CIC columns."""
        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        logger.info(f"[*] Reading packets from {pcap_path} ...")
        packets = rdpcap(str(path))
        logger.info(f"[+] Loaded {len(packets)} raw packets. Building bidirectional flows ...")

        builder = FlowBuilder(feature_columns=self.feature_columns, flow_timeout_sec=timeout_sec)
        completed_flows: List[Dict[str, float]] = []

        last_time = 0.0
        for pkt in packets:
            p_info = self._scapy_to_packet_info(pkt)
            if p_info is None:
                continue
            last_time = p_info.timestamp
            f = builder.process_packet(p_info)
            if f is not None:
                completed_flows.append(f)

        # Flush any remaining active flows at the end of capture
        flushed = builder.flush_expired(current_time=last_time + timeout_sec + 1.0)
        for _, f in flushed:
            completed_flows.append(f)

        if not completed_flows:
            logger.warning("[!] No bidirectional flows assembled from PCAP.")
            return pd.DataFrame(columns=self.feature_columns)

        df = pd.DataFrame(completed_flows)[self.feature_columns]
        logger.info(f"[+] Successfully extracted {len(df)} bidirectional flows.")
        return df

    def transform_to_model_input(self, df_flows: pd.DataFrame) -> np.ndarray:
        """Apply pre-fitted MinMax scaler to extracted flows."""
        if self.scaler is None:
            raise ValueError("Scaler is not provided to PcapFlowExtractor.")
        return self.scaler.transform(df_flows).astype(np.float32)

