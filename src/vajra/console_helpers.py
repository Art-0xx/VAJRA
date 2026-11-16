# VAJRA Live Console & Demonstration Helpers (Blueprint Sections 5 & 7)
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from vajra.flows.builder import FlowBuilder, PacketInfo
from vajra.flows.extractor import PcapFlowExtractor 
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import yaml
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet, Raw

from vajra.attacks.constrained_pgd import CONTROLLABLE_PATTERNS
from vajra.config import load_config
from vajra.justification.divergence import compute_divergence
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution
from vajra.models.registry import get_model

# Suppress PyTorch nested tensor warnings cleanly
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

# ANSI Color Codes
CLR_R = "\033[91m"  # Malicious / Alert (Red)
CLR_G = "\033[92m"  # Benign (Green)
CLR_Y = "\033[93m"  # Suspicious / Attention Shift (Yellow)
CLR_C = "\033[96m"  # Info / Telemetry (Cyan)
CLR_M = "\033[95m"  # Adversarial / Vector (Magenta)
CLR_B = "\033[1m"   # Bold
CLR_DIM = "\033[2m" # Dim
CLR_X = "\033[0m"   # Reset


def load_default_config() -> Dict[str, Any]:
    """Loads configurations from configs/default.yaml."""
    cfg_path = Path("configs/default.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "version": "2.0.0",
        "thresholds": {"baseline": 0.880, "defended": 0.500},
        "divergence": {"threshold": 0.050},
        "network": {"interface": "lo"}
    }


# console_helpers.py — replace extract_packet_features entirely

def scapy_to_packet_info(pkt) -> Optional[PacketInfo]:
    """Real extraction: no scenario branching, just read the packet."""
    if IP not in pkt:
        return None
    ip_layer = pkt[IP]
    if TCP in pkt:
        t = pkt[TCP]
        return PacketInfo(
            timestamp=float(pkt.time), src_ip=ip_layer.src, dst_ip=ip_layer.dst,
            src_port=t.sport, dst_port=t.dport, protocol="TCP",
            length=len(pkt), header_length=len(t) - len(t.payload),
            flags=str(t.flags), window_size=t.window,
            payload_len=len(t.payload),
        )
    elif UDP in pkt:
        u = pkt[UDP]
        return PacketInfo(
            timestamp=float(pkt.time), src_ip=ip_layer.src, dst_ip=ip_layer.dst,
            src_port=u.sport, dst_port=u.dport, protocol="UDP",
            length=len(pkt), header_length=8, flags="", window_size=0,
            payload_len=len(u.payload),
        )
    return None


def score_completed_flow(flow_feats: Dict[str, float], preprocessor, model, device):
    """The ONLY path from features to a score — no shortcuts, no hardcoding."""
    df = pd.DataFrame([flow_feats])[preprocessor.feature_columns]
    x = torch.tensor(preprocessor.transform(df), dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()
    return df, x, prob

def format_feature_diff_table(
    clean_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    clean_tensor: torch.Tensor,
    adv_tensor: torch.Tensor,
    feature_names: List[str],
    clean_pkt: Optional[Any] = None,
    adv_pkt: Optional[Any] = None,
    perturbation_label: Optional[str] = None,
    top_k: int = 3
) -> str:
    """
    Computes packet pair diffs and top-k changed scaled features,
    returning a formatted terminal table.
    """
    delta_vec = torch.abs(adv_tensor - clean_tensor).cpu().numpy().flatten()
    sorted_idx = np.argsort(delta_vec)[::-1]

    # Packet-level attributes
    c_dport, a_dport = 0, 0
    c_len, a_len = 0, 0
    c_flags, a_flags = "-", "-"

    if clean_pkt is not None:
        c_len = len(clean_pkt)
        if IP in clean_pkt:
            if TCP in clean_pkt:
                c_dport = clean_pkt[TCP].dport
                c_flags = str(clean_pkt[TCP].flags)
            elif UDP in clean_pkt:
                c_dport = clean_pkt[UDP].dport
                c_flags = "UDP"

    if adv_pkt is not None:
        a_len = len(adv_pkt)
        if IP in adv_pkt:
            if TCP in adv_pkt:
                a_dport = adv_pkt[TCP].dport
                a_flags = str(adv_pkt[TCP].flags)
            elif UDP in adv_pkt:
                a_dport = adv_pkt[UDP].dport
                a_flags = "UDP"

    pert_text = perturbation_label or "Perturbation applied"

    lines = []
    lines.append(f"  {CLR_DIM}┌────────────────────────────────────────────────────────────────────┐{CLR_X}")
    lines.append(f"  {CLR_DIM}│{CLR_X} {CLR_B}PACKET PAIR DIFF:{CLR_X} DstPort: {c_dport}->{a_dport} | Len: {c_len}B->{a_len}B | TCP Flags: {c_flags}->{a_flags:<8} {CLR_DIM}│{CLR_X}")
    lines.append(f"  {CLR_DIM}│{CLR_X} {CLR_DIM}► Note: {pert_text:<57}{CLR_X} {CLR_DIM}│{CLR_X}")
    lines.append(f"  {CLR_DIM}├─────────────────────────────┬──────────┬──────────┬──────────┤{CLR_X}")
    lines.append(f"  {CLR_DIM}│{CLR_X} {CLR_B}FEATURE ATTRIBUTE (SCALED){CLR_X}  {CLR_DIM}│{CLR_X} {CLR_B}CLEAN{CLR_X}    {CLR_DIM}│{CLR_X} {CLR_B}ADVERSAL{CLR_X} {CLR_DIM}│{CLR_X} {CLR_B}DELTA{CLR_X}    {CLR_DIM}│{CLR_X}")
    lines.append(f"  {CLR_DIM}├─────────────────────────────┼──────────┼──────────┼──────────┤{CLR_X}")

    shown = 0
    for idx in sorted_idx:
        fname = feature_names[idx]
        c_val = float(clean_tensor[0, idx].item())
        a_val = float(adv_tensor[0, idx].item())
        d_val = float(delta_vec[idx])
        if d_val > 1e-4 or shown < top_k:
            fname_trunc = (fname[:27] + "..") if len(fname) > 27 else fname
            lines.append(f"  {CLR_DIM}│{CLR_X} {fname_trunc:<27} {CLR_DIM}│{CLR_X} {c_val:8.4f} {CLR_DIM}│{CLR_X} {a_val:8.4f} {CLR_DIM}│{CLR_X} {CLR_M}+{d_val:7.4f}{CLR_X} {CLR_DIM}│{CLR_X}")
            shown += 1
            if shown >= top_k:
                break

    lines.append(f"  {CLR_DIM}└─────────────────────────────┴──────────┴──────────┴──────────┘{CLR_X}")
    return "\n".join(lines)


def log_alert_entry(
    flow_tuple: str,
    pkt_len: int,
    verdict: str,
    confidence: float,
    model_name: str,
    divergence: float,
    alerts_file: str = "artifacts/alerts/alerts.jsonl"
) -> None:
    """Appends an alert entry to artifacts/alerts/alerts.jsonl."""
    Path(alerts_file).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "flow_tuple": flow_tuple,
        "length": pkt_len,
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "model": model_name,
        "divergence": round(float(divergence), 4),
    }
    with open(alerts_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_live_run_summary(summary_data: Dict[str, Any]) -> str:
    """Saves live run summary scoreboard to artifacts/results/live_run_<timestamp>.json."""
    res_dir = Path("artifacts/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = res_dir / f"live_run_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    return str(filepath)
