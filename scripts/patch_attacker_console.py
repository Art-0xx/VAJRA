#!/usr/bin/env python
"""Patches attacker_console.py in place (backup: attacker_console.py.bak).

Replaces the 3 synthetic SYN-flood waves with structural replays of REAL
CIC-IDS2017 rows (PortScan, DDoS, DoS Hulk), fixes the 2.0s inter-packet
delay that caused the 10s latency, and makes the scoreboard's "evaded"
count use only attacks the clean flow actually got detected.

Run from /mnt/d/VAJRA:  python scripts/patch_attacker_console.py
"""
import shutil
import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "attacker_console.py")

NEW_CODE = r'''
# ---------------------------------------------------------------------
# Real-row replay. Reconstructs the STRUCTURE (packet counts, byte totals,
# windows, flag pattern, timing scale) of one real CIC-IDS2017 row.
# CAVEAT: built from aggregate CSV stats, not a byte-level PCAP replay, and
# duration is capped at 1s so the demo stays fast. The row is chosen by
# median packet count, NOT by whether the model detects it (no cherry-picking).
# ---------------------------------------------------------------------
DATA_DIR = Path("data/raw/MachineLearningCVE")
ROW_SEED = 7
_NEEDED = {
    "Label", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "SYN Flag Count", "FIN Flag Count", "RST Flag Count", "PSH Flag Count",
}
ATTACK_SPECS = [
    {"name": "PortScan", "file": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "label": "PortScan", "port": 80},
    {"name": "DDoS", "file": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "label": "DDoS", "port": 80},
    {"name": "DoS Hulk", "file": "Wednesday-workingHours.pcap_ISCX.csv", "label": "DoS Hulk", "port": 80},
]


def load_attack_rows() -> None:
    """Fills spec['row'] for each attack spec (one real, representative row)."""
    for spec in ATTACK_SPECS:
        if "row" in spec:
            continue
        fpath = DATA_DIR / spec["file"]
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath} (run from the project root).")
        df = pd.read_csv(fpath, usecols=lambda c: c.strip() in _NEEDED, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[df["Label"].astype(str).str.strip() == spec["label"]]
        num = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
        df = df[np.isfinite(num.to_numpy()).all(axis=1)]
        if len(df) == 0:
            raise RuntimeError(f"No usable '{spec['label']}' rows in {fpath}")
        df = df.sample(n=min(200, len(df)), random_state=ROW_SEED)
        pk = df["Total Fwd Packets"] + df["Total Backward Packets"]
        spec["row"] = df.loc[pk.sort_values().index[len(df) // 2]]


def build_flow_from_csv_row(spec, src_ip, dst_ip, sport, dport) -> List[Tuple[float, Packet]]:
    r = spec["row"]
    n_f = int(min(max(r["Total Fwd Packets"], 1), 30))
    n_b = int(min(max(r["Total Backward Packets"], 0), 30))
    fwd_len = max(0, int(r["Total Length of Fwd Packets"]))
    bwd_len = max(0, int(r["Total Length of Bwd Packets"]))
    dur = min(max(float(r["Flow Duration"]), 0.0) / 1e6, 1.0)
    w_f = int(min(max(r["Init_Win_bytes_forward"], 0), 65535))
    w_b = int(min(max(r["Init_Win_bytes_backward"], 0), 65535))
    syn, fin = r["SYN Flag Count"] > 0, r["FIN Flag Count"] > 0
    rst, psh = r["RST Flag Count"] > 0, r["PSH Flag Count"] > 0

    packets: List[Tuple[float, Packet]] = []
    f_times = np.linspace(0.0, dur, n_f) if n_f > 1 else [0.0]
    seq = 1000
    for i, t in enumerate(f_times):
        flags = "S" if (i == 0 and syn) else ("PA" if (i == 0 and psh) else "A")
        if fin and i == n_f - 1 and n_b == 0:
            flags = "FA"
        size = fwd_len // n_f + (fwd_len % n_f if i == 0 else 0)
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags, window=w_f, seq=seq)
        if size > 0:
            pkt = pkt / Raw(load=b"X" * min(size, 60000))
        packets.append((float(t), pkt))
        seq += max(size, 1)

    if n_b > 0:
        b_times = np.linspace(dur * 0.05, dur, n_b) if n_b > 1 else [dur * 0.05]
        seq = 500
        for i, t in enumerate(b_times):
            flags = "RA" if (rst and i == n_b - 1) else ("FA" if (fin and i == n_b - 1) else "A")
            size = bwd_len // n_b + (bwd_len % n_b if i == 0 else 0)
            pkt = IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags=flags, window=w_b, seq=seq)
            if size > 0:
                pkt = pkt / Raw(load=b"Y" * min(size, 60000))
            packets.append((float(t), pkt))
            seq += max(size, 1)

    packets.sort(key=lambda x: x[0])
    return packets


'''

EDITS = [
    ('port_names = {80: "HTTP Web Service", 22: "SSH Management", 53: "DNS Name Resolution"}',
     'port_names = {}'),
    ('target_ports = [80, 22, 53]',
     'load_attack_rows()\n    target_ports = [s["port"] for s in ATTACK_SPECS]'),
    ('pname = port_names.get(port, f"Port {port}")',
     'spec = ATTACK_SPECS[wave_idx - 1]; pname = spec["name"]'),
    ('Target Service -> {pname}', 'Real CIC-IDS2017 flow -> {pname}'),
    ('build_syn_flood_flow(src_ip, dst_ip, sport, port, n_packets=6, inter_packet_delay=2.0)',
     'build_flow_from_csv_row(spec, src_ip, dst_ip, sport, port)'),
    ('build_syn_flood_flow(src_ip, dst_ip, adv_sport, port, n_packets=6, inter_packet_delay=2.0)',
     'build_flow_from_csv_row(spec, src_ip, dst_ip, adv_sport, port)'),
    ('Sent clean SYN-flood flow ({len(clean_packets)} packets, port {port})',
     "Sent clean {spec['name']} flow ({len(clean_packets)} packets; CSV row: fwd={int(spec['row']['Total Fwd Packets'])}, bwd={int(spec['row']['Total Backward Packets'])}, dur={spec['row']['Flow Duration']:.0f}us)"),
    ('{adv_evaded}/{total_attacks}{CLR_X}', '{adv_evaded}/{clean_detected} (of detected){CLR_X}'),
    ('"adv_total": total_attacks', '"adv_total": clean_detected'),
]
ANCHOR = "def build_real_hulk_reconstruction("


def main():
    src = TARGET.read_text(encoding="utf-8")
    if "build_flow_from_csv_row" in src:
        sys.exit("[!] Already patched.")
    for old, _ in EDITS:
        n = src.count(old)
        if n != 1:
            sys.exit(f"[!] Expected exactly 1 match, found {n} for:\n    {old}\nNo changes written.")
    if src.count(ANCHOR) != 1:
        sys.exit("[!] Anchor function not found. No changes written.")
    for old, new in EDITS:
        src = src.replace(old, new)
    src = src.replace(ANCHOR, NEW_CODE.lstrip("\n") + ANCHOR)
    shutil.copy(TARGET, str(TARGET) + ".bak")
    TARGET.write_text(src, encoding="utf-8")
    print(f"[+] Patched {TARGET} (backup: {TARGET}.bak)")


if __name__ == "__main__":
    main()