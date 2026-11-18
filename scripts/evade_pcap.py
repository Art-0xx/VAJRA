import joblib
import torch
import pandas as pd
from scapy.all import rdpcap, wrpcap, Raw
from vajra.flows.extractor import PcapFlowExtractor
# ... (load model) ...

def apply_evasion(pcap_path, out_path):
    packets = rdpcap(pcap_path)
    
    # Mutation: Add padding to payload, change port, add delay
    for pkt in packets:
        if TCP in pkt or UDP in pkt:
            # Pad payload (changes Total Length of Fwd Packets)
            pkt = pkt / Raw(load=b"X" * 400) 
            pkt = pkt.__class__(bytes(pkt))
            
    wrpcap(out_path, packets)

# 1. Score Clean PCAP
clean_flows = extractor.extract_from_pcap("live_sample.pcap")
# ... (score and print) ...

# 2. Mutate and Score
apply_evasion("live_sample.pcap", "evasive.pcap")
adv_flows = extractor.extract_from_pcap("evasive.pcap")
# ... (score and print) ...

# If adv_prob < tau and clean_prob > tau, you have a TRUE EVASION.