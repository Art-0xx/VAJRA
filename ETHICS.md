# Ethics and Safety Statement

VAJRA includes a live-replay component that reconstructs adversarial network
flows into PCAP files and injects them into an emulated network. This is
defensive security research.

## Isolation Guarantees

1. All replay experiments run inside a Mininet virtual topology with no route
   to the host network, the university network, or the public internet.
2. No traffic is injected onto production interfaces.
3. Reconstructed PCAPs contain only traffic derived from public benchmarks.
4. The vajra-replay Docker profile is opt-in.
