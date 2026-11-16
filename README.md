# VAJRA // Adversarially Robust Transformer Network Intrusion Detection System

Vector-Attention Justification for Robust Anomaly Detection (**VAJRA**) with full-stack Security Operations Center (SOC) dashboard.

---

## Architecture Overview

VAJRA combines deep self-attention Transformer encoders with problem-space adversarial defense mechanisms for high-throughput network flow inspection.

```
                      +---------------------------------------+
                      |       Next.js 14 SOC Dashboard        |
                      |  (3D Topology, Red/Blue Team, Metrics)|
                      +-------------------+-------------------+
                                          |
                        REST Endpoints    |   WebSocket (ws://localhost:8000/ws)
                                          v
                      +---------------------------------------+
                      |       FastAPI Defense Engine          |
                      |  - Hot-Swap Dispatcher (Zero Downtime)|
                      |  - L-inf Constrained PGD Evaluator    |
                      |  - Attention Rollout & JSD Monitor    |
                      +-------------------+-------------------+
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
       +-------------------------+                 +-------------------------+
       |   Baseline Transformer  |                 | Defended Transformer    |
       |   (Undefended, tau=0.880)               | (PGD-AT, tau=0.612)     |
       +-------------------------+                 +-------------------------+
```

---

## Full-Stack SOC Operations Dashboard

The dashboard delivers a dark-themed tactical SOC interface featuring:

1. **Live Capture & 3D Topographic Surface Topology**:
   - **Zone 1: 3D Data Surface Topology**: Interactive WebGL/Canvas wireframe mountain mesh and point cloud displaying real-time 77-dimensional network telemetry with mouse drag-to-rotate.
   - **Zone 2: Multi-Layered Time-Series Matrix**: High-speed network fluctuation oscillogram with active decision threshold ($\tau$) and multi-track signal decomposition (Clean, Perturbation Delta, PGD-AT Consensus).
   - **Real-Time Feed**: Live WebSocket alert streaming backed by SQLite storage (`artifacts/alerts/alerts.db`).
   - **One-Click 3-Phase Demo**: Single button executing Benign $\rightarrow$ Baseline Attack (Evaded) $\rightarrow$ Defended Attack (Held).

2. **Red Team // Attack Studio**:
   - Target Service selection (HTTP 80, HTTPS 443, DNS 53, SSH 22).
   - Attack vectors: DDoS-LOIC-HTTP, PortScan, DoS-Slowloris, Botnet-ARES.
   - Constrained Gradient PGD vs Heuristic Problem-Space perturbations with adjustable $\epsilon \in [0.005, 0.100]$.
   - Side-by-side verdict cards and top 5 mutable feature perturbation delta table.

3. **Blue Team // Defense & Attention Rollout**:
   - Zero-downtime hot-swap between `baseline_model.pt` and `defended_model.pt`.
   - Layer 4 Transformer Attention Rollout bar chart comparing clean vs adversarial feature weights.
   - Jensen-Shannon Divergence ($D_{JS}$) monitoring with automated rerouting when $D_{JS} > \tau_{JSD} = 0.150$.

4. **Metrics & Empirical SOC Benchmarks**:
   - Top KPI cards: Clean F1 (95.9%), Defended Robust F1 (67.9%), Baseline Robust F1 (1.1%), False Positive Rate (0.08%), Latency (p50: 1.4ms / p99: 3.2ms), Throughput (7,850 pkts/sec).
   - Attack Success Rate (ASR) vs Perturbation Budget ($\epsilon$) curves.
   - Threat Family $\times$ Model Resilience matrix table.

---

## Running the Full-Stack Application

### Terminal 1: FastAPI Inference & Defense Engine
```bash
# Activate Python environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Start FastAPI server on port 8000
python -m uvicorn vajra.service.main:app --host 127.0.0.1 --port 8000
```
- API Health Check: `http://localhost:8000/health`
- WebSocket Telemetry: `ws://localhost:8000/ws`

### Terminal 2: Next.js SOC Dashboard
```bash
cd frontend

# Run production build
npm run start

# Or run interactive development mode
npm run dev
```
Open **`http://localhost:3000`** in any modern browser.

---

## Interactive Live Demo via CLI

If you prefer testing directly in the terminal:

### Terminal 1: Tactical NIDS Console
```bash
# Launch with unhardened baseline model (tau = 0.880)
sudo $(which python) nids_console.py --model baseline

# Or launch directly with defended model (tau = 0.612)
sudo $(which python) nids_console.py --model defended
```
* **Interactive Hot-Swapping**: Type `m` + `Enter` at any time to toggle between models without restarting.
* **Attention Divergence Gate**: Automatically reroutes to defended model when $D_{JS} > 0.150$.

### Terminal 2: Red Team Attacker Console
```bash
# Real constrained gradient attack (PGD optimization on controllable features)
sudo $(which python) attacker_console.py --mode gradient --eps 0.050

# Full head-to-head comparative campaign (evaluates baseline vs defended side-by-side)
sudo $(which python) attacker_console.py --compare
```

---

## Verification & Benchmarks

Validated against empirical test results in `artifacts/metrics/table_B.csv`:
- Baseline model collapses from **95.9% F1** down to **1.1% F1** under $\epsilon=0.05$ PGD perturbation.
- PGD-AT defended model maintains **67.9% F1** under identical adversarial perturbations (+66.8% absolute gain).
