# VAJRA Architecture

## Services

| Service | Port | Responsibility |
|---|---|---|
| vajra-api | 8000 | Inference, attack endpoint, attention endpoint |
| vajra-dashboard | 8501 | Streamlit UI |
| vajra-replay | internal | PCAP + Mininet replay (opt-in) |

## Invariants

1. API never imports training code.
2. Preprocessing uses the same preprocessor.joblib as training.
3. Feature order is enforced.
4. No service has external network access.
