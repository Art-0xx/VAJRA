# VAJRA

Vector-Attention Justification for Robust Anomaly Detection.

## Quick Start

    docker compose up --build

- API: http://localhost:8000/health
- Dashboard: http://localhost:8501

## Local Development

    pip install -e ".[dev]"
    pytest -q
    uvicorn vajra.api.main:app --reload
    streamlit run src/vajra/dashboard/app.py
