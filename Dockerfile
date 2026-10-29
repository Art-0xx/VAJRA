FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY src/ ./src/
COPY configs/ ./configs/
RUN mkdir -p artifacts/checkpoints artifacts/metrics artifacts/figures

CMD ["uvicorn", "vajra.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
