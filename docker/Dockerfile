# Multi-stage build for nanochat-jax
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV JAX_PLATFORMS=cpu

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--model-size", "nano", "--total-steps", "100"]
