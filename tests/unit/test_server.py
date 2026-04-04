"""Tests for the FastAPI server (nanochat.server.app)."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nanochat.core.device import setup_device, reset_for_testing


@pytest.fixture(scope="module", autouse=True)
def _setup_cpu():
    """Ensure CPU device is configured for the test module."""
    reset_for_testing()
    setup_device("cpu")


@pytest.fixture()
def client():
    """Return a FastAPI TestClient backed by a real nano-scale model."""
    from flax import nnx
    from fastapi.testclient import TestClient

    from nanochat.config import ModelConfig
    from nanochat.model.transformer import TransformerLM
    from nanochat.tokenizer.bpe import BPETokenizer
    from nanochat.inference.engine import InferenceEngine
    from nanochat.server.app import create_app

    cfg = ModelConfig.for_scale("nano")
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
    tokenizer = BPETokenizer.from_pretrained()
    engine = InferenceEngine(model, tokenizer, cfg)
    app = create_app(engine)
    with TestClient(app) as tc:
        yield tc


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device_type" in body
    assert "jax_version" in body


def test_chat_completion_non_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "nanochat-jax",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "content" in body["choices"][0]["message"]
    assert body["usage"]["total_tokens"] > 0


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "nanochat-jax"


def test_invalid_request_returns_422(client):
    # Missing required 'messages' field
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nanochat-jax"},
    )
    assert resp.status_code == 422
