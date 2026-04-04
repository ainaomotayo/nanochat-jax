"""FastAPI application providing an OpenAI-compatible API for nanochat-jax.

Usage::

    from nanochat.server.app import create_app
    app = create_app(engine)

Endpoints:
    POST /v1/chat/completions  — chat completion (streaming + non-streaming)
    POST /v1/completions       — raw text completion
    GET  /v1/models            — list available models
    GET  /health               — health check with device info
"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator, Generator, Literal

import structlog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanochat.core.device import device_info
from nanochat.inference.engine import InferenceEngine

logger = structlog.get_logger()

MODEL_NAME = "nanochat-jax"


# ---------------------------------------------------------------------------
# Pydantic request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = MODEL_NAME
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = MODEL_NAME
    choices: list[CompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = MODEL_NAME
    choices: list[StreamChoice]


class ModelInfo(BaseModel):
    id: str = MODEL_NAME
    object: str = "model"
    created: int = 0
    owned_by: str = "nanochat"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _render_chat_prompt(engine: InferenceEngine, messages: list[ChatMessage]) -> str:
    """Convert a list of ChatMessage objects into a prompt string."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    return engine.tokenizer.apply_chat_template(msg_dicts)


def _count_tokens(engine: InferenceEngine, text: str) -> int:
    return len(engine.tokenizer.encode(text))


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

def _stream_chat_sse(
    engine: InferenceEngine,
    prompt: str,
    req: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> Generator[str, None, None]:
    """Yield SSE-formatted chunks from the streaming generator."""
    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"), finish_reason=None)],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Content chunks
    gen = engine.generate(
        prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        stream=True,
    )
    assert not isinstance(gen, (str, list))  # must be a generator

    for fragment in gen:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            choices=[StreamChoice(delta=DeltaMessage(content=fragment), finish_reason=None)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk
    done_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {done_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(engine: InferenceEngine) -> FastAPI:
    """Create and return a configured FastAPI application.

    Args:
        engine: A ready-to-use :class:`InferenceEngine`.

    Returns:
        A :class:`FastAPI` instance with all routes registered.
    """
    app = FastAPI(title="NanoChat-JAX API", version="0.1.0")

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        info = device_info()
        return {"status": "ok", **info}

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    def list_models() -> ModelListResponse:
        return ModelListResponse(data=[ModelInfo()])

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        request_id = _request_id()
        created = int(time.time())
        prompt = _render_chat_prompt(engine, req.messages)

        logger.info(
            "chat_completion_request",
            request_id=request_id,
            n_messages=len(req.messages),
            max_tokens=req.max_tokens,
            stream=req.stream,
        )

        if req.stream:
            return StreamingResponse(
                _stream_chat_sse(engine, prompt, req, request_id, created),
                media_type="text/event-stream",
            )

        # Non-streaming
        prompt_tokens = _count_tokens(engine, prompt)
        result = engine.generate(
            prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        assert isinstance(result, str)

        completion_tokens = _count_tokens(engine, result)

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            choices=[
                ChatCompletionChoice(
                    message=ChoiceMessage(content=result),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # POST /v1/completions
    # ------------------------------------------------------------------
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        request_id = _request_id()
        created = int(time.time())

        logger.info(
            "completion_request",
            request_id=request_id,
            prompt_len=len(req.prompt),
            max_tokens=req.max_tokens,
            stream=req.stream,
        )

        prompt_tokens = _count_tokens(engine, req.prompt)

        if req.stream:
            def _sse_gen():
                gen = engine.generate(
                    req.prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    stream=True,
                )
                for fragment in gen:
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{"index": 0, "text": fragment, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                done = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse_gen(), media_type="text/event-stream")

        result = engine.generate(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        assert isinstance(result, str)

        completion_tokens = _count_tokens(engine, result)

        return CompletionResponse(
            id=request_id,
            created=created,
            choices=[CompletionChoice(text=result)],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app
