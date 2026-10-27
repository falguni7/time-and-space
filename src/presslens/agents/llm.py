"""One place that builds the chat client every agent uses.

Backends (environment variable PRESSLENS_LLM, default: azure when a key is set, otherwise offline):

    azure    Azure AI Foundry deployment (AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, AZURE_AI_DEPLOYMENT in .env)
    claude   Claude through Anthropic's OpenAI-compatible endpoint (ANTHROPIC_API_KEY; not tested yet)
    offline  no network: replays cached answers of PRESSLENS_OFFLINE_MODEL, misses come back empty

Every backend runs behind the same disk cache (data/llm_cache.sqlite), so a repeated request never
costs twice and a recorded demo replays exactly. The cache sits in chat middleware, i.e. inside the
tool-calling loop, so each model round trip of an agent is cached on its own.

    from presslens.agents.llm import chat_client, is_offline_miss
    client = chat_client()
    agent = Agent(client=client, instructions="...", name="Explainer")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatContext,
    ChatMiddleware,
    ChatMiddlewareLayer,
    ChatResponse,
    FunctionInvocationLayer,
    Message,
)

from ..paths import DATA_DIR, ROOT

ENV_FILE = ROOT / ".env"
CACHE_PATH = DATA_DIR / "llm_cache.sqlite"
OFFLINE_MISS = "presslens_offline_miss"
DEFAULT_OFFLINE_MODEL = "gpt-4.1-mini"
CLAUDE_BASE_URL = "https://api.anthropic.com/v1/"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
MAX_RETRIES = 6        # openai client retries 429/5xx with exponential backoff
TIMEOUT_S = 60.0


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_FILE, override=False)


def _azure_base_url(endpoint: str) -> str:
    """Foundry shows the full Responses/Chat URL; the client wants the .../openai/v1/ base."""
    base = re.sub(r"/(responses|chat/completions)/?$", "", endpoint.strip()).rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    return base + "/"


def backend_name() -> str:
    _load_env()
    name = os.environ.get("PRESSLENS_LLM", "").strip().lower()
    if name:
        return name
    return "azure" if os.environ.get("AZURE_AI_API_KEY") else "offline"


# --------------------------------------------------------------------------- cache


def _jsonable(obj: Any) -> Any:
    """Stable JSON view of chat options: classes become their JSON schema, tools their declaration."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, type) and hasattr(obj, "model_json_schema"):
        return {"schema": obj.__name__, "json": obj.model_json_schema()}
    if hasattr(obj, "to_json_schema_spec"):
        return obj.to_json_schema_spec()
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return repr(obj)


def _strip_ids(d: Any) -> Any:
    """Drop per-run identifiers so the same conversation hashes the same way twice."""
    if isinstance(d, dict):
        return {k: _strip_ids(v) for k, v in d.items()
                if k not in {"id", "message_id", "response_id", "created_at", "additional_properties",
                             "raw_representation"}}  # "id": framework's random af-call-... per tool call
    if isinstance(d, list):
        return [_strip_ids(v) for v in d]
    return d


def request_key(model: str, messages: list[Message], options: dict[str, Any] | None) -> str:
    opts = {k: v for k, v in (options or {}).items() if v is not None and k not in {"model", "store"}}
    payload = {
        "model": model,
        "messages": _strip_ids([m.to_dict() for m in messages]),
        "options": _jsonable(opts),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode()).hexdigest()


class ResponseCache:
    """sqlite file of model responses keyed by request hash (one file: /hnvme limits file counts)."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.execute(
            "create table if not exists responses (key text primary key, model text, response text,"
            " prompt_tokens integer, completion_tokens integer, seconds real, created real)")
        self._con.commit()

    def get(self, key: str) -> ChatResponse | None:
        with self._lock:
            row = self._con.execute("select response from responses where key = ?", (key,)).fetchone()
        return ChatResponse.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, model: str, response: ChatResponse, seconds: float) -> None:
        usage = response.usage_details or {}
        with self._lock:
            self._con.execute(
                "insert or replace into responses values (?, ?, ?, ?, ?, ?, ?)",
                (key, response.model or model, json.dumps(response.to_dict()),
                 usage.get("input_token_count"), usage.get("output_token_count"), seconds, time.time()))
            self._con.commit()

    def stats(self) -> list[tuple]:
        with self._lock:
            return self._con.execute(
                "select model, count(*), sum(prompt_tokens), sum(completion_tokens) "
                "from responses group by model order by model").fetchall()


@dataclass
class CallStats:
    hits: int = 0
    misses: int = 0
    offline_misses: int = 0


class CacheMiddleware(ChatMiddleware):
    """Serve repeated requests from disk; store new ones. Streaming calls pass straight through."""

    def __init__(self, cache: ResponseCache, model: str):
        self.cache = cache
        self.model = model
        self.stats = CallStats()

    async def process(self, context: ChatContext, call_next) -> None:
        if context.stream:
            await call_next()
            return
        key = request_key(self.model, list(context.messages), dict(context.options or {}))
        cached = self.cache.get(key)
        if cached is not None:
            cached.additional_properties = {**(cached.additional_properties or {}), "cache_hit": True}
            context.result = cached
            self.stats.hits += 1
            return
        t0 = time.time()
        await call_next()
        result = context.result
        if isinstance(result, ChatResponse):
            if is_offline_miss(result):
                self.stats.offline_misses += 1
            else:
                self.stats.misses += 1
                self.cache.put(key, self.model, result, time.time() - t0)


# --------------------------------------------------------------------------- offline backend


class OfflineChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """No network. Behind the cache middleware it replays recorded answers; a miss returns an
    empty assistant message flagged OFFLINE_MISS so callers can fall back to templates."""

    OTEL_PROVIDER_NAME = "presslens-offline"

    def __init__(self, model: str = DEFAULT_OFFLINE_MODEL, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        if stream:
            raise NotImplementedError("offline backend does not stream")
        return ChatResponse(
            messages=[Message(role="assistant", contents=[""])],
            model=self.model,
            additional_properties={OFFLINE_MISS: True},
        )


def is_offline_miss(response: Any) -> bool:
    """True when the offline backend had no recorded answer (agents then use their template)."""
    props = getattr(response, "additional_properties", None) or {}
    if props.get(OFFLINE_MISS):
        return True
    raw = getattr(response, "raw_representation", None)
    return isinstance(raw, ChatResponse) and is_offline_miss(raw)


# --------------------------------------------------------------------------- factory


@dataclass
class LLM:
    """The chat client plus what the UI and evaluation need to label its output."""
    client: Any
    backend: str
    model: str
    cache: CacheMiddleware

    @property
    def label(self) -> str:
        return f"{self.model} ({self.backend})"


_shared: dict[str, LLM] = {}


def chat_client(backend: str | None = None, *, cache_path: Path = CACHE_PATH) -> LLM:
    """Build (once per backend and cache file) the client the agents share."""
    backend = (backend or backend_name()).lower()
    memo = f"{backend}|{cache_path}"
    if memo in _shared:
        return _shared[memo]
    _load_env()
    cache = ResponseCache(cache_path)

    if backend == "offline":
        model = os.environ.get("PRESSLENS_OFFLINE_MODEL", DEFAULT_OFFLINE_MODEL)
        mw = CacheMiddleware(cache, model)
        client = OfflineChatClient(model=model, middleware=[mw])
    elif backend in ("azure", "claude"):
        from agent_framework.openai import OpenAIChatCompletionClient
        from openai import AsyncOpenAI
        if backend == "azure":
            model = os.environ.get("AZURE_AI_DEPLOYMENT", DEFAULT_OFFLINE_MODEL)
            base_url = _azure_base_url(_require("AZURE_AI_ENDPOINT"))
            api_key = _require("AZURE_AI_API_KEY")
        else:
            model = os.environ.get("PRESSLENS_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            base_url = CLAUDE_BASE_URL
            api_key = _require("ANTHROPIC_API_KEY")
        mw = CacheMiddleware(cache, model)
        client = OpenAIChatCompletionClient(
            model=model,
            async_client=AsyncOpenAI(base_url=base_url, api_key=api_key,
                                     max_retries=MAX_RETRIES, timeout=TIMEOUT_S),
            middleware=[mw],
        )
    else:
        raise ValueError(f"unknown PRESSLENS_LLM backend {backend!r} (azure | claude | offline)")

    llm = LLM(client=client, backend=backend, model=model, cache=mw)
    _shared[memo] = llm
    return llm


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to {ENV_FILE} or use PRESSLENS_LLM=offline")
    return value
