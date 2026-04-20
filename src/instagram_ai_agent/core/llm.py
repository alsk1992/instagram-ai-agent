"""Unified LLM layer — OpenRouter primary, Groq/Gemini/Cerebras fallbacks.

All providers expose OpenAI-compatible endpoints so one client library
suffices. Task-based routing decides the model tier and fallback chain.

Usage:
    from instagram_ai_agent.core.llm import generate, generate_json
    text = await generate("caption", "Write a caption about X")
    data = await generate_json("critic", "...", schema={"score": float, "notes": str})
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from openai._exceptions import APIError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

Task = Literal["caption", "critic", "bulk", "script", "vision", "analyze"]


@dataclass(frozen=True)
class Endpoint:
    provider: str
    base_url: str
    env_key: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.7
    # Some providers don't support JSON mode; we'll fall back to regex parse.
    supports_json_mode: bool = True


# OpenRouter-only routing (verified live against /api/v1/models on 2026-04-20).
# This project uses OpenRouter as the sole provider — no Groq/Gemini/Cerebras
# accounts to juggle. Redundancy comes from listing multiple :free siblings
# so failover stays inside OpenRouter when any single model rate-limits.
#
# Per-model :free RPM caps (Apr 2026) range from ~8 (hot models like
# llama-3.3-70b:free) to ~20 globally across the pool. The auto-router
# `openrouter/free` is primary because it spreads load across all free
# models in real time, dodging single-model RPM caps. Concrete models
# follow as named fallbacks for when the router itself is overloaded.
_OR = "https://openrouter.ai/api/v1"

CAPTION_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "minimax/minimax-m2.5:free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "arcee-ai/trinity-large-preview:free",
             max_tokens=1024, temperature=0.85),
]

CRITIC_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "nvidia/nemotron-3-super-120b-a12b:free",
             max_tokens=512, temperature=0.1),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "nvidia/nemotron-3-nano-30b-a3b:free",
             max_tokens=512, temperature=0.1),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=512, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=512, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "minimax/minimax-m2.5:free",
             max_tokens=512, temperature=0.2),
]

BULK_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=2048, temperature=0.6),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "minimax/minimax-m2.5:free",
             max_tokens=2048, temperature=0.6),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=2048, temperature=0.6),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=2048, temperature=0.6),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "arcee-ai/trinity-large-preview:free",
             max_tokens=2048, temperature=0.6),
]

SCRIPT_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "arcee-ai/trinity-large-preview:free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "minimax/minimax-m2.5:free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=2048, temperature=0.75),
]

ANALYZE_CHAIN: list[Endpoint] = CRITIC_CHAIN  # same reasoning, same models

# Vision: only models with image input are eligible. Verified 2026-04-20:
# both Gemma-4 :free SKUs and the openrouter/free router accept image input;
# the Nemotron / MiniMax / Arcee / LFM :free models are text-only.
VISION_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=1024, temperature=0.3),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=1024, temperature=0.3),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=1024, temperature=0.3),
]


ROUTES: dict[Task, list[Endpoint]] = {
    "caption": CAPTION_CHAIN,
    "critic": CRITIC_CHAIN,
    "bulk": BULK_CHAIN,
    "script": SCRIPT_CHAIN,
    "analyze": ANALYZE_CHAIN,
    "vision": VISION_CHAIN,
}


# Client cache keyed by (base_url, api_key) — avoids rebuilding SSL contexts
_clients: dict[tuple[str, str], AsyncOpenAI] = {}

# Rate-limit awareness. Free-tier quotas (2026-Apr):
#   OpenRouter:   ~20 RPM global over :free pool; specific hot models ~8 RPM
#   Groq:         30 RPM on llama-3.3-70b-versatile (14.4k/day)
#   Gemini Flash: 15 RPM (1500/day)
#   Cerebras:     30 RPM (14.4k/day)
# When an endpoint returns 429, park it for a cooldown window so the
# next task in the chain skips straight past instead of re-hitting it.
# Key: (provider, model). Value: epoch seconds until the endpoint is usable again.
_cooldown: dict[tuple[str, str], float] = {}

# Defaults when the 429 response has no Retry-After header.
_DEFAULT_COOLDOWN_S = 60.0
_MAX_COOLDOWN_S = 600.0


def _ep_key(ep: Endpoint) -> tuple[str, str]:
    return (ep.provider, ep.model)


def _is_cooling_down(ep: Endpoint) -> float:
    """Return seconds remaining in cooldown, or 0.0 if the endpoint is ready."""
    until = _cooldown.get(_ep_key(ep), 0.0)
    remaining = until - time.monotonic()
    return remaining if remaining > 0 else 0.0


def _park(ep: Endpoint, seconds: float) -> None:
    seconds = min(max(seconds, 5.0), _MAX_COOLDOWN_S)
    _cooldown[_ep_key(ep)] = time.monotonic() + seconds
    log.info(
        "llm cooldown: %s/%s parked for %.0fs after rate-limit",
        ep.provider, ep.model, seconds,
    )


def _retry_after_seconds(err: Exception) -> float | None:
    """Pull Retry-After (seconds or HTTP-date) from an httpx response, if present."""
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
    if not ra:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def _client_for(ep: Endpoint) -> AsyncOpenAI | None:
    key = os.environ.get(ep.env_key)
    if not key:
        return None
    cache_key = (ep.base_url, key)
    client = _clients.get(cache_key)
    if client is None:
        client = AsyncOpenAI(
            base_url=ep.base_url,
            api_key=key,
            timeout=httpx.Timeout(60.0, connect=10.0),
            max_retries=0,  # we handle retries via tenacity on the outer loop
        )
        _clients[cache_key] = client
    return client


class AllProvidersFailed(RuntimeError):
    """Raised when every provider in a task's chain has failed. The original
    failure is chained via ``__cause__``."""


# Retry only on transient network/5xx errors. 429 is NOT retried at this
# layer — per-model RPM caps won't reset in a 2-20s window, so burning
# attempts here just slows the outer fallback chain. generate() catches
# RateLimitError and moves to the next endpoint immediately.
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
async def _call_one(
    ep: Endpoint,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    client = _client_for(ep)
    if client is None:
        raise RuntimeError(f"No API key for {ep.provider}")

    kwargs: dict[str, Any] = {
        "model": ep.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode and ep.supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    r = await client.chat.completions.create(**kwargs)
    text = (r.choices[0].message.content or "").strip()
    if not text:
        raise APIError("Empty completion", request=None, body=None)
    return text


async def generate(
    task: Task,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Generate text for a task. Tries each endpoint in the chain until one succeeds."""
    chain = ROUTES[task]
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for ep in chain:
        if _client_for(ep) is None:
            continue
        cooling = _is_cooling_down(ep)
        if cooling > 0:
            log.debug(
                "llm %s skip %s/%s — cooldown %.0fs left",
                task, ep.provider, ep.model, cooling,
            )
            continue
        try:
            out = await _call_one(
                ep,
                messages,
                max_tokens=max_tokens or ep.max_tokens,
                temperature=temperature if temperature is not None else ep.temperature,
                json_mode=json_mode,
            )
            log.debug("llm %s via %s/%s ok (%d chars)", task, ep.provider, ep.model, len(out))
            return out
        except RateLimitError as e:
            # 429 — park this endpoint and move to the next in the chain.
            _park(ep, _retry_after_seconds(e) or _DEFAULT_COOLDOWN_S)
            last_err = e
            continue
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            # 404/400 — dead model or bad request; don't retry, skip to next.
            # 5xx — transient server issue; brief jitter then next.
            if code == 429:
                _park(ep, _retry_after_seconds(e) or _DEFAULT_COOLDOWN_S)
            elif code and code >= 500:
                await asyncio.sleep(0.5 + random.random())
            log.warning("llm %s via %s/%s failed (%s): %s",
                        task, ep.provider, ep.model, code, e)
            last_err = e
            continue
        except Exception as e:
            log.warning("llm %s via %s/%s failed: %s", task, ep.provider, ep.model, e)
            last_err = e
            await asyncio.sleep(0.5)
            continue

    raise AllProvidersFailed(
        f"All providers failed for task={task}: {last_err!r}"
    ) from last_err


_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
_FIRST_BRACE = re.compile(r"[{\[]")


def _strip_json(raw: str) -> str:
    """Extract the JSON payload from a possibly-markdown-wrapped LLM response."""
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if m:
        return m.group(1).strip()
    # Skip over leading prose if any
    m2 = _FIRST_BRACE.search(s)
    if m2:
        return s[m2.start():]
    return s


async def generate_json(
    task: Task,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    """Generate and parse JSON. Tolerates providers that don't support strict JSON mode."""
    enriched = (
        (system or "")
        + "\nRespond with valid JSON only. No prose, no markdown fences."
    ).strip()
    raw = await generate(
        task,
        prompt,
        system=enriched,
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else 0.2,
        json_mode=True,
    )
    cleaned = _strip_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # One-shot repair pass: try shrinking to balanced braces
        repaired = _extract_balanced(cleaned)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # Repair attempt didn't help — fall through to the raise below
                # which surfaces the ORIGINAL raw payload (more debuggable).
                pass
        raise ValueError(f"LLM returned unparseable JSON: {raw[:400]!r}") from e


def _extract_balanced(s: str) -> str | None:
    """Greedy balanced-braces extract — handles trailing garbage from CoT models."""
    if not s:
        return None
    open_c = s[0]
    close_c = {"{": "}", "[": "]"}.get(open_c)
    if close_c is None:
        return None
    depth = 0
    for i, ch in enumerate(s):
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return None


# ───── Vision helper ─────
async def describe_image(image_url: str, question: str = "Describe this image.") -> str:
    chain = ROUTES["vision"]
    last_err: Exception | None = None
    for ep in chain:
        if _client_for(ep) is None:
            continue
        try:
            client = _client_for(ep)
            r = await client.chat.completions.create(
                model=ep.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=ep.max_tokens,
                temperature=ep.temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            continue
    raise AllProvidersFailed(f"Vision failed: {last_err!r}") from last_err


def providers_configured() -> list[str]:
    # OpenRouter-only project. Returned as a list for backward compat with
    # callers that expect an iterable.
    return ["openrouter"] if os.environ.get("OPENROUTER_API_KEY") else []
