"""Unified LLM layer — OpenRouter primary, Groq/Gemini/Cerebras fallbacks.

All providers expose OpenAI-compatible endpoints so one client library
suffices. Task-based routing decides the model tier and fallback chain.

Usage:
    from src.core.llm import generate, generate_json
    text = await generate("caption", "Write a caption about X")
    data = await generate_json("critic", "...", schema={"score": float, "notes": str})
"""
from __future__ import annotations

import asyncio
import json
import os
import re
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

from src.core.logging_setup import get_logger

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


# Free-tier models verified Apr 2026. The ":free" tag on OpenRouter means
# the free daily quota; remove it (and supply credits) to use paid.
CAPTION_CHAIN: list[Endpoint] = [
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        max_tokens=1024,
        temperature=0.85,
    ),
    Endpoint(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
        max_tokens=1024,
        temperature=0.85,
    ),
    Endpoint(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
        max_tokens=1024,
        temperature=0.85,
    ),
]

CRITIC_CHAIN: list[Endpoint] = [
    Endpoint(
        "cerebras",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
        "llama-3.3-70b",
        max_tokens=512,
        temperature=0.2,
    ),
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "deepseek/deepseek-r1:free",
        max_tokens=512,
        temperature=0.1,
    ),
    Endpoint(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
        max_tokens=512,
        temperature=0.2,
    ),
    Endpoint(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
        max_tokens=512,
        temperature=0.2,
    ),
]

BULK_CHAIN: list[Endpoint] = [
    Endpoint(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
        max_tokens=2048,
        temperature=0.6,
    ),
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        max_tokens=2048,
        temperature=0.6,
    ),
    Endpoint(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
        max_tokens=2048,
        temperature=0.6,
    ),
]

SCRIPT_CHAIN: list[Endpoint] = [
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "openai/gpt-oss-120b:free",
        max_tokens=2048,
        temperature=0.75,
    ),
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "qwen/qwen3-coder-480b:free",
        max_tokens=2048,
        temperature=0.75,
    ),
    Endpoint(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
        max_tokens=2048,
        temperature=0.75,
    ),
]

ANALYZE_CHAIN: list[Endpoint] = CRITIC_CHAIN  # same reasoning, same models

VISION_CHAIN: list[Endpoint] = [
    Endpoint(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
        max_tokens=1024,
        temperature=0.3,
    ),
    Endpoint(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        max_tokens=1024,
        temperature=0.3,
    ),
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
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((RateLimitError, APIStatusError, httpx.HTTPError)),
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

    try:
        r = await client.chat.completions.create(**kwargs)
    except APIStatusError as e:
        # 429/5xx — let tenacity retry; 4xx client errors bubble up fast.
        if e.status_code is not None and 400 <= e.status_code < 500 and e.status_code != 429:
            raise APIError(
                message=str(e),
                request=getattr(e, "request", None),
                body=getattr(e, "body", None),
            ) from e
        raise

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
        except Exception as e:
            log.warning("llm %s via %s/%s failed: %s", task, ep.provider, ep.model, e)
            last_err = e
            # Briefly back off before trying next provider
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
    return [p for p, key in [
        ("openrouter", "OPENROUTER_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("cerebras", "CEREBRAS_API_KEY"),
    ] if os.environ.get(key)]
