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
#
# Two separate endpoint pools because freeform text and structured JSON have
# different failure modes:
#
#   FREEFORM_CHAIN — used by `generate()` for prose (captions, reel scripts,
#   vision descriptions). Accepts `openrouter/free` (the auto-router) as the
#   top of the chain because prose-quality is forgiving and the router spreads
#   load across the whole :free pool.
#
#   JSON_CHAIN — used by `generate_json()` for anything structured. We
#   DELIBERATELY exclude `openrouter/free` and `minimax/minimax-m2.5:free`
#   here: empirical evidence from this project's logs shows the router
#   routes JSON-mode calls to 1-4B models (Gemma-3n-e2b, LFM-2.5) that
#   either (a) reject `response_format=json_object` with 400
#   "Developer instruction is not enabled for models/gemma-3n-e2b-it",
#   (b) return empty completions, or (c) emit chain-of-thought prose
#   before the JSON and get truncated before reaching it. MiniMax-M2.5
#   returns empty completions in ~30% of JSON-mode calls. Neither is
#   fit for schema-following output.
#
#   JSON_CHAIN is ordered big-reasoning-first. Nemotron-120B is a
#   reasoning-tuned MoE and follows JSON schemas reliably; Gemma-4-31B
#   is a solid mid-tier; Nemotron-Nano and Gemma-4-26B are fallbacks;
#   Trinity is a final hedge. All are ≥26B params — the smallest model
#   in this pool is ~13x larger than the ones the router likes to pick.
_OR = "https://openrouter.ai/api/v1"

FREEFORM_CAPTION_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=1024, temperature=0.85),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "arcee-ai/trinity-large-preview:free",
             max_tokens=1024, temperature=0.85),
]

FREEFORM_SCRIPT_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "arcee-ai/trinity-large-preview:free",
             max_tokens=2048, temperature=0.75),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=2048, temperature=0.75),
]

# Vision: only models that accept image inputs. Verified 2026-04-20 via
# /api/v1/models — the Gemma-4 :free SKUs and the openrouter/free router
# advertise image support; Nemotron / MiniMax / Arcee / LFM are text-only.
FREEFORM_VISION_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "openrouter/free",
             max_tokens=1024, temperature=0.3),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",
             max_tokens=1024, temperature=0.3),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",
             max_tokens=1024, temperature=0.3),
]

# JSON-mode endpoints: big schema-following models only. This is the single
# pool used by generate_json() regardless of the caller's `task` label —
# the task label still drives max_tokens / temperature defaults, but never
# endpoint selection. Ordered reasoning-first.
JSON_CHAIN: list[Endpoint] = [
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "nvidia/nemotron-3-super-120b-a12b:free",
             max_tokens=2048, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "google/gemma-4-31b-it:free",
             max_tokens=2048, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "nvidia/nemotron-3-nano-30b-a3b:free",
             max_tokens=2048, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "google/gemma-4-26b-a4b-it:free",
             max_tokens=2048, temperature=0.2),
    Endpoint("openrouter", _OR, "OPENROUTER_API_KEY",
             "arcee-ai/trinity-large-preview:free",
             max_tokens=2048, temperature=0.2),
]

# Legacy aliases — kept so any callers that still reference CAPTION_CHAIN
# et al. keep working. ROUTES is the canonical lookup for `generate()`.
CAPTION_CHAIN = FREEFORM_CAPTION_CHAIN
SCRIPT_CHAIN = FREEFORM_SCRIPT_CHAIN
VISION_CHAIN = FREEFORM_VISION_CHAIN
CRITIC_CHAIN = JSON_CHAIN
BULK_CHAIN = JSON_CHAIN
ANALYZE_CHAIN = JSON_CHAIN


ROUTES: dict[Task, list[Endpoint]] = {
    "caption": FREEFORM_CAPTION_CHAIN,
    "critic":  JSON_CHAIN,
    "bulk":    JSON_CHAIN,
    "script":  FREEFORM_SCRIPT_CHAIN,
    "analyze": JSON_CHAIN,
    "vision":  FREEFORM_VISION_CHAIN,
}


# Per-task defaults for generate_json (since JSON_CHAIN is shared, each
# task's endpoint max_tokens/temperature is the same — caller overrides
# via kwargs if they need different values).
_JSON_TASK_DEFAULTS: dict[str, tuple[int, float]] = {
    "caption": (1024, 0.3),   # short structured captions
    "critic":  (1024, 0.1),   # deterministic scoring
    "bulk":    (2500, 0.3),   # longer lists (themes, ideas)
    "script":  (2048, 0.4),
    "analyze": (2500, 0.1),   # theme clustering, angle brainstorm
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


# Markers that signal chain-of-thought preamble from small :free models.
# When ``strip_cot`` is enabled in generate(), we detect these and either
# trim to the line after them or reject the response so the chain falls
# through to the next endpoint. Real posted captions were corrupted by
# responses like "We need to output caption text only: ..." — the model
# spoke its reasoning out loud and we saved the reasoning as the caption.
_COT_MARKERS = (
    "we need to output",
    "we need to write",
    "we should output",
    "we should write",
    "we must output",
    "the caption is:",
    "caption text:",
    "let me think",
    "thinking:",
    "analysis:",
    "reasoning:",
    "okay, so the",
    "okay, let me",
    "so we need",
    "first, let me",
    "here's the caption:",
    "here is the caption:",
    "final caption:",
    "output:",
)


def _strip_cot(text: str) -> str:
    """Strip chain-of-thought preamble from freeform LLM output.

    Two-stage:
      1. If the text starts with (or its first 600 chars contain) a CoT
         marker, look for a later line that introduces the final answer —
         common patterns: ``Final caption:``, ``Output:``, or a trailing
         quoted block — and return that. Otherwise return the empty string
         so the caller can treat this as a bad response and fall through.
      2. If no marker is present, return the original text unchanged.

    We only examine the first 600 chars for markers because legitimate
    captions can include the word 'output' or 'we' and we don't want to
    false-positive on a 200-word story caption that happens to contain
    reflexive pronouns mid-text.
    """
    if not text:
        return text
    head = text[:600].lower()
    marker_hit = any(m in head for m in _COT_MARKERS)
    if not marker_hit:
        return text

    # Look for a final-answer introducer that tells us where the real
    # caption starts. We try several patterns in priority order.
    import re as _re
    intro_patterns = [
        r"(?:^|\n)\s*(?:final caption|final answer|output|caption|result)\s*:\s*(.+)$",
        r'"([^"]{20,})"\s*\.?\s*$',   # last quoted block at end of text
        r"(?:^|\n)\s*['“]([^'”]{20,})['”]\s*\.?\s*$",  # smart-quoted
    ]
    for pat in intro_patterns:
        m = _re.search(pat, text, flags=_re.IGNORECASE | _re.DOTALL | _re.MULTILINE)
        if m:
            candidate = m.group(1).strip().strip('"').strip("'").strip()
            if candidate and len(candidate) >= 10:
                return candidate
    # CoT detected but no clean answer extractable — return empty so
    # generate()'s chain loop falls through to a different endpoint.
    return ""


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
            # Strip chain-of-thought preamble from small :free models that
            # spell out their reasoning before the final answer. When the
            # response is pure CoT with no extractable answer, _strip_cot
            # returns "" and we treat it as an endpoint failure so the
            # next model in the chain gets a chance.
            if not json_mode:
                cleaned = _strip_cot(out)
                if not cleaned:
                    log.warning(
                        "llm %s via %s/%s: response was chain-of-thought "
                        "with no extractable answer — trying next",
                        task, ep.provider, ep.model,
                    )
                    last_err = ValueError("chain-of-thought response")
                    await asyncio.sleep(0.3)
                    continue
                if cleaned != out:
                    log.debug(
                        "llm %s via %s/%s: stripped CoT preamble (%d→%d chars)",
                        task, ep.provider, ep.model, len(out), len(cleaned),
                    )
                out = cleaned
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
            # Demote benign fall-through cases ("Empty completion" from
            # openrouter/free's auto-router) to debug — they're not
            # actionable, and the chain is working as designed when the
            # next endpoint fills in. APIError message is "Empty completion"
            # exactly.
            msg = str(e)
            level = log.debug if "Empty completion" in msg else log.warning
            level("llm %s via %s/%s failed: %s", task, ep.provider, ep.model, e)
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
    expect: Literal["object", "array", "any"] = "object",
) -> Any:
    """Generate and parse JSON. Tolerates providers that don't support strict JSON mode.

    ``expect`` lets the caller state the shape they need. When set to "object"
    (default) and the model returns ``[{...}]``, we unwrap the single dict —
    small models (Gemma-3n, LFM-2.5) frequently wrap a single object in an
    array even when asked for an object. If we can't coerce, we raise so
    ``generate()``'s outer loop falls through to the next endpoint.
    """
    enriched = (
        (system or "")
        + "\nRespond with valid JSON only. No prose, no markdown fences, "
        + "no chain-of-thought. Start with { or [."
    ).strip()
    messages: list[dict[str, Any]] = []
    if enriched:
        messages.append({"role": "system", "content": enriched})
    messages.append({"role": "user", "content": prompt})

    # JSON-mode calls always use JSON_CHAIN — schema-following models only.
    # See the JSON_CHAIN comment in this file for why openrouter/free and
    # minimax-m2.5 are excluded here (tiny-model CoT / empty completions).
    # On parse/shape rejection we fall through to the next endpoint in the
    # chain just like on 429/5xx — bad JSON is an endpoint failure, not a
    # job failure.
    default_mt, default_temp = _JSON_TASK_DEFAULTS.get(task, (2048, 0.2))
    effective_temp = temperature if temperature is not None else default_temp
    effective_mt = max_tokens or default_mt
    chain = JSON_CHAIN
    last_err: Exception | None = None
    for ep in chain:
        if _client_for(ep) is None:
            continue
        cooling = _is_cooling_down(ep)
        if cooling > 0:
            log.debug("llm_json %s skip %s/%s — cooldown %.0fs left",
                      task, ep.provider, ep.model, cooling)
            continue
        try:
            raw = await _call_one(
                ep,
                messages,
                max_tokens=effective_mt,
                temperature=effective_temp,
                json_mode=True,
            )
        except RateLimitError as e:
            _park(ep, _retry_after_seconds(e) or _DEFAULT_COOLDOWN_S)
            last_err = e
            continue
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                _park(ep, _retry_after_seconds(e) or _DEFAULT_COOLDOWN_S)
            elif code and code >= 500:
                await asyncio.sleep(0.5 + random.random())
            log.warning("llm_json %s via %s/%s failed (%s): %s",
                        task, ep.provider, ep.model, code, e)
            last_err = e
            continue
        except Exception as e:
            msg = str(e)
            level = log.debug if "Empty completion" in msg else log.warning
            level("llm_json %s via %s/%s failed: %s",
                  task, ep.provider, ep.model, e)
            last_err = e
            continue

        try:
            parsed = _parse_and_coerce_json(raw, expect)
            log.debug("llm_json %s via %s/%s ok", task, ep.provider, ep.model)
            return parsed
        except ValueError as e:
            log.warning("llm_json %s via %s/%s: JSON rejected — %s (trying next)",
                        task, ep.provider, ep.model, str(e)[:180])
            last_err = e
            await asyncio.sleep(0.3)
            continue

    raise AllProvidersFailed(
        f"All providers failed for JSON task={task}: {last_err!r}"
    ) from last_err


def _parse_and_coerce_json(raw: str, expect: Literal["object", "array", "any"]) -> Any:
    """Parse raw text into JSON of the expected shape, with two repair passes.
    Raises ValueError on unrecoverable failure so callers can try another endpoint."""
    cleaned = _strip_json(raw)
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Repair 1: balanced-braces extract (trims trailing CoT prose)
        repaired = _extract_balanced(cleaned)
        if repaired:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass
        # Repair 2: truncation repair (model hit max_tokens mid-value)
        if parsed is None:
            repaired = _repair_truncated_json(cleaned)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        if parsed is None:
            raise ValueError(f"unparseable JSON: {raw[:200]!r}")

    # Shape coercion — small :free models routinely ignore json_object and
    # return a single-element array when asked for an object (or wrap an
    # array inside a one-key object when asked for an array).
    if expect == "object" and isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        raise ValueError(f"list where object expected: {raw[:200]!r}")
    if expect == "array" and isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"object where array expected: {raw[:200]!r}")
    return parsed


def _bracket_stack(src: str) -> tuple[list[str], bool]:
    """Return (unclosed-brackets-stack, is-currently-inside-string) for src."""
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in src:
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    return stack, in_str


def _repair_truncated_json(s: str) -> str | None:
    """Recover JSON truncated mid-value by progressively trimming the tail and
    closing unclosed brackets. Returns the first candidate that parses, or
    None if nothing under ~500 chars of trim recovers.

    Common failure this handles: model hit ``max_tokens`` partway through an
    element (a string or object). We trim back to the nearest "clean break"
    (after the previous complete value) and synthesise matching closers.
    """
    closers = {"{": "}", "[": "]"}
    # Trim progressively — start minimal, escalate up to ~500 chars.
    for trim in range(0, min(len(s), 500)):
        candidate = s[: len(s) - trim]
        # Strip trailing whitespace/commas/colons/partial-key artefacts
        candidate = candidate.rstrip().rstrip(",").rstrip()
        stack, in_str = _bracket_stack(candidate)
        if in_str:
            # Mid-string — keep trimming until we're outside the string
            continue
        if not stack:
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        # Close unclosed brackets in reverse order
        closed = candidate
        for opener in reversed(stack):
            closed += closers[opener]
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            continue
    return None


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
