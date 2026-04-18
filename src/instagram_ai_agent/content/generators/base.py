"""Shared types and helpers for content generators."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from instagram_ai_agent.core.config import MEDIA_STAGED


@dataclass
class GeneratedContent:
    """The full output of a generator before queue insertion."""

    format: str
    media_paths: list[str]             # absolute paths on disk
    visible_text: str = ""             # on-image/on-video text for critic context
    caption_context: str = ""          # freeform hint for caption LLM
    generator: str = ""                # which generator produced it
    meta: dict = field(default_factory=dict)


def staging_path(format_name: str, suffix: str) -> Path:
    MEDIA_STAGED.mkdir(parents=True, exist_ok=True)
    name = f"{format_name}_{uuid.uuid4().hex[:10]}{suffix}"
    return MEDIA_STAGED / name
