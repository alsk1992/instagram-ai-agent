"""Sanity tests for LLM JSON extraction (the repair/strip logic)."""
from __future__ import annotations

from instagram_ai_agent.core.llm import _extract_balanced, _strip_json


def test_strip_markdown_fences():
    raw = '```json\n{"x": 1}\n```'
    assert _strip_json(raw) == '{"x": 1}'


def test_strip_no_fence():
    raw = 'Sure! Here\'s the JSON:\n{"x": 1, "y": 2}'
    assert _strip_json(raw) == '{"x": 1, "y": 2}'


def test_extract_balanced_trims_trailing_garbage():
    raw = '{"x": 1}  and here is some trailing text'
    assert _extract_balanced(raw) == '{"x": 1}'


def test_extract_balanced_nested():
    raw = '{"a": {"b": [1, 2]}}trailing'
    assert _extract_balanced(raw) == '{"a": {"b": [1, 2]}}'
