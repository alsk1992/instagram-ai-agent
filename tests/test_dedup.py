"""Smoke tests for PHash + Hamming."""
from __future__ import annotations

from src.content.dedup import hamming


def test_hamming_identical():
    assert hamming("abcd1234" * 4, "abcd1234" * 4) == 0


def test_hamming_single_bit():
    # 1-bit difference between 0 and 1 in the last nibble
    assert hamming("0", "1") == 1
    assert hamming("f", "e") == 1


def test_hamming_length_mismatch():
    assert hamming("ff", "fff") >= 4
