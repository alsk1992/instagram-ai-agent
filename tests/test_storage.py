"""R2 storage graceful-fallback tests (without boto3 mocks — pure config)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core import storage


def test_unconfigured_returns_false(monkeypatch: pytest.MonkeyPatch):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    assert storage.configured() is False


def test_archive_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "m.jpg"
    f.write_bytes(b"x")
    out = storage.archive_posted_media(1, [str(f)])
    assert out == [str(f)]
    # Local file must remain untouched when unconfigured
    assert f.exists()


def test_cleanup_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "m.jpg"
    f.write_bytes(b"x")
    removed = storage.cleanup_local([str(f)])
    assert removed == 0
    assert f.exists()


def test_configured_reads_all_env(monkeypatch: pytest.MonkeyPatch):
    # Reset cached client
    monkeypatch.setattr(storage, "_client", None)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    assert storage.configured() is True


def test_upload_unconfigured_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(storage, "_client", None)
    f = tmp_path / "m.jpg"
    f.write_bytes(b"x")
    assert storage.upload(f, "k") is None
