"""Cloudflare R2 media storage (S3-compatible).

R2 is free up to 10 GB and has zero egress, making it the default for posted
media archive. When unconfigured, storage ops become no-ops and local files
are left in place (graceful fallback).

Expected env vars:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET
  R2_PUBLIC_URL     (optional, e.g. https://media.mypage.com)
"""
from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_url: str | None


def _config() -> R2Config | None:
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if not all(os.environ.get(k) for k in required):
        return None
    return R2Config(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
        public_url=os.environ.get("R2_PUBLIC_URL"),
    )


def configured() -> bool:
    return _config() is not None


_client = None  # lazy; boto3 is only imported when needed


def _get_client():
    global _client
    if _client is not None:
        return _client
    cfg = _config()
    if cfg is None:
        return None
    import boto3
    from botocore.config import Config

    _client = boto3.client(
        "s3",
        endpoint_url=f"https://{cfg.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )
    return _client


def upload(local_path: str | Path, key: str, *, content_type: str | None = None) -> str | None:
    """Upload a local file to R2. Returns the public URL (or s3:// path) or None if unconfigured."""
    cfg = _config()
    client = _get_client()
    if cfg is None or client is None:
        return None

    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {p}")

    ct = content_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    extra: dict = {"ContentType": ct}

    with open(p, "rb") as f:
        client.upload_fileobj(f, cfg.bucket, key, ExtraArgs=extra)

    log.info("r2 upload %s → s3://%s/%s", p.name, cfg.bucket, key)

    if cfg.public_url:
        return f"{cfg.public_url.rstrip('/')}/{key}"
    return f"s3://{cfg.bucket}/{key}"


def archive_posted_media(content_id: int, media_paths: list[str]) -> list[str]:
    """Upload each media file to R2 under content/<id>/<filename>, return R2 URLs.

    Returns the list of resulting URLs (or the originals if R2 is unconfigured).
    """
    if not configured():
        return media_paths
    out: list[str] = []
    for p in media_paths:
        src = Path(p)
        if not src.exists():
            out.append(p)
            continue
        key = f"content/{content_id}/{src.name}"
        try:
            url = upload(src, key)
            out.append(url or p)
        except Exception as e:
            log.warning("R2 upload failed for %s: %s", src.name, e)
            out.append(p)
    return out


def cleanup_local(media_paths: list[str], *, keep_thumbnails: bool = True) -> int:
    """Delete local staging files after they've been archived to R2.

    ``keep_thumbnails`` preserves ``.thumb.jpg`` next to reel videos for local
    inspection of past posts.
    """
    if not configured():
        return 0
    removed = 0
    for p in media_paths:
        path = Path(p)
        # Skip URLs (already in R2)
        if not path.is_absolute() and "://" in p:
            continue
        if not path.exists():
            continue
        if keep_thumbnails and path.suffix == ".jpg" and path.stem.endswith(".thumb"):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as e:
            log.debug("cleanup skipped %s: %s", path.name, e)
    return removed
