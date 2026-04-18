"""Content deduplication — perceptual hashing (images/video) + caption Hamming."""
from __future__ import annotations

import subprocess
from pathlib import Path

import imagehash
from PIL import Image

from instagram_ai_agent.core import db


def compute_phash(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        frame = _extract_middle_frame(p)
        try:
            with Image.open(frame) as img:
                h = imagehash.phash(img, hash_size=16)
        finally:
            frame.unlink(missing_ok=True)
    else:
        with Image.open(p) as img:
            h = imagehash.phash(img, hash_size=16)
    return str(h)


def hamming(a: str, b: str) -> int:
    """Hamming distance between two hex hash strings, per-bit."""
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    ia = int(a, 16)
    ib = int(b, 16)
    return bin(ia ^ ib).count("1")


def is_duplicate(new_hash: str, threshold: int, lookback: int = 60) -> tuple[bool, str | None]:
    """Return (is_dup, match_hash_if_any). Compare against recent queue."""
    for h in db.existing_phashes(lookback):
        if hamming(new_hash, h) <= threshold:
            return True, h
    return False, None


def _extract_middle_frame(video: Path) -> Path:
    out = video.with_suffix(".dedupframe.jpg")
    # Probe duration
    dur = _probe_duration(video)
    midpoint = max(0.1, dur / 2.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{midpoint:.2f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "4",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _probe_duration(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 1.0
