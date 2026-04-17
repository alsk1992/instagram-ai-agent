"""Beat-synced cuts — librosa-powered boundary snapping for reels.

Single responsibility: given a music file and a list of scene boundary
times, nudge each boundary onto the nearest musical beat so edits land
on the downbeat instead of mid-bar.

Two public surfaces:

  * :func:`detect_beats(path)` — wraps ``librosa.beat.beat_track`` with a
    tight exception budget. Returns ``[]`` on any failure (missing
    librosa, corrupt file, quiet audio). Never raises.
  * :func:`snap_boundaries(boundaries, beats, ...)` — pure function, no
    librosa import, so it tests trivially. Respects a configurable
    ``window_s`` (only snap boundaries that already fall within this
    distance of a beat) and ``min_scene_s`` (never create a scene
    shorter than this).

We deliberately avoid BeatNet / beat_this (both CC-BY-NC) — librosa is
ISC-licensed and the quality gap is not material for reel-length content.
"""
from __future__ import annotations

from pathlib import Path

from src.core.logging_setup import get_logger

log = get_logger(__name__)


# ───── librosa availability ─────
def _librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def detect_beats(
    audio_path: str | Path,
    *,
    duration_s: float | None = None,
    sr: int = 22050,
) -> list[float]:
    """Return beat timestamps in seconds. Empty list on any failure.

    ``duration_s`` optionally caps how much of the file we analyse — the
    mixer truncates music to the VO length anyway, so beats past that
    point are useless.
    """
    path = Path(audio_path)
    if not path.exists():
        return []
    if not _librosa_available():
        return []

    import librosa

    try:
        y, _sr = librosa.load(str(path), sr=sr, mono=True, duration=duration_s)
        if y.size == 0:
            return []
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=_sr)
        if beat_frames is None:
            return []
        times = librosa.frames_to_time(beat_frames, sr=_sr)
        return [float(t) for t in times if t >= 0]
    except Exception as e:
        log.debug("beat detection failed on %s: %s", path.name, e)
        return []


# ───── Pure snapping logic ─────
def snap_boundaries(
    boundaries: list[float],
    beats: list[float],
    *,
    window_s: float = 0.3,
    min_scene_s: float = 0.8,
) -> list[float]:
    """Snap interior boundaries onto the nearest beat.

    * The LAST boundary is never moved — it equals the total VO duration
      which the mixer uses as ground truth.
    * A boundary only moves if the nearest beat is within ``window_s``.
    * A boundary only moves if the move doesn't shrink either adjacent
      scene below ``min_scene_s``.
    * Beats don't need to be sorted; we sort internally.
    """
    if not beats or len(boundaries) < 2:
        return list(boundaries)

    sorted_beats = sorted(beats)
    result = list(boundaries)
    for i in range(len(result) - 1):  # never touch the last boundary
        original = result[i]
        prev_end = result[i - 1] if i > 0 else 0.0
        next_end = result[i + 1]

        # Nearest beat within window
        best = None
        best_dist = window_s + 1
        for b in sorted_beats:
            if b - original > window_s:
                break
            if original - b > window_s:
                continue
            d = abs(b - original)
            if d < best_dist:
                best_dist = d
                best = b

        if best is None:
            continue

        # Guard against too-short neighbouring scenes
        if (best - prev_end) < min_scene_s:
            continue
        if (next_end - best) < min_scene_s:
            continue

        result[i] = best
    return result


# ───── Convenience converters ─────
def durs_to_boundaries(durs: list[float]) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for d in durs:
        acc += d
        out.append(acc)
    return out


def boundaries_to_durs(boundaries: list[float]) -> list[float]:
    out: list[float] = []
    prev = 0.0
    for b in boundaries:
        out.append(b - prev)
        prev = b
    return out


def snap_scene_durs(
    scene_durs: list[float],
    audio_path: str | Path,
    *,
    vo_duration_s: float,
    window_s: float = 0.3,
    min_scene_s: float = 0.8,
) -> tuple[list[float], bool]:
    """High-level helper: take scene durations + a music path, return
    beat-snapped durations and a boolean indicating whether any snap
    actually happened (for observability / meta).

    When librosa is absent or no beats were detected, ``snapped == False``
    and the input durations are returned unchanged.
    """
    if not scene_durs:
        return list(scene_durs), False
    beats = detect_beats(audio_path, duration_s=vo_duration_s)
    if not beats:
        return list(scene_durs), False

    # Use VO_duration as the final boundary anchor; beat-snap interior ones
    boundaries = durs_to_boundaries(scene_durs)
    # Force the last element to the exact VO duration (defensive: drift from
    # float math during proportional rescaling).
    if boundaries:
        boundaries[-1] = float(vo_duration_s)
    snapped = snap_boundaries(
        boundaries, beats, window_s=window_s, min_scene_s=min_scene_s,
    )
    changed = any(abs(a - b) > 1e-6 for a, b in zip(snapped, boundaries, strict=True))
    return boundaries_to_durs(snapped), changed
