"""Beat-synced cuts — pure-function snap tests, graceful librosa absence."""
from __future__ import annotations

from pathlib import Path

import pytest

from instagram_ai_agent.plugins import beat_sync


# ─── Pure conversion helpers ───
def test_durs_to_boundaries():
    assert beat_sync.durs_to_boundaries([1.0, 2.5, 0.5]) == [1.0, 3.5, 4.0]
    assert beat_sync.durs_to_boundaries([]) == []


def test_boundaries_to_durs_inverse():
    durs = [1.0, 2.5, 0.5, 3.0]
    bounds = beat_sync.durs_to_boundaries(durs)
    assert beat_sync.boundaries_to_durs(bounds) == durs


# ─── snap_boundaries — empty / trivial ───
def test_snap_no_beats_returns_input():
    assert beat_sync.snap_boundaries([1.0, 2.0], []) == [1.0, 2.0]


def test_snap_single_boundary_returns_input():
    """Only one boundary exists → it's the final VO-anchored one, can't move."""
    assert beat_sync.snap_boundaries([5.0], [0.5, 1.0, 1.5, 2.0]) == [5.0]


def test_snap_last_boundary_is_never_touched():
    """The final boundary equals the VO duration; leave it alone even if a
    beat is within the window."""
    boundaries = [1.0, 2.0, 5.0]
    beats = [0.9, 1.9, 4.95]   # 4.95 is inside the 0.3 window of 5.0
    out = beat_sync.snap_boundaries(boundaries, beats)
    assert out[-1] == 5.0


# ─── Happy path ───
def test_snap_within_window():
    boundaries = [2.00, 4.00, 6.00]
    # Beats at 1.95 and 3.90 — within ±0.3s of first two boundaries
    beats = [0.5, 1.0, 1.5, 1.95, 2.5, 3.0, 3.5, 3.90, 4.5, 5.0, 5.5]
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.3, min_scene_s=0.8)
    assert out[0] == pytest.approx(1.95)
    assert out[1] == pytest.approx(3.90)
    assert out[2] == 6.00   # last stays put


def test_snap_outside_window_noop():
    boundaries = [2.00, 4.00, 6.00]
    # Beats are far from the boundaries — nothing should move
    beats = [0.5, 1.0, 5.0, 5.5]
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.3)
    assert out == boundaries


def test_snap_prefers_nearest_beat():
    boundaries = [2.00, 5.00]
    # Two beats within window; 2.08 is closer to 2.00 than 2.25
    beats = [2.25, 2.08, 1.80]
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.3, min_scene_s=0.8)
    assert out[0] == pytest.approx(2.08)


# ─── Min-scene guard ───
def test_snap_refuses_to_shrink_scene_below_min():
    # Boundary at 1.0, next boundary at 1.5, min_scene_s=0.8 → moving
    # boundary forward must NOT happen (would make next scene < 0.8s)
    boundaries = [1.00, 1.50, 5.00]
    beats = [1.20]   # within window of 1.0 but moving would break scene 2 (1.20-1.50=0.30 < 0.8)
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.3, min_scene_s=0.8)
    assert out[0] == 1.00, "must NOT snap since it would shrink scene 2"


def test_snap_refuses_to_shrink_previous_scene():
    boundaries = [2.00, 5.00]
    # Beat at 1.80 — window ok, but prev_end=0 so scene 1 would be 1.80 (>min). OK.
    beats = [1.80]
    out = beat_sync.snap_boundaries(boundaries, beats, min_scene_s=0.8)
    assert out[0] == 1.80

    # Now a tighter case: prev_end > 0, moving would shrink scene N-1.
    # boundaries [1.0, 2.0, 5.0], move 2.0 back to 1.5: scene 2 = 0.5 (below min).
    boundaries = [1.00, 2.00, 5.00]
    beats = [1.50]
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.6, min_scene_s=0.8)
    assert out[1] == 2.00, "must NOT snap — would make scene 2 too short"


# ─── snap_scene_durs (high-level) ───
def test_snap_scene_durs_without_librosa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If librosa is unavailable, snap_scene_durs is a no-op (returns durs + False)."""
    monkeypatch.setattr(beat_sync, "_librosa_available", lambda: False)
    fake_music = tmp_path / "m.mp3"
    fake_music.write_bytes(b"not real audio")
    durs, changed = beat_sync.snap_scene_durs(
        [2.0, 3.0, 4.0], fake_music, vo_duration_s=9.0,
    )
    assert durs == [2.0, 3.0, 4.0]
    assert changed is False


def test_snap_scene_durs_no_beats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No beats detected → original durations preserved."""
    monkeypatch.setattr(beat_sync, "detect_beats", lambda *a, **k: [])
    fake = tmp_path / "m.mp3"; fake.write_bytes(b"x")
    durs, changed = beat_sync.snap_scene_durs(
        [2.0, 3.0, 4.0], fake, vo_duration_s=9.0,
    )
    assert durs == [2.0, 3.0, 4.0]
    assert changed is False


def test_snap_scene_durs_applies_snap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """With stubbed beats landing on interior boundaries, durations shift."""
    # Original boundaries: [2.0, 5.0, 9.0]; VO duration = 9.0
    monkeypatch.setattr(
        beat_sync, "detect_beats",
        lambda *a, **k: [0.5, 1.0, 1.5, 2.10, 2.5, 3.0, 4.85, 5.5, 7.0, 8.5],
    )
    fake = tmp_path / "m.mp3"; fake.write_bytes(b"x")
    durs, changed = beat_sync.snap_scene_durs(
        [2.0, 3.0, 4.0], fake, vo_duration_s=9.0,
    )
    assert changed is True
    # Boundaries moved to 2.10 and 4.85; scene durs become 2.10, 2.75, 4.15
    assert durs[0] == pytest.approx(2.10, abs=1e-6)
    assert durs[1] == pytest.approx(2.75, abs=1e-6)
    # Final scene closes at exactly VO duration
    assert durs[0] + durs[1] + durs[2] == pytest.approx(9.0, abs=1e-6)


def test_snap_scene_durs_empty_input():
    durs, changed = beat_sync.snap_scene_durs([], Path("/nonexistent"), vo_duration_s=0.0)
    assert durs == []
    assert changed is False


# ─── detect_beats graceful ───
def test_detect_beats_missing_file_returns_empty():
    assert beat_sync.detect_beats(Path("/does/not/exist.mp3")) == []


def test_detect_beats_librosa_absent_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Without librosa installed, detect_beats must return [] not crash."""
    fake = tmp_path / "m.mp3"; fake.write_bytes(b"x")
    monkeypatch.setattr(beat_sync, "_librosa_available", lambda: False)
    assert beat_sync.detect_beats(fake) == []


# ─── Post-snap invariant ───
def test_post_snap_no_scene_shorter_than_min():
    """Audit-critical invariant: after snap, every scene is ≥ min_scene_s."""
    boundaries = [1.0, 2.5, 5.0, 8.0, 12.0]
    beats = [0.98, 2.45, 3.0, 4.9, 7.85]
    out = beat_sync.snap_boundaries(boundaries, beats, window_s=0.3, min_scene_s=0.8)
    durs = beat_sync.boundaries_to_durs(out)
    assert all(d >= 0.79 for d in durs), durs   # 1e-2 tolerance for float drift


def test_vo_duration_is_preserved():
    """The total must remain VO duration even after snap."""
    boundaries = [2.0, 4.0, 6.0, 10.0]
    beats = [1.90, 3.85, 5.80, 9.95]
    out = beat_sync.snap_boundaries(boundaries, beats)
    assert out[-1] == 10.0


# ─── Audit follow-ups ───
def test_music_config_exposes_beat_knobs():
    from instagram_ai_agent.core.config import MusicConfig
    mc = MusicConfig()
    assert hasattr(mc, "beat_window_s")
    assert hasattr(mc, "beat_min_scene_s")
    assert 0.0 <= mc.beat_window_s <= 1.5
    assert 0.3 <= mc.beat_min_scene_s <= 5.0


def test_beat_sync_window_zero_short_circuits():
    """A niche can disable beat-sync without removing librosa by setting
    window_s=0. Call sites gate on cfg.music.beat_window_s > 0."""
    durs, changed = beat_sync.snap_scene_durs(
        [2.0, 3.0, 4.0], Path("/nonexistent"),
        vo_duration_s=9.0, window_s=0.0,
    )
    assert durs == [2.0, 3.0, 4.0]
    assert changed is False


def test_snap_custom_min_scene_floor_is_respected():
    """Higher min_scene_s blocks snaps a lower floor would allow."""
    boundaries = [2.00, 2.80, 6.00]
    beats_hit = [2.10]
    relaxed = beat_sync.snap_boundaries(boundaries, beats_hit, window_s=0.3, min_scene_s=0.6)
    assert relaxed[0] == pytest.approx(2.10)
    strict = beat_sync.snap_boundaries(boundaries, beats_hit, window_s=0.3, min_scene_s=0.8)
    assert strict[0] == 2.00