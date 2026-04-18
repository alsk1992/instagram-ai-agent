"""Reel → carousel repurpose: config, candidate picking, keyframe math,
slide compression, render, dedup, orchestrator wiring."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from instagram_ai_agent.content.generators import carousel_repurpose as cr
from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.core import db


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#000", "#fff", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fresh = tmp_path / "brain.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", fresh)
    monkeypatch.setattr(db, "DB_PATH", fresh)
    db.close()
    db.init_db()
    yield db
    db.close()


def _iso_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _enqueue_posted_reel(
    tmp_path: Path,
    *,
    content_id_label: str = "r",
    days_ago: int = 14,
    format: str = "reel_stock",
    scenes: list[dict] | None = None,
    make_file: bool = True,
) -> tuple[int, Path]:
    """Insert a `posts`-status reel row, and (optionally) create the mp4."""
    mp4 = tmp_path / f"{content_id_label}.mp4"
    if make_file:
        mp4.write_bytes(b"fake-mp4-bytes")
    scenes = scenes if scenes is not None else [
        {"query": "dad pullup", "line": "Most guys quit pullups too early."},
        {"query": "hanging", "line": "Start with dead hangs for two weeks."},
        {"query": "progress", "line": "Then add negatives every other day."},
    ]
    cid = db.content_enqueue(
        format=format,
        caption="old reel",
        hashtags=[],
        media_paths=[str(mp4)],
        phash=None,
        critic_score=None,
        critic_notes=None,
        generator=f"{format}:tips",
        status="posted",
        meta={"scenes": scenes, "variant": "tips"},
    )
    # Stamp posted_at manually to control age
    db.get_conn().execute(
        "UPDATE content_queue SET posted_at=? WHERE id=?",
        (_iso_ago(days_ago), cid),
    )
    return cid, mp4


# ─── Config ───
def test_repurpose_config_defaults_sane():
    cfg = _mkcfg()
    assert cfg.reel_repurpose.enabled is False
    assert cfg.reel_repurpose.min_reel_age_days == 7
    assert cfg.reel_repurpose.max_reel_age_days == 60
    assert 3 <= cfg.reel_repurpose.max_slides <= 10
    assert cfg.reel_repurpose.template_variant == "photo_caption"


def test_repurpose_config_roundtrips():
    import yaml
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=7))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.reel_repurpose.enabled is True
    assert loaded.reel_repurpose.max_slides == 7


def test_repurpose_config_rejects_absurd_slides():
    with pytest.raises(Exception):
        cfg_mod.ReelRepurposeConfig(max_slides=42)
    with pytest.raises(Exception):
        cfg_mod.ReelRepurposeConfig(max_slides=1)


def test_repurpose_config_rejects_inverted_age_window():
    """Audit fix: min > max silently collapsed the eligibility window."""
    with pytest.raises(Exception):
        cfg_mod.ReelRepurposeConfig(min_reel_age_days=60, max_reel_age_days=7)


# ─── pick_candidate ───
def test_pick_candidate_returns_none_when_disabled(tmp_db, tmp_path: Path):
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg()  # disabled by default
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_finds_in_age_window(tmp_db, tmp_path: Path):
    cid, mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    got = cr.pick_candidate(cfg)
    assert got is not None
    assert got.content_id == cid
    assert got.mp4_path == mp4
    assert len(got.scenes) == 3


def test_pick_candidate_rejects_too_recent(tmp_db, tmp_path: Path):
    _enqueue_posted_reel(tmp_path, days_ago=2)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, min_reel_age_days=7))
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_rejects_too_old(tmp_db, tmp_path: Path):
    _enqueue_posted_reel(tmp_path, days_ago=90)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_reel_age_days=60))
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_skips_missing_mp4(tmp_db, tmp_path: Path):
    cid, mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    mp4.unlink()
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_skips_non_reel_formats(tmp_db, tmp_path: Path):
    # Enqueue a carousel with scenes (shouldn't ever happen but belt+braces)
    db.content_enqueue(
        format="carousel", caption="", hashtags=[], media_paths=["/tmp/x.jpg"],
        phash=None, critic_score=None, critic_notes=None, generator="carousel",
        status="posted", meta={"scenes": [{"line": "x"}]},
    )
    db.get_conn().execute(
        "UPDATE content_queue SET posted_at=? WHERE format='carousel'",
        (_iso_ago(14),),
    )
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_skips_already_repurposed(tmp_db, tmp_path: Path):
    cid, _mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    cr._mark_repurposed(cid)
    assert cr.pick_candidate(cfg) is None


def test_pick_candidate_picks_most_recent_eligible(tmp_db, tmp_path: Path):
    # Two eligible — newest wins (ORDER BY posted_at DESC)
    c_old, _ = _enqueue_posted_reel(tmp_path, content_id_label="old", days_ago=40)
    c_new, _ = _enqueue_posted_reel(tmp_path, content_id_label="new", days_ago=10)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    got = cr.pick_candidate(cfg)
    assert got is not None
    assert got.content_id == c_new


def test_pick_candidate_skips_empty_scenes(tmp_db, tmp_path: Path):
    _enqueue_posted_reel(tmp_path, days_ago=14, scenes=[])
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    assert cr.pick_candidate(cfg) is None


# ─── scene midpoint math ───
def test_scene_midpoints_word_proportional():
    scenes = [
        {"line": "one two three four"},    # 4 words
        {"line": "five six"},                # 2 words
        {"line": "seven eight nine ten eleven twelve"},  # 6 words → widest
    ]
    mids = cr._scene_midpoints(scenes, duration_s=24.0)
    # Total words = 12; durations = 8, 4, 12s → midpoints = 4, 10, 18
    assert mids == pytest.approx([4.0, 10.0, 18.0], abs=0.01)


def test_scene_midpoints_clamps_to_safe_range():
    scenes = [{"line": "only scene here"}]
    mids = cr._scene_midpoints(scenes, duration_s=0.5)
    assert 0.1 <= mids[0] <= 0.4


def test_scene_midpoints_empty_when_no_duration():
    assert cr._scene_midpoints([{"line": "x"}], duration_s=0) == []
    assert cr._scene_midpoints([], duration_s=10) == []


# ─── keyframe subsampling ───
def test_extract_keyframes_subsamples_when_scenes_exceed_max(monkeypatch, tmp_path: Path):
    # Stub ffprobe duration + ffmpeg frame grab
    monkeypatch.setattr(cr, "_probe_duration", lambda p: 30.0)
    saved: list[tuple[float, Path]] = []

    def fake_extract(mp4, ts, out):
        saved.append((ts, out))
        out.write_bytes(b"\xff\xd8fake")
        return True

    monkeypatch.setattr(cr, "_extract_frame", fake_extract)

    scenes = [{"line": f"scene {i} with some words"} for i in range(8)]
    out_dir = tmp_path / "kf"
    frames = cr.extract_keyframes(Path("/fake.mp4"), scenes, out_dir, max_frames=5)
    assert len(frames) == 5
    # Timestamps should span the reel (monotonic-ish, NOT all at 0)
    timestamps = [ts for ts, _ in saved]
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] > timestamps[0]


def test_extract_keyframes_handles_fewer_scenes_than_max(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cr, "_probe_duration", lambda p: 10.0)

    def fake_extract(mp4, ts, out):
        out.write_bytes(b"\xff\xd8fake")
        return True

    monkeypatch.setattr(cr, "_extract_frame", fake_extract)
    scenes = [{"line": "a b c"}, {"line": "d e f"}]
    frames = cr.extract_keyframes(Path("/fake.mp4"), scenes, tmp_path / "kf", max_frames=5)
    assert len(frames) == 2


def test_extract_keyframes_returns_empty_on_zero_duration(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cr, "_probe_duration", lambda p: 0.0)
    monkeypatch.setattr(cr, "_extract_frame", lambda *a, **k: True)
    frames = cr.extract_keyframes(Path("/fake.mp4"), [{"line": "x"}], tmp_path, max_frames=3)
    assert frames == []


def test_extract_keyframes_continues_on_single_grab_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cr, "_probe_duration", lambda p: 12.0)
    calls = {"n": 0}

    def fake_extract(mp4, ts, out):
        calls["n"] += 1
        if calls["n"] == 2:
            return False  # second scene's grab fails
        out.write_bytes(b"\xff\xd8fake")
        return True

    monkeypatch.setattr(cr, "_extract_frame", fake_extract)
    scenes = [{"line": "a b"}, {"line": "c d"}, {"line": "e f"}]
    frames = cr.extract_keyframes(Path("/fake.mp4"), scenes, tmp_path, max_frames=5)
    assert len(frames) == 2  # one scene dropped, two succeed


# ─── heuristic slide builder ───
def test_heuristic_slide_splits_title_and_body():
    out = cr._heuristic_slide("Most guys quit pullups too early and never hit ten reps.")
    assert out["title"]
    assert out["body"]
    assert len(out["title"].split()) <= 6


def test_heuristic_slide_handles_empty():
    out = cr._heuristic_slide("")
    assert out == {"title": "", "body": ""}


def test_heuristic_slide_repeats_when_line_shorter_than_title_cap():
    out = cr._heuristic_slide("Just do it.")
    # Both fields populated even when body would be empty
    assert out["title"]
    assert out["body"]


def test_heuristic_fallback_respects_n_slides():
    lines = ["a", "b", "c", "d"]
    out = cr._heuristic_fallback(lines, n_slides=3)
    assert len(out) == 3
    assert out[0]["kind"] == "hook"
    assert out[-1]["kind"] == "cta"
    assert all(s["kind"] == "content" for s in out[1:-1])


def test_heuristic_fallback_pads_when_too_few_lines():
    out = cr._heuristic_fallback(["only line"], n_slides=3)
    assert len(out) == 3


def test_heuristic_fallback_injects_cta_when_last_body_lacks_it():
    out = cr._heuristic_fallback(["one line here with enough words"], n_slides=3)
    last = out[-1]
    assert any(w in last["body"].lower() for w in ("save", "follow", "share", "tag", "try"))


def test_heuristic_fallback_respects_existing_cta_language():
    # A long enough line that the body half still contains the CTA word.
    cta_line = "One honest tip before you scroll please save this post for later"
    out = cr._heuristic_fallback(["hook line", "middle", cta_line], n_slides=3)
    last = out[-1]
    # Body contains a CTA word → NO augmentation should fire
    assert "save" in last["body"].lower()
    assert "save this and try it" not in last["body"].lower()


# ─── _llm_compress fallback ───
@pytest.mark.asyncio
async def test_llm_compress_falls_back_on_llm_failure(monkeypatch):
    async def broken_llm(*a, **k):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(cr, "generate_json", broken_llm)
    cfg = _mkcfg()
    out = await cr._llm_compress(cfg, ["a b c", "d e f", "save this"], n_slides=3)
    assert len(out) == 3
    assert out[0]["kind"] == "hook"


@pytest.mark.asyncio
async def test_llm_compress_falls_back_when_too_few_slides(monkeypatch):
    async def stingy_llm(*a, **k):
        return {"slides": [{"kind": "hook", "title": "only", "body": ""}]}

    monkeypatch.setattr(cr, "generate_json", stingy_llm)
    cfg = _mkcfg()
    out = await cr._llm_compress(cfg, ["a b c", "d e f", "g h i"], n_slides=3)
    # Must still return 3 slides via the heuristic fallback
    assert len(out) == 3


@pytest.mark.asyncio
async def test_llm_compress_enforces_hook_and_cta_kinds(monkeypatch):
    """Audit fix: if the LLM mislabels kinds, we force slide 0 = hook
    and slide N-1 = cta so the template rendering stays consistent."""
    async def disobedient_llm(*a, **k):
        return {"slides": [
            {"kind": "content", "title": "Wrong", "body": "A"},
            {"kind": "content", "title": "Middle", "body": "B"},
            {"kind": "content", "title": "Also wrong", "body": "C"},
        ]}

    monkeypatch.setattr(cr, "generate_json", disobedient_llm)
    cfg = _mkcfg()
    out = await cr._llm_compress(cfg, ["x", "y", "z"], n_slides=3)
    assert out[0]["kind"] == "hook"
    assert out[-1]["kind"] == "cta"


@pytest.mark.asyncio
async def test_llm_compress_passes_through_valid_response(monkeypatch):
    async def good_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "Pullup math", "body": ""},
            {"kind": "content", "title": "Dead hangs", "body": "Two weeks, every morning."},
            {"kind": "cta", "title": "Start today", "body": "Save this. Try tomorrow."},
        ]}

    monkeypatch.setattr(cr, "generate_json", good_llm)
    cfg = _mkcfg()
    out = await cr._llm_compress(cfg, ["hook line", "middle line", "cta line"], n_slides=3)
    assert [s["kind"] for s in out] == ["hook", "content", "cta"]
    assert out[0]["title"] == "Pullup math"


# ─── dedup mark/prune ───
def test_mark_repurposed_persists(tmp_db):
    cr._mark_repurposed(42)
    assert "42" in cr._seen()


def test_mark_repurposed_prunes_entries_older_than_year(tmp_db):
    # Inject a stale entry directly
    db.state_set_json(
        cr._SEEN_KEY,
        {
            "42": (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "43": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    cr._mark_repurposed(44)
    seen = cr._seen()
    assert "42" not in seen
    assert "43" in seen
    assert "44" in seen


def test_seen_empty_when_state_missing(tmp_db):
    assert cr._seen() == {}


# ─── generate() wiring ───
@pytest.mark.asyncio
async def test_generate_returns_none_when_disabled(tmp_db, tmp_path: Path):
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg()
    assert await cr.generate(cfg) is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_no_eligible_reel(tmp_db):
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    assert await cr.generate(cfg) is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_keyframe_extraction_empty(
    tmp_db, tmp_path: Path, monkeypatch,
):
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))

    monkeypatch.setattr(cr, "extract_keyframes", lambda *a, **k: [])
    assert await cr.generate(cfg) is None


@pytest.mark.asyncio
async def test_generate_rejects_template_without_background_image_hook(
    tmp_db, tmp_path: Path, monkeypatch,
):
    """Audit fix: if the configured template doesn't declare
    $background_image, skip rather than wasting keyframe work on a
    template that will render without the photo."""
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    def fake_extract(mp4_path, scenes, out_dir, *, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(3):
            p = out_dir / f"kf_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract)
    # Return a template string WITHOUT $background_image
    monkeypatch.setattr(
        cr, "pick_template",
        lambda folder, *, variant=None: ("default", "<html>$title $body</html>"),
    )

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "A", "body": ""},
            {"kind": "content", "title": "B", "body": "bb"},
            {"kind": "cta", "title": "C", "body": "cc"},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)
    assert await cr.generate(cfg) is None


@pytest.mark.asyncio
async def test_generate_cleans_up_work_dir_on_success(
    tmp_db, tmp_path: Path, monkeypatch,
):
    """The keyframe workdir must not linger under MEDIA_STAGED after
    a successful render — Playwright inlines frames as base64 so the
    on-disk originals are disposable."""
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    captured: dict[str, Path] = {}

    def fake_extract(mp4_path, scenes, out_dir, *, max_frames):
        captured["work_dir"] = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(3):
            p = out_dir / f"kf_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract)

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    monkeypatch.setattr(cr, "render_html_to_png", fake_render)
    monkeypatch.setattr(cr, "apply_lut_image", lambda p, c: p)

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "A", "body": ""},
            {"kind": "content", "title": "B", "body": "bb"},
            {"kind": "cta", "title": "C", "body": "cc"},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)
    content = await cr.generate(cfg)
    assert content is not None
    assert "work_dir" in captured
    assert not captured["work_dir"].exists(), "repurpose work dir must be cleaned up"


@pytest.mark.asyncio
async def test_generate_cleans_up_partial_slides_on_mid_render_failure(
    tmp_db, tmp_path: Path, monkeypatch,
):
    """If slide 2 of 3 fails, slide 1's JPEG must not leak into staging."""
    _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    def fake_extract(mp4_path, scenes, out_dir, *, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(3):
            p = out_dir / f"kf_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract)

    rendered: list[Path] = []
    call_n = {"n": 0}

    async def flaky_render(html, out, **kw):
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise RuntimeError("playwright hiccup")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        rendered.append(out)
        return out

    monkeypatch.setattr(cr, "render_html_to_png", flaky_render)
    monkeypatch.setattr(cr, "apply_lut_image", lambda p, c: p)

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "A", "body": ""},
            {"kind": "content", "title": "B", "body": "bb"},
            {"kind": "cta", "title": "C", "body": "cc"},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)
    assert await cr.generate(cfg) is None
    # The one slide that did render must now be gone
    for p in rendered:
        assert not p.exists(), f"partial slide {p} leaked after mid-render failure"


@pytest.mark.asyncio
async def test_generate_produces_content_end_to_end(
    tmp_db, tmp_path: Path, monkeypatch,
):
    cid, mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    # Stub ffmpeg and the LUT pass so tests don't depend on ffmpeg/PIL state
    def fake_extract_keyframes(mp4_path, scenes, out_dir, *, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(min(max_frames, len(scenes))):
            p = out_dir / f"kf_{i}.jpg"
            # Valid-ish JPEG magic bytes so _image_data_url doesn't blow up
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract_keyframes)

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    monkeypatch.setattr(cr, "render_html_to_png", fake_render)
    monkeypatch.setattr(cr, "apply_lut_image", lambda p, c: p)

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "Pullup myth", "body": ""},
            {"kind": "content", "title": "Try hangs", "body": "Dead hang 30s daily."},
            {"kind": "cta", "title": "Start now", "body": "Save this and try tomorrow."},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)

    content = await cr.generate(cfg)
    assert content is not None
    assert content.format == "carousel"
    assert len(content.media_paths) == 3
    assert content.meta["source"] == "repurpose_reel"
    assert content.meta["source_reel_content_id"] == cid
    assert content.meta["source_reel_format"] == "reel_stock"


# ─── run_once enqueues + dedups ───
@pytest.mark.asyncio
async def test_run_once_enqueues_and_marks_dedup(
    tmp_db, tmp_path: Path, monkeypatch,
):
    cid, _mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    def fake_extract(mp4_path, scenes, out_dir, *, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(3):
            p = out_dir / f"kf_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract)

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    monkeypatch.setattr(cr, "render_html_to_png", fake_render)
    monkeypatch.setattr(cr, "apply_lut_image", lambda p, c: p)

    # Stub caption + hashtags + dedup to avoid full dep chain
    from instagram_ai_agent.content import captions as cap_mod
    from instagram_ai_agent.content import hashtags as ht_mod
    from instagram_ai_agent.content import dedup as dd_mod

    async def fake_caption(*a, **k):
        return "Repurposed carousel caption."

    monkeypatch.setattr(cap_mod, "generate_caption", fake_caption)
    monkeypatch.setattr(ht_mod, "build_hashtags", lambda cfg: ["calisthenics"])
    monkeypatch.setattr(ht_mod, "format_hashtags", lambda t: " ".join(f"#{x}" for x in t))
    monkeypatch.setattr(dd_mod, "compute_phash", lambda p: "deadbeef")
    monkeypatch.setattr(dd_mod, "is_duplicate", lambda h, thr: (False, None))

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "Pullup myth", "body": ""},
            {"kind": "content", "title": "Try hangs", "body": "Dead hang 30s daily."},
            {"kind": "cta", "title": "Start now", "body": "Save this and try tomorrow."},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)

    new_cid = await cr.run_once(cfg)
    assert new_cid is not None
    assert new_cid != cid  # it's a new row
    row = db.content_get(new_cid)
    assert row is not None
    assert row["format"] == "carousel"
    assert row["meta"]["source"] == "repurpose_reel"
    assert row["meta"]["source_reel_content_id"] == cid
    assert row["hashtags"] == ["calisthenics"]

    # Dedup: marked so a second run skips
    assert str(cid) in cr._seen()
    second = await cr.run_once(cfg)
    assert second is None


@pytest.mark.asyncio
async def test_run_once_skips_on_phash_duplicate_but_marks_source(
    tmp_db, tmp_path: Path, monkeypatch,
):
    """Audit guard: if the render happens to phash-collide, we still mark
    the source reel as repurposed so we don't re-attempt forever."""
    cid, _mp4 = _enqueue_posted_reel(tmp_path, days_ago=14)
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True, max_slides=3))

    def fake_extract(mp4_path, scenes, out_dir, *, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(3):
            p = out_dir / f"kf_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            paths.append(p)
        return paths

    monkeypatch.setattr(cr, "extract_keyframes", fake_extract)

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xd8jpeg")
        return out

    monkeypatch.setattr(cr, "render_html_to_png", fake_render)
    monkeypatch.setattr(cr, "apply_lut_image", lambda p, c: p)

    from instagram_ai_agent.content import dedup as dd_mod
    monkeypatch.setattr(dd_mod, "compute_phash", lambda p: "deadbeef")
    monkeypatch.setattr(dd_mod, "is_duplicate", lambda h, thr: (True, "deadbeef"))

    async def fake_llm(*a, **k):
        return {"slides": [
            {"kind": "hook", "title": "Pullup myth", "body": ""},
            {"kind": "content", "title": "Try hangs", "body": "Dead hang 30s daily."},
            {"kind": "cta", "title": "Start now", "body": "Save this."},
        ]}

    monkeypatch.setattr(cr, "generate_json", fake_llm)

    result = await cr.run_once(cfg)
    assert result is None
    assert str(cid) in cr._seen()


# ─── orchestrator wiring ───
@pytest.mark.asyncio
async def test_orchestrator_registers_repurpose_job_when_enabled():
    from instagram_ai_agent import orchestrator
    cfg = _mkcfg(reel_repurpose=cfg_mod.ReelRepurposeConfig(enabled=True))
    orch = orchestrator.Orchestrator(cfg)
    orch.start()
    try:
        assert orch.scheduler.get_job("repurpose") is not None
    finally:
        orch.scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_orchestrator_skips_repurpose_when_disabled():
    from instagram_ai_agent import orchestrator
    cfg = _mkcfg()  # disabled by default
    orch = orchestrator.Orchestrator(cfg)
    orch.start()
    try:
        assert orch.scheduler.get_job("repurpose") is None
    finally:
        orch.scheduler.shutdown(wait=False)
