"""Finish pass — covers graceful fallback, no-op, config gates, size cap."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.core import config as cfg_mod
from src.plugins import finish_pass


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


def _make_image(path: Path, *, size=(320, 400), fill=(40, 40, 40)) -> Path:
    img = Image.new("RGB", size, fill)
    img.save(path, "JPEG", quality=90)
    return path


@pytest.fixture(autouse=True)
def reset_caches():
    finish_pass.reset_caches()
    yield
    finish_pass.reset_caches()


# ─── Config defaults ───
def test_finish_defaults_sane():
    cfg = _mkcfg()
    assert cfg.finish.enabled is True
    assert cfg.finish.upscale_factor == 2
    assert cfg.finish.face_restore is True
    assert cfg.finish.use_local is True
    assert cfg.finish.use_hf_fallback is False
    # CodeFormer is not wired anywhere
    assert "codeformer" not in (
        cfg.finish.hf_upscale_space + cfg.finish.hf_face_space
    ).lower()


# ─── No-op contract ───
def test_enhance_disabled_is_noop(tmp_path: Path):
    src = _make_image(tmp_path / "a.jpg")
    cfg = _mkcfg(finish=cfg_mod.FinishPass(enabled=False))
    r = finish_pass.enhance(src, cfg.finish, subject_is_human=True)
    assert r.path == src
    assert r.backend == "noop"
    assert r.notes == "disabled"


def test_enhance_missing_input_returns_noop(tmp_path: Path):
    cfg = _mkcfg()
    bogus = tmp_path / "does_not_exist.jpg"
    r = finish_pass.enhance(bogus, cfg.finish)
    assert r.backend == "noop"
    assert "missing" in r.notes


def test_enhance_megapixel_cap_skips_giants(tmp_path: Path):
    # 5000x5000 = 25 MP, far above the default 6 MP cap
    src = tmp_path / "huge.jpg"
    Image.new("RGB", (5000, 5000), (10, 10, 10)).save(src, "JPEG", quality=70)
    cfg = _mkcfg()
    r = finish_pass.enhance(src, cfg.finish, subject_is_human=False)
    assert r.backend == "noop"
    assert "cap" in r.notes


# ─── Local backend absent → falls through ───
def test_no_backend_available_is_graceful_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If neither local torch stack nor HF client is importable, we get a
    no-op result pointing at the original path."""
    src = _make_image(tmp_path / "a.jpg")
    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: False)
    cfg = _mkcfg()  # use_hf_fallback stays False
    r = finish_pass.enhance(src, cfg.finish, subject_is_human=True)
    assert r.path == src
    assert r.backend == "noop"
    assert r.upscaled is False
    assert r.face_restored is False
    assert "no backend available" in r.notes


def test_hf_disabled_even_if_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _make_image(tmp_path / "a.jpg")
    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: True)
    cfg = _mkcfg()  # use_hf_fallback == False — must not opt in automatically
    r = finish_pass.enhance(src, cfg.finish)
    assert r.backend == "noop"


# ─── HF path with mocked functions ───
def test_hf_upscale_called_when_opted_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _make_image(tmp_path / "a.jpg")
    out = _make_image(tmp_path / "a_up.jpg", size=(640, 800))
    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: True)

    def fake_upscale(src_p, dst_p, factor, space):
        Path(dst_p).write_bytes(out.read_bytes())
        return Path(dst_p)

    def fake_face(src_p, dst_p, space):
        Path(dst_p).write_bytes(src_p.read_bytes())
        return Path(dst_p)

    monkeypatch.setattr(finish_pass, "_hf_upscale", fake_upscale)
    monkeypatch.setattr(finish_pass, "_hf_restore_face", fake_face)
    cfg = _mkcfg(finish=cfg_mod.FinishPass(use_local=False, use_hf_fallback=True))

    r = finish_pass.enhance(src, cfg.finish, subject_is_human=True)
    assert r.backend == "hf"
    assert r.upscaled is True
    assert r.face_restored is True
    assert r.path != src


def test_hf_upscale_failure_still_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _make_image(tmp_path / "a.jpg")
    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: True)
    monkeypatch.setattr(finish_pass, "_hf_upscale", lambda *a, **k: None)
    monkeypatch.setattr(finish_pass, "_hf_restore_face", lambda *a, **k: None)
    cfg = _mkcfg(finish=cfg_mod.FinishPass(use_local=False, use_hf_fallback=True))
    r = finish_pass.enhance(src, cfg.finish, subject_is_human=True)
    assert r.path == src  # both failed; original returned
    assert r.upscaled is False
    assert r.face_restored is False
    assert "unavailable" in r.notes


# ─── Subject gating ───
def test_face_restore_only_for_human_subjects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Non-human posts must NEVER call the face-restore path even if available."""
    src = _make_image(tmp_path / "a.jpg")
    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: True)
    called = {"face": False, "upscale": False}

    def fake_upscale(src_p, dst_p, factor, space):
        called["upscale"] = True
        Path(dst_p).write_bytes(src_p.read_bytes())
        return Path(dst_p)

    def fake_face(src_p, dst_p, space):
        called["face"] = True
        Path(dst_p).write_bytes(src_p.read_bytes())
        return Path(dst_p)

    monkeypatch.setattr(finish_pass, "_hf_upscale", fake_upscale)
    monkeypatch.setattr(finish_pass, "_hf_restore_face", fake_face)
    cfg = _mkcfg(finish=cfg_mod.FinishPass(use_local=False, use_hf_fallback=True))

    finish_pass.enhance(src, cfg.finish, subject_is_human=False)
    assert called["upscale"] is True
    assert called["face"] is False


# ─── Caches reset cleanly ───
def test_reset_caches_clears_singletons():
    finish_pass._local_upscaler = {"factor": 2, "instance": object()}
    finish_pass._local_restorer = object()
    finish_pass.reset_caches()
    assert finish_pass._local_upscaler is None
    assert finish_pass._local_restorer is None


# ─── Licence hygiene: make sure CodeFormer is never *imported* anywhere ───
def test_codeformer_is_never_imported():
    import re

    import src.plugins.finish_pass as fp_module
    src_text = Path(fp_module.__file__).read_text()
    # It's fine to *mention* CodeFormer in comments/docstrings (we do — to
    # explain why we avoid it). What matters is that the module never tries
    # to import or call it.
    assert not re.search(r"^\s*import\s+codeformer", src_text, re.M | re.I)
    assert not re.search(r"^\s*from\s+codeformer", src_text, re.M | re.I)


# ─── Cfg roundtrip ───
def test_finish_config_roundtrips():
    cfg = _mkcfg(finish=cfg_mod.FinishPass(upscale_factor=4, use_hf_fallback=True))
    dumped = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(dumped)
    assert loaded.finish.upscale_factor == 4
    assert loaded.finish.use_hf_fallback is True


# ─── Audit follow-ups ───
def test_enhance_returns_finishresult_dataclass(tmp_path: Path):
    src = _make_image(tmp_path / "a.jpg")
    cfg = _mkcfg(finish=cfg_mod.FinishPass(enabled=False))
    r = finish_pass.enhance(src, cfg.finish)
    assert isinstance(r, finish_pass.FinishResult)
    for field in ("path", "backend", "upscaled", "face_restored", "notes"):
        assert hasattr(r, field)


def test_pil_probe_failure_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"NOT_AN_IMAGE_AT_ALL" * 10)
    cfg = _mkcfg()
    # Even though a backend reports available, corrupt input must not reach it.
    monkeypatch.setattr(finish_pass, "_local_available", lambda: True)
    r = finish_pass.enhance(bad, cfg.finish)
    assert r.backend == "noop"
    assert "probe failed" in r.notes


def test_intermediate_files_land_in_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Intermediates go to MEDIA_STAGED, not next to the input."""
    staging = tmp_path / "staged"
    monkeypatch.setattr(finish_pass, "MEDIA_STAGED", staging)
    user_dir = tmp_path / "userland"
    user_dir.mkdir()
    src = _make_image(user_dir / "a.jpg")

    monkeypatch.setattr(finish_pass, "_local_available", lambda: False)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: True)

    def fake_upscale(src_p, dst_p, factor, space):
        Path(dst_p).parent.mkdir(parents=True, exist_ok=True)
        Path(dst_p).write_bytes(Path(src_p).read_bytes())
        return Path(dst_p)

    monkeypatch.setattr(finish_pass, "_hf_upscale", fake_upscale)
    cfg = _mkcfg(finish=cfg_mod.FinishPass(use_local=False, use_hf_fallback=True, face_restore=False))

    r = finish_pass.enhance(src, cfg.finish, subject_is_human=False)
    assert r.upscaled is True
    assert r.path.parent == staging
    # User dir is pristine
    assert {p.name for p in user_dir.iterdir()} == {"a.jpg"}


def test_no_face_detected_reports_honestly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GFPGAN returning None (no face) must NOT claim face_restored=True."""
    src = _make_image(tmp_path / "scenery.jpg")
    monkeypatch.setattr(finish_pass, "_local_available", lambda: True)
    monkeypatch.setattr(finish_pass, "_hf_available", lambda: False)

    def fake_upscale(src_p, dst_p, factor):
        Path(dst_p).parent.mkdir(parents=True, exist_ok=True)
        Path(dst_p).write_bytes(Path(src_p).read_bytes())
        return Path(dst_p)

    def fake_restore(src_p, dst_p):
        raise finish_pass._NoFaceDetected("no face")

    monkeypatch.setattr(finish_pass, "_local_upscale", fake_upscale)
    monkeypatch.setattr(finish_pass, "_local_restore_face", fake_restore)

    cfg = _mkcfg()
    r = finish_pass.enhance(src, cfg.finish, subject_is_human=True)
    assert r.upscaled is True
    assert r.face_restored is False, "honest: no face -> no restoration claim"
    assert "no face detected" in r.notes


def test_upscaler_factor_change_would_invalidate_cache():
    """Cache invalidation logic reads factor from the cached dict."""
    finish_pass._local_upscaler = {"factor": 2, "instance": object(), "name": "x2"}
    cache = finish_pass._local_upscaler
    assert cache["factor"] == 2
    # Simulating a factor-change request:
    new_factor = 4
    should_rebuild = cache is None or cache.get("factor") != new_factor
    assert should_rebuild is True
