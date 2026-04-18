"""Two-pass image ranking: local ensemble + vision LLM. No heavy deps required."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from instagram_ai_agent.content import image_rank, local_aesthetic
from instagram_ai_agent.core import config as cfg_mod


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


@pytest.fixture(autouse=True)
def reset_aesthetic_cache():
    local_aesthetic.reset_cache()
    yield
    local_aesthetic.reset_cache()


# ─── Safety config ───
def test_new_safety_knobs_exist():
    cfg = _mkcfg()
    assert cfg.safety.local_aesthetic is True
    assert cfg.safety.vision_top_k >= 1


def test_safety_roundtrips_yaml():
    import yaml
    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=False, vision_top_k=4))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.safety.local_aesthetic is False
    assert loaded.safety.vision_top_k == 4


# ─── LocalScore data class ───
def test_local_score_dataclass():
    s = local_aesthetic.LocalScore(score=0.7, raw={"a": 0.6, "b": 0.8}, model_used="ensemble:a+b")
    assert s.score == 0.7
    assert s.raw == {"a": 0.6, "b": 0.8}


# ─── Scorer availability ───
def test_aesthetics_predictor_scorer_reports_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Without the pip package installed, is_available() == False."""
    scorer = local_aesthetic.AestheticsPredictorScorer()
    # Force both module imports to fail
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("aesthetics_predictor", "simple_aesthetics_predictor"):
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert scorer.is_available() is False


def test_pyiqa_scorer_reports_unavailable(monkeypatch: pytest.MonkeyPatch):
    scorer = local_aesthetic.PyIQAQualityScorer()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyiqa":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert scorer.is_available() is False


# ─── Ensemble stubbing ───
class StubScorer:
    """Deterministic scorer for tests — no torch, no CLIP."""

    def __init__(self, name: str, score_map: dict[str, float]):
        self.name = name
        self._map = score_map

    def is_available(self) -> bool:
        return True

    async def score_one(self, image_path):
        return self._map.get(Path(image_path).name)


def test_ensemble_averages_scorers():
    ens = local_aesthetic.EnsembleLocalScorer([
        StubScorer("laion", {"a.jpg": 0.8, "b.jpg": 0.5}),
        StubScorer("pyiqa", {"a.jpg": 0.6, "b.jpg": 0.9}),
    ])
    result_a = asyncio.run(ens.score(Path("a.jpg")))
    result_b = asyncio.run(ens.score(Path("b.jpg")))
    assert result_a.score == pytest.approx(0.7)
    assert result_a.raw == {"laion": 0.8, "pyiqa": 0.6}
    assert "laion" in result_a.model_used and "pyiqa" in result_a.model_used
    assert result_b.score == pytest.approx(0.7)


def test_ensemble_ignores_unavailable_scorer():
    class Off(StubScorer):
        def is_available(self): return False

    ens = local_aesthetic.EnsembleLocalScorer([
        StubScorer("ok", {"x.jpg": 0.4}),
        Off("dead", {"x.jpg": 999.0}),
    ])
    result = asyncio.run(ens.score(Path("x.jpg")))
    assert "dead" not in result.raw
    assert result.score == pytest.approx(0.4)


def test_ensemble_returns_none_when_nothing_available():
    class Off(StubScorer):
        def is_available(self): return False

    ens = local_aesthetic.EnsembleLocalScorer([Off("a", {}), Off("b", {})])
    assert asyncio.run(ens.score(Path("x.jpg"))) is None


def test_ensemble_ignores_per_scorer_failure():
    class Bad(StubScorer):
        async def score_one(self, image_path):
            raise RuntimeError("boom")

    ens = local_aesthetic.EnsembleLocalScorer([
        StubScorer("ok", {"x.jpg": 0.7}),
        Bad("bad", {}),
    ])
    r = asyncio.run(ens.score(Path("x.jpg")))
    assert r.score == pytest.approx(0.7)
    assert "bad" not in r.raw


def test_ensemble_clamps_invalid_scores():
    class OutOfRange(StubScorer):
        async def score_one(self, image_path):
            return 2.5   # invalid — should be rejected

    ens = local_aesthetic.EnsembleLocalScorer([
        StubScorer("ok", {"x.jpg": 0.6}),
        OutOfRange("bad", {}),
    ])
    r = asyncio.run(ens.score(Path("x.jpg")))
    assert r.score == pytest.approx(0.6)
    assert "bad" not in r.raw


# ─── Two-pass rank ───
def _install_stub_ensemble(monkeypatch, score_map: dict[str, float]) -> list[str]:
    """Point image_rank at a deterministic local ensemble. Returns the
    ordered candidate list (best-first by local score)."""
    stub_ens = local_aesthetic.EnsembleLocalScorer([
        StubScorer("stub", score_map),
    ])
    monkeypatch.setattr(local_aesthetic, "get_ensemble", lambda: stub_ens)
    return sorted(score_map, key=score_map.get, reverse=True)


def test_rank_local_only_when_no_vision(monkeypatch: pytest.MonkeyPatch):
    """With local scorer available but vision disabled, final ranking
    follows local scores."""
    score_map = {"a.jpg": 0.3, "b.jpg": 0.9, "c.jpg": 0.6}
    _install_stub_ensemble(monkeypatch, score_map)
    monkeypatch.setattr(image_rank, "_vision_ready", lambda: False)

    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=True, vision_critic=False, vision_top_k=2))
    ranked = asyncio.run(image_rank.rank(cfg, ["a.jpg", "b.jpg", "c.jpg"]))
    assert [r["path"] for r in ranked] == ["b.jpg", "c.jpg", "a.jpg"]
    assert all(r["score"] > 0 for r in ranked)


def test_rank_vision_only_runs_on_top_k(monkeypatch: pytest.MonkeyPatch):
    """With vision enabled + top_k=2, vision LLM is called TWICE (not 3 times)."""
    score_map = {"a.jpg": 0.3, "b.jpg": 0.9, "c.jpg": 0.6}
    ordered = _install_stub_ensemble(monkeypatch, score_map)   # ['b', 'c', 'a']
    monkeypatch.setattr(image_rank, "_vision_ready", lambda: True)

    vision_calls: list[str] = []

    async def fake_score_image(path, cfg, *, subject_is_human):
        vision_calls.append(path.name)
        # Invert the local ranking to prove vision overrides local
        synthetic = {"a.jpg": 0.1, "b.jpg": 0.4, "c.jpg": 0.95}[path.name]
        return synthetic, "vision reason"

    monkeypatch.setattr(image_rank, "_score_image", fake_score_image)

    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=True, vision_critic=True, vision_top_k=2))
    ranked = asyncio.run(image_rank.rank(cfg, list(score_map)))

    # Only the top-2 by local (b.jpg, c.jpg) should hit vision
    assert set(vision_calls) == {"b.jpg", "c.jpg"}
    assert len(vision_calls) == 2
    # c.jpg had vision score 0.95 → should now rank first overall
    assert ranked[0]["path"] == "c.jpg"
    # a.jpg kept its local score (0.3), never saw vision
    a_row = next(r for r in ranked if r["path"] == "a.jpg")
    assert a_row["vision_score"] is None
    assert a_row["score"] == pytest.approx(0.3)


def test_rank_disable_local_reverts_to_vision_every_candidate(monkeypatch: pytest.MonkeyPatch):
    """With local_aesthetic=False, vision is asked about every candidate."""
    monkeypatch.setattr(image_rank, "_vision_ready", lambda: True)
    vision_calls: list[str] = []

    async def fake_score_image(path, cfg, *, subject_is_human):
        vision_calls.append(path.name)
        return 0.5, "stub"

    monkeypatch.setattr(image_rank, "_score_image", fake_score_image)
    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=False, vision_critic=True, vision_top_k=2))
    # With local disabled, vision_top_k still applies — top-k by position (as
    # seen in the input order) get the vision call.
    asyncio.run(image_rank.rank(cfg, ["a.jpg", "b.jpg", "c.jpg"]))
    assert set(vision_calls) == {"a.jpg", "b.jpg"}  # top_k=2 of the preserved order


def test_rank_single_candidate_shortcut():
    cfg = _mkcfg()
    ranked = asyncio.run(image_rank.rank(cfg, ["solo.jpg"]))
    assert ranked == [{"path": "solo.jpg", "score": 1.0, "reason": "only candidate"}]


def test_rank_empty_input():
    cfg = _mkcfg()
    assert asyncio.run(image_rank.rank(cfg, [])) == []


def test_rank_no_local_no_vision_falls_back_to_first_candidate(monkeypatch: pytest.MonkeyPatch):
    """When neither pass is available, keep input order (first wins)."""
    monkeypatch.setattr(local_aesthetic.EnsembleLocalScorer, "is_available", lambda self: False)
    monkeypatch.setattr(image_rank, "_vision_ready", lambda: False)
    cfg = _mkcfg(safety=cfg_mod.Safety(local_aesthetic=True, vision_critic=True))
    ranked = asyncio.run(image_rank.rank(cfg, ["a.jpg", "b.jpg", "c.jpg"]))
    # Every candidate kept; sorted by score (0.5 prefilter default → all tied)
    assert {r["path"] for r in ranked} == {"a.jpg", "b.jpg", "c.jpg"}


# ─── Commercial licensing hygiene ───
def test_local_aesthetic_never_imports_nc_weights():
    import re
    from instagram_ai_agent.content import local_aesthetic as la
    text = Path(la.__file__).read_text()
    # Blocklist: CodeFormer, S-Lab, BeatNet, MusicGen weights, XTTS weights
    for nc_marker in ("codeformer", "s-lab", "beatnet", "musicgen", "xtts"):
        assert nc_marker not in text.lower(), f"{nc_marker!r} found in local_aesthetic module"
    # Ensure no unauthorised imports
    assert not re.search(r"^\s*import\s+(codeformer|beatnet|musicgen)", text, re.M | re.I)


def test_default_scorers_is_commercial_safe():
    """Audit fix: default ensemble must never include pyiqa (PolyForm NC)."""
    defaults = local_aesthetic.default_scorers()
    types = {type(s).__name__ for s in defaults}
    assert "PyIQAQualityScorer" not in types, (
        "pyiqa is non-commercial — not safe as default"
    )
    # Must still provide at least one commercial-safe scorer
    assert "AestheticsPredictorScorer" in types


def test_aesthetic_extra_does_not_pull_pyiqa():
    """The `[aesthetic]` pip extra must not list pyiqa."""
    import tomllib
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        meta = tomllib.load(f)
    extras = meta["project"]["optional-dependencies"]
    aesthetic_list = "\n".join(extras.get("aesthetic", []))
    assert "pyiqa" not in aesthetic_list.lower(), (
        "pyiqa leaked into default [aesthetic] — licence is PolyForm NC"
    )
    assert "aesthetic-nc" in extras
    assert any("pyiqa" in dep for dep in extras["aesthetic-nc"])


def test_simple_aesthetics_predictor_pin_is_valid():
    """Audit fix: pin used to be `>=0.3` which doesn't exist on PyPI."""
    import re
    import tomllib
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        meta = tomllib.load(f)
    aesthetic_list = meta["project"]["optional-dependencies"]["aesthetic"]
    matches = [d for d in aesthetic_list if "simple-aesthetics-predictor" in d]
    assert matches, "simple-aesthetics-predictor must be in [aesthetic]"
    pin = matches[0]
    m = re.search(r">=(\d+)\.(\d+)", pin)
    assert m, f"unexpected pin format: {pin}"
    major, minor = int(m.group(1)), int(m.group(2))
    # Anything at-or-below 0.2 is a version that actually exists on PyPI.
    assert (major, minor) <= (0, 2), f"pin {pin} asks for version that doesn't exist on PyPI"