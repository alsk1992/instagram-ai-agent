"""Contrarian / hot-take mode — config, archetype bias, prompt
injection, hard safety blocklist, pipeline dice roll + thread through."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.brain import idea_bank
from src.content import captions, contrarian_safety, critic, pipeline as pipe
from src.core import config as cfg_mod
from src.core import db


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


# ─── Config ───
def test_contrarian_config_defaults():
    cfg = _mkcfg()
    assert cfg.contrarian.enabled is False
    assert cfg.contrarian.frequency == pytest.approx(0.15)
    assert cfg.contrarian.intensity == "moderate"
    assert "medical claims" in cfg.contrarian.avoid_topics


def test_contrarian_config_bounds():
    with pytest.raises(Exception):
        cfg_mod.ContrarianConfig(frequency=1.5)
    with pytest.raises(Exception):
        cfg_mod.ContrarianConfig(frequency=-0.1)


def test_contrarian_config_roundtrips():
    import yaml
    cfg = _mkcfg(contrarian=cfg_mod.ContrarianConfig(
        enabled=True, frequency=0.3, intensity="high",
        avoid_topics=["my custom topic"],
    ))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.contrarian.enabled is True
    assert loaded.contrarian.intensity == "high"
    assert loaded.contrarian.avoid_topics == ["my custom topic"]


# ─── is_contrarian_archetype ───
@pytest.mark.parametrize("archetype,expected", [
    ("contrarian_hot_take", True),
    ("contrarian_stop_doing", True),
    ("contrarian_overrated", True),
    ("trend_contra", True),
    ("myth_bust_single", True),
    ("myth_bust_list", True),
    ("listicle_mistakes", False),
    ("teaching_how_to", False),
    ("before_after_journey", False),
    ("trend_riff", False),       # riff, not contra — not contrarian
    ("", False),
    (None, False),
])
def test_is_contrarian_archetype(archetype, expected):
    assert idea_bank.is_contrarian_archetype(archetype or "") is expected


# ─── Caption system prompt injection ───
def test_build_system_no_contrarian_block_when_mode_off():
    cfg = _mkcfg()
    out = captions.build_system(cfg, "meme", contrarian=False)
    assert "CONTRARIAN MODE" not in out


def test_build_system_injects_contrarian_block_when_on():
    cfg = _mkcfg(contrarian=cfg_mod.ContrarianConfig(avoid_topics=["diet X"]))
    out = captions.build_system(cfg, "meme", contrarian=True)
    assert "CONTRARIAN MODE" in out
    assert "diet X" in out
    # Safety topics are always called out
    assert "medical" in out.lower()


def test_build_system_contrarian_includes_avoid_topics():
    cfg = _mkcfg(contrarian=cfg_mod.ContrarianConfig(
        avoid_topics=["nephew's recital", "goats"],
    ))
    out = captions.build_system(cfg, "carousel", contrarian=True)
    assert "nephew" in out
    assert "goats" in out


# ─── Idea bank bias ───
def test_pick_for_prefers_contrarian_when_flag_set(tmp_db):
    """Seeded the bank with a mix; with prefer_contrarian=True, the
    contrarian archetype should win the lion's share of 200 rolls."""
    # Insert minimal rows directly (skipping the JSON seed file)
    conn = tmp_db.get_conn()
    conn.execute("""
        INSERT INTO ideas (archetype, hook_formula, format_hint, niche_tags,
                           body_template, source, license, score)
        VALUES
          ('contrarian_hot_take',  'Unpopular: {x}',         'any', '[]', 't', 'c', 'CC0', 0.5),
          ('teaching_how_to',      'How to actually {x}',    'any', '[]', 't', 'c', 'CC0', 0.5),
          ('listicle_mistakes',    '{N} mistakes',           'any', '[]', 't', 'c', 'CC0', 0.5)
    """)
    conn.commit()

    cfg = _mkcfg()
    random.seed(1234)
    contrarian_wins = 0
    # Clear recency so every pick stays in the pool — 300 rolls
    for _ in range(300):
        idea = idea_bank.pick_for(
            cfg, format_name="carousel",
            commercial_only=False, prefer_contrarian=True,
        )
        if idea and idea.archetype == "contrarian_hot_take":
            contrarian_wins += 1

    # With 4× weight boost on 1 of 3 archetypes, contrarian should
    # win ~67% of rolls (4/6 = 66.7%). Allow generous tolerance.
    assert contrarian_wins > 140, f"expected majority wins, got {contrarian_wins}/300"


def test_pick_for_no_bias_when_flag_off(tmp_db):
    """Without the flag, the three archetypes should roughly split 1/3 each."""
    conn = tmp_db.get_conn()
    conn.execute("""
        INSERT INTO ideas (archetype, hook_formula, format_hint, niche_tags,
                           body_template, source, license, score)
        VALUES
          ('contrarian_hot_take',  'U',  'any', '[]', 't', 'c', 'CC0', 0.5),
          ('teaching_how_to',      'H',  'any', '[]', 't', 'c', 'CC0', 0.5),
          ('listicle_mistakes',    'L',  'any', '[]', 't', 'c', 'CC0', 0.5)
    """)
    conn.commit()

    cfg = _mkcfg()
    random.seed(4321)
    contrarian_wins = 0
    for _ in range(300):
        idea = idea_bank.pick_for(
            cfg, format_name="carousel",
            commercial_only=False, prefer_contrarian=False,
        )
        if idea and idea.archetype == "contrarian_hot_take":
            contrarian_wins += 1
    # Uniform expected ~100/300; with 4x boost absent should be 33%-ish
    assert 60 < contrarian_wins < 140


# ─── Contrarian safety — hard blocklist ───
def test_safety_passes_clean_hot_take():
    result = contrarian_safety.check(
        "Most coaches tell you to fast. That's wrong — here's the evidence.",
        "Fasting doesn't beat protein + progressive overload.",
    )
    assert result.safe is True
    assert result.reason is None


@pytest.mark.parametrize("payload,expected_label", [
    # Medical misinformation
    ("vaccines cause autism and they're lying to you", "vaccines"),
    ("avoid vaccines — your body has everything it needs", "vaccines_reverse"),
    ("This supplement cures cancer. Big Pharma hid it for years.", "cancer_cure"),
    ("Big Pharma hides the real cure for diabetes", "big_pharma"),
    # This payload is BOTH a cancer_cure claim and an alt-med overreach.
    # Earlier pattern (cancer_cure) wins — either label is acceptable.
    ("Essential oils cure cancer and depression", "cancer_cure"),
    ("Depression isn't real, it's just weakness", "anti_psychiatry"),

    # Conspiracy tropes
    ("flat earth is the obvious truth", "flat_earth"),
    ("moon landing was faked by NASA", "moon_landing_hoax"),
    ("chemtrails are poisoning us daily", "chemtrails"),
    ("climate change is a hoax designed to raise taxes", "climate_denial"),

    # Eating disorder glorification
    ("just starve yourself for two weeks", "eating_disorder_glorification"),
    ("eat under 500 calories a day for rapid loss", "extreme_calorie_restriction"),

    # Self-harm
    ("killing yourself is a valid option for depression", "self_harm"),

    # Blanket group claims
    ("all men are predators who deserve nothing", "blanket_group_claim"),
    ("every muslim should be deported", "blanket_group_claim"),
])
def test_safety_blocks_known_toxic_patterns(payload, expected_label):
    result = contrarian_safety.check(payload)
    assert result.safe is False
    assert result.reason == expected_label


def test_safety_checks_multiple_parts():
    """Pattern match across any of the supplied parts."""
    result = contrarian_safety.check(
        "completely innocuous caption",
        "chemtrails are poisoning us daily",
    )
    assert result.safe is False
    assert result.reason == "chemtrails"


def test_safety_handles_empty_inputs():
    assert contrarian_safety.check().safe is True
    assert contrarian_safety.check("").safe is True
    assert contrarian_safety.check(None, None).safe is True  # type: ignore[arg-type]


def test_safety_result_is_bool_coercible():
    clean = contrarian_safety.check("totally fine")
    dirty = contrarian_safety.check("chemtrails")
    assert bool(clean) is True
    assert bool(dirty) is False


def test_safety_normalises_whitespace_in_inputs():
    """A multi-line caption with odd whitespace must still match."""
    payload = "vaccines\n   cause  \n\n  autism"
    result = contrarian_safety.check(payload)
    assert result.safe is False


# ─── Audit-fix regression guards ───
@pytest.mark.parametrize("benign_phrase", [
    "skip the hype about vaccine-hesitancy posts",
    "don't take travel vaccine advice from TikTok",
    "avoid vaccine-related drama in the comments",
    "reject vaccine misinformation from influencers",
])
def test_safety_allows_benign_vaccine_adjacent(benign_phrase):
    """Audit fix #1: vaccines_reverse was too loose; must pass benign
    vaccine-hesitancy-debunking or travel-advice framings."""
    assert contrarian_safety.check(benign_phrase).safe is True


@pytest.mark.parametrize("positive_group_phrase", [
    "all trans people deserve healthcare",
    "every muslim family deserves respect",
    "all women are welcome here",
    "no men are excluded from this space",
])
def test_safety_allows_positive_group_phrasings(positive_group_phrase):
    """Audit fix #2: blanket_group_claim was too broad; must NOT match
    positive / advocacy framings of protected-class groups."""
    assert contrarian_safety.check(positive_group_phrase).safe is True


@pytest.mark.parametrize("payload,expected_label", [
    ("Big Pharma is lying about side effects", "big_pharma"),
    ("big pharma lied about opioids for decades", "big_pharma"),
    ("Big Pharma deceived the public for years", "big_pharma"),
    ("starve for three days for fast results", "eating_disorder_glorification"),
    ("starving for a full week resets your metabolism", "eating_disorder_glorification"),
])
def test_safety_catches_audit_false_negatives(payload, expected_label):
    """Audit fix #3: new patterns for the mutations the auditor flagged."""
    result = contrarian_safety.check(payload)
    assert result.safe is False
    assert result.reason == expected_label


def test_safety_normalises_unicode_variants():
    """Audit fix: NFKC-normalise fullwidth chars + strip zero-width
    joiners so obfuscated payloads can't bypass the blocklist."""
    # Fullwidth "chemtrails"
    fullwidth = "ｃｈｅｍｔｒａｉｌｓ are real"
    assert contrarian_safety.check(fullwidth).safe is False
    # Zero-width joiner splitting "chemtrails"
    zw_split = "chem\u200btrails are real"
    assert contrarian_safety.check(zw_split).safe is False
    # Non-breaking space between words
    nbsp = "vaccines\u00a0cause autism"
    assert contrarian_safety.check(nbsp).safe is False


# ─── Generator-level contrarian threading ───
@pytest.mark.asyncio
async def test_carousel_outline_receives_contrarian_flag(monkeypatch):
    """Audit fix #4: carousel._llm_outline must get the flag so slide
    titles carry the contrarian framing, not just the caption."""
    from src.content.generators import carousel as carousel_mod

    captured: dict = {}

    async def fake_outline(cfg_, trend, n, *, contrarian=False):
        captured["contrarian"] = contrarian
        return [
            {"kind": "hook", "title": "t1", "body": "b1", "index": 1},
            {"kind": "content", "title": "t2", "body": "b2", "index": 2},
            {"kind": "cta", "title": "t3", "body": "b3", "index": 3},
        ]

    async def fake_render(html, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpeg")
        return out

    monkeypatch.setattr(carousel_mod, "_llm_outline", fake_outline)
    monkeypatch.setattr(carousel_mod, "render_html_to_png", fake_render)
    monkeypatch.setattr(carousel_mod, "apply_lut_image", lambda p, c: p)

    cfg = _mkcfg()
    await carousel_mod.generate(cfg, "trend", slides=3, contrarian=True)
    assert captured["contrarian"] is True


def test_carousel_outline_system_prompt_includes_contrarian_block():
    """The system prompt for contrarian outlines should carry the flag."""
    # We can't easily inspect the system prompt without running the LLM,
    # so verify via string match on the function body if possible. Trust
    # the fake_outline route test above + the caption prompt test below
    # for direct semantic coverage.
    import src.content.generators.carousel as carousel_mod
    import inspect
    src = inspect.getsource(carousel_mod._llm_outline)
    assert "CONTRARIAN MODE" in src


def test_meme_fill_text_accepts_contrarian_flag():
    from src.content.generators import meme as meme_mod
    import inspect
    src = inspect.getsource(meme_mod._fill_text)
    assert "contrarian" in src.lower()
    assert "CONTRARIAN MEME" in src


def test_quote_card_llm_quote_accepts_contrarian_flag():
    from src.content.generators import quote_card as qc_mod
    import inspect
    src = inspect.getsource(qc_mod._llm_quote)
    assert "contrarian" in src.lower()
    assert "CONTRARIAN QUOTE" in src


def test_reel_stock_script_accepts_contrarian_flag():
    from src.content.generators import reel_stock as reel_mod
    import inspect
    src = inspect.getsource(reel_mod._script)
    assert "contrarian" in src.lower()
    assert "CONTRARIAN FRAMING" in src


# ─── Critic rubric ───
def test_rubric_adds_claim_defensible_only_when_contrarian():
    base = critic._rubric(threshold=0.65, contrarian=False)
    with_c = critic._rubric(threshold=0.65, contrarian=True)
    assert "claim_defensible" not in base
    assert "claim_defensible" in with_c
    assert "CONTRARIAN MODE" in with_c


# ─── Pipeline dice roll ───
@pytest.mark.asyncio
async def test_pipeline_forces_contrarian_via_override(tmp_db, monkeypatch):
    """contrarian_override=True must bypass the random dice."""
    cfg = _mkcfg(contrarian=cfg_mod.ContrarianConfig(enabled=False, frequency=0.0))
    captured_flags: dict = {}

    async def fake_dispatch(format_name, cfg_, trend_context, *, contrarian=False):
        from src.content.generators.base import GeneratedContent
        return GeneratedContent(format=format_name, media_paths=["/x.jpg"], meta={"stub": True})

    async def fake_candidates(cfg_, format_name, content, *, n, knowledge=None, contrarian=False):
        captured_flags["candidates_contrarian"] = contrarian
        return [{
            "caption": "unpopular opinion: rest days are overrated",
            "hashtags": [], "visible_text": "", "image_path": None,
        }]

    async def fake_rank(cfg_, *, format_name, candidates, recent_captions, knowledge=None, contrarian=False):
        captured_flags["rank_contrarian"] = contrarian
        return [{**c, "critique": {"overall": 0.8, "verdict": "approve", "reasons": "ok"}}
                for c in candidates]

    from src.content import dedup
    monkeypatch.setattr(pipe, "_dispatch", fake_dispatch)
    monkeypatch.setattr(pipe, "_build_caption_candidates", fake_candidates)
    monkeypatch.setattr(critic, "rank_candidates", fake_rank)
    monkeypatch.setattr(dedup, "compute_phash", lambda p: "abc")
    monkeypatch.setattr(dedup, "is_duplicate", lambda h, t: (False, None))

    cid = await pipe.generate_one(cfg, format_override="meme", contrarian_override=True)
    assert cid is not None
    assert captured_flags["candidates_contrarian"] is True
    assert captured_flags["rank_contrarian"] is True

    # Meta stamped the contrarian flag so retro learning can slice it
    row = db.content_get(cid)
    assert row["meta"]["contrarian_mode"] is True


@pytest.mark.asyncio
async def test_pipeline_skips_contrarian_when_disabled(tmp_db, monkeypatch):
    cfg = _mkcfg()  # enabled=False, frequency=0.15
    captured: dict = {"contrarian": None}

    async def fake_dispatch(format_name, cfg_, trend_context, *, contrarian=False):
        from src.content.generators.base import GeneratedContent
        return GeneratedContent(format=format_name, media_paths=["/x.jpg"], meta={})

    async def fake_candidates(cfg_, format_name, content, *, n, knowledge=None, contrarian=False):
        captured["contrarian"] = contrarian
        return [{"caption": "fine", "hashtags": [], "visible_text": "", "image_path": None}]

    async def fake_rank(cfg_, **k):
        return [{**c, "critique": {"overall": 0.8, "verdict": "approve", "reasons": "ok"}}
                for c in k["candidates"]]

    from src.content import dedup
    monkeypatch.setattr(pipe, "_dispatch", fake_dispatch)
    monkeypatch.setattr(pipe, "_build_caption_candidates", fake_candidates)
    monkeypatch.setattr(critic, "rank_candidates", fake_rank)
    monkeypatch.setattr(dedup, "compute_phash", lambda p: "abc")
    monkeypatch.setattr(dedup, "is_duplicate", lambda h, t: (False, None))

    # Even with the random dice, enabled=False means we never fire
    for _ in range(20):
        await pipe.generate_one(cfg, format_override="meme")

    assert captured["contrarian"] is False


@pytest.mark.asyncio
async def test_pipeline_drops_unsafe_contrarian_candidates(tmp_db, monkeypatch):
    """When contrarian mode produces a candidate matching the hard
    blocklist, it must be dropped. If ALL candidates are unsafe, the
    cycle regenerates (and eventually gives up)."""
    cfg = _mkcfg(contrarian=cfg_mod.ContrarianConfig(enabled=True))

    async def fake_dispatch(format_name, cfg_, trend_context, *, contrarian=False):
        from src.content.generators.base import GeneratedContent
        return GeneratedContent(format=format_name, media_paths=["/x.jpg"], meta={})

    async def fake_candidates(cfg_, format_name, content, *, n, knowledge=None, contrarian=False):
        # Three candidates — two toxic, one clean
        return [
            {"caption": "chemtrails are real, look up", "hashtags": [], "visible_text": "", "image_path": None},
            {"caption": "moon landing was faked duh", "hashtags": [], "visible_text": "", "image_path": None},
            {"caption": "most coaches say rest days. I say train fasted.", "hashtags": [], "visible_text": "", "image_path": None},
        ]

    ranked_counts: list[int] = []

    async def fake_rank(cfg_, *, format_name, candidates, recent_captions, knowledge=None, contrarian=False):
        ranked_counts.append(len(candidates))
        return [{**c, "critique": {"overall": 0.8, "verdict": "approve", "reasons": "ok"}}
                for c in candidates]

    from src.content import dedup
    monkeypatch.setattr(pipe, "_dispatch", fake_dispatch)
    monkeypatch.setattr(pipe, "_build_caption_candidates", fake_candidates)
    monkeypatch.setattr(critic, "rank_candidates", fake_rank)
    monkeypatch.setattr(dedup, "compute_phash", lambda p: "abc")
    monkeypatch.setattr(dedup, "is_duplicate", lambda h, t: (False, None))

    cid = await pipe.generate_one(cfg, format_override="meme", contrarian_override=True)
    assert cid is not None
    # The critic saw ONE candidate — the two toxic ones were dropped pre-rank
    assert ranked_counts and ranked_counts[0] == 1


@pytest.mark.asyncio
async def test_pipeline_regens_when_all_candidates_unsafe(tmp_db, monkeypatch):
    cfg = _mkcfg(
        contrarian=cfg_mod.ContrarianConfig(enabled=True),
        safety=cfg_mod.Safety(critic_max_regens=1, require_review=False),
    )

    async def fake_dispatch(format_name, cfg_, trend_context, *, contrarian=False):
        from src.content.generators.base import GeneratedContent
        return GeneratedContent(format=format_name, media_paths=["/x.jpg"], meta={})

    async def unsafe_only(cfg_, format_name, content, *, n, knowledge=None, contrarian=False):
        return [
            {"caption": "chemtrails are real", "hashtags": [], "visible_text": "", "image_path": None},
            {"caption": "starve yourself for a week", "hashtags": [], "visible_text": "", "image_path": None},
        ]

    rank_calls = {"n": 0}

    async def fake_rank(cfg_, **k):
        rank_calls["n"] += 1
        return []  # irrelevant — should never be called since all candidates dropped

    from src.content import dedup
    monkeypatch.setattr(pipe, "_dispatch", fake_dispatch)
    monkeypatch.setattr(pipe, "_build_caption_candidates", unsafe_only)
    monkeypatch.setattr(critic, "rank_candidates", fake_rank)
    monkeypatch.setattr(dedup, "compute_phash", lambda p: "abc")
    monkeypatch.setattr(dedup, "is_duplicate", lambda h, t: (False, None))

    cid = await pipe.generate_one(cfg, format_override="meme", contrarian_override=True)
    assert cid is None
    # critic was never reached because every candidate was dropped
    assert rank_calls["n"] == 0
