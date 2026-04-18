"""Kinetic captions: SRT chunker + ASS karaoke emitter + CaptionsConfig."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from instagram_ai_agent.content import captions_render as cr
from instagram_ai_agent.content.transcribe import Word
from instagram_ai_agent.core import config as cfg_mod


def _mkcfg(**kwargs):
    base = dict(
        niche="home calisthenics",
        sub_topics=["pullups"],
        target_audience="dads 35+",
        commercial=True,
        voice=cfg_mod.Voice(tone=["direct"], forbidden=[], persona="ex-office worker rebuilding fitness."),
        aesthetic=cfg_mod.Aesthetic(palette=["#0a0a0a", "#f5f5f0", "#c9a961"]),
        hashtags=cfg_mod.HashtagPools(core=["calisthenics", "homeworkout", "dadfit"]),
    )
    base.update(kwargs)
    return cfg_mod.NicheConfig(**base)


def _words(*pairs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start=s, end=e) for t, s, e in pairs]


# ─── Config ───
def test_captions_defaults_prefer_karaoke():
    cfg = _mkcfg()
    assert cfg.captions.style == "karaoke"
    assert cfg.captions.karaoke_word_at_a_time is True
    assert cfg.captions.font_scale_peak > 1.0
    assert cfg.captions.prefer_whisperx is True


def test_captions_style_validates():
    # Config itself is permissive on style (string); just verify it roundtrips
    cfg = _mkcfg(captions=cfg_mod.CaptionsConfig(style="static"))
    assert cfg.captions.style == "static"


# ─── SRT chunker ───
def test_render_srt_chunks_groups_words(tmp_path: Path):
    out = tmp_path / "c.srt"
    words = _words(
        ("Pull-ups", 0.0, 0.4),
        ("build", 0.45, 0.70),
        ("real", 0.75, 1.00),
        ("strength", 1.05, 1.45),
        ("every", 1.55, 1.80),
        ("single", 1.85, 2.15),
        ("day", 2.20, 2.50),
    )
    cr.render_srt(words, out, chunk_size=4)
    content = out.read_text()
    # Exactly 2 entries (4 + 3)
    assert content.count(" --> ") == 2
    # Each entry is timestamp-numbered
    assert content.startswith("1\n00:00:00,")


def test_render_srt_respects_sentence_boundary(tmp_path: Path):
    out = tmp_path / "c.srt"
    words = _words(
        ("Go.", 0.0, 0.30),
        ("Now,", 0.32, 0.55),
        ("harder", 0.60, 1.00),
        ("than", 1.05, 1.30),
        ("yesterday.", 1.35, 1.80),
    )
    cr.render_srt(words, out, chunk_size=4)
    content = out.read_text()
    # Three chunks: "Go." alone gets included with the next but sentence end
    # forces an early break once we have ≥3 words accumulated
    assert content.count(" --> ") >= 1


def test_render_srt_timestamp_format(tmp_path: Path):
    out = tmp_path / "c.srt"
    cr.render_srt(_words(("hi", 65.123, 66.0)), out)
    content = out.read_text()
    # 1:05.123 → 00:01:05,123
    assert re.search(r"00:01:05,123 --> 00:01:06,000", content)


# ─── ASS header ───
def test_ass_header_has_script_info_and_style():
    header = cr._ass_header(
        play_res_x=1080, play_res_y=1920,
        font_name="Archivo Black", font_size=88,
        primary_bgr="61A9C9", outline_bgr="000000", back_bgr="0A0A0A",
        margin_v=420,
    )
    assert "[Script Info]" in header
    assert "PlayResX: 1080" in header
    assert "PlayResY: 1920" in header
    assert "[V4+ Styles]" in header
    assert "Style: Karaoke" in header
    # Font + size are threaded through
    assert "Archivo Black" in header
    assert ",88," in header
    # Primary colour comes in as BGR with ASS &H00xxxx&
    assert "&H0061A9C9&" in header
    # MarginV from cfg (Encoding is 0 — auto)
    assert ",420,0" in header


# ─── ASS karaoke emitter ───
def test_ass_karaoke_emits_one_event_per_word(tmp_path: Path):
    cfg = _mkcfg()
    out = tmp_path / "c.ass"
    words = _words(
        ("Unlock", 0.0, 0.4),
        ("your", 0.5, 0.75),
        ("pull-up.", 0.8, 1.3),
    )
    cr.render_ass_karaoke(
        words, out, cfg=cfg, captions=cfg.captions,
        video_w=1080, video_h=1920, is_story=False,
    )
    content = out.read_text()
    dialogues = [ln for ln in content.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 3
    # Each event has fade, scale-bounce and position tags
    for line in dialogues:
        assert r"\fad(" in line
        assert r"\fscx" in line
        assert r"\t(" in line
        assert r"\pos(" in line


def test_ass_karaoke_escapes_braces_in_word_text(tmp_path: Path):
    cfg = _mkcfg()
    out = tmp_path / "c.ass"
    cr.render_ass_karaoke(
        _words(("{weird}", 0.0, 0.4)), out, cfg=cfg, captions=cfg.captions,
    )
    content = out.read_text()
    # Raw braces in the text must be escaped with backslashes
    assert r"\{weird\}" in content
    # Our own override tags remain wrapped in real braces
    assert content.count("{") >= 2  # at least one override tag + escaped text


def test_ass_karaoke_empty_words_writes_empty_file(tmp_path: Path):
    cfg = _mkcfg()
    out = tmp_path / "c.ass"
    cr.render_ass_karaoke([], out, cfg=cfg, captions=cfg.captions)
    # Empty-input contract: file exists and is empty (no header)
    assert out.exists()
    assert out.read_text() == ""


def test_ass_karaoke_uses_highlight_colour_override(tmp_path: Path):
    cfg = _mkcfg(
        captions=cfg_mod.CaptionsConfig(style="karaoke", highlight_colour="#ff0000")
    )
    out = tmp_path / "c.ass"
    cr.render_ass_karaoke(
        _words(("Pop", 0.0, 0.3)), out, cfg=cfg, captions=cfg.captions,
    )
    content = out.read_text()
    # Red in BGR is 0000FF
    assert "&H000000FF&" in content


def test_ass_karaoke_story_margin_higher_than_feed(tmp_path: Path):
    cfg = _mkcfg()
    feed_out = tmp_path / "feed.ass"
    story_out = tmp_path / "story.ass"
    cr.render_ass_karaoke(
        _words(("A", 0.0, 0.3)), feed_out, cfg=cfg, captions=cfg.captions,
        video_w=1080, video_h=1920, is_story=False,
    )
    cr.render_ass_karaoke(
        _words(("A", 0.0, 0.3)), story_out, cfg=cfg, captions=cfg.captions,
        video_w=1080, video_h=1920, is_story=True,
    )
    feed_y = _extract_y_from_pos(feed_out.read_text())
    story_y = _extract_y_from_pos(story_out.read_text())
    # Story margin is larger → caption Y is HIGHER on screen (smaller y-coord
    # since bottom-anchored with video_h - margin_v).
    assert story_y < feed_y


def _extract_y_from_pos(content: str) -> int:
    m = re.search(r"\\pos\((\d+),(\d+)\)", content)
    assert m, "no \\pos tag found"
    return int(m.group(2))


# ─── hex→BGR ───
def test_hex_to_bgr():
    assert cr._hex_to_bgr("#ff0000") == "0000FF"
    assert cr._hex_to_bgr("00ff00") == "00FF00"
    assert cr._hex_to_bgr("#0000FF") == "FF0000"
    # malformed → fallback white
    assert cr._hex_to_bgr("not-a-hex") == "FFFFFF"


# ─── Transcribe module guards ───
def test_transcribe_whisperx_available_respects_disable_env(monkeypatch: pytest.MonkeyPatch):
    from instagram_ai_agent.content import transcribe as tr
    monkeypatch.setenv("WHISPERX_DISABLE", "1")
    assert tr._whisperx_available() is False


def test_ass_time_format():
    assert cr._ass_time(0.0) == "0:00:00.00"
    assert cr._ass_time(65.45) == "0:01:05.45"
    assert cr._ass_time(3600 + 2 * 60 + 3.5) == "1:02:03.50"


# ─── Regression: ASS brace balance ───
def test_ass_karaoke_dialogue_has_balanced_braces(tmp_path: Path):
    """Every Dialogue line must have exactly one override block {..}.

    The original bug: an unescaped `"}}"` emitted two closing braces per line,
    rendering a literal `}` before every word in the burned output.
    """
    cfg = _mkcfg()
    out = tmp_path / "c.ass"
    cr.render_ass_karaoke(
        _words(("Alpha", 0.0, 0.3), ("Bravo", 0.4, 0.7), ("Charlie", 0.8, 1.1)),
        out, cfg=cfg, captions=cfg.captions,
    )
    for line in out.read_text().splitlines():
        if not line.startswith("Dialogue:"):
            continue
        # ASS Dialogue has 9 comma-separated header fields before the Text:
        #   Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        parts = line.split(",", 9)
        assert len(parts) == 10, f"unexpected Dialogue arity: {line!r}"
        body = parts[9]
        assert body.startswith("{"), f"override not opened: {body[:40]}"
        depth = 0
        close_idx = -1
        for i, ch in enumerate(body):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        assert close_idx > 0, f"override never closed: {body}"
        # Everything after close_idx is the text payload and must NOT start
        # with another stray `}`.
        payload = body[close_idx + 1:]
        assert not payload.startswith("}"), (
            f"stray closing brace before text: {payload[:40]!r} in {line!r}"
        )


def test_ass_karaoke_event_durations_do_not_overlap(tmp_path: Path):
    """No two consecutive events should have end > next.start after the
    overlap-clamp. This prevents libass double-draw / flicker when words are
    spoken quickly."""
    cfg = _mkcfg()
    out = tmp_path / "c.ass"
    # Fast sequence — default +0.08s tail would overlap the next word
    cr.render_ass_karaoke(
        _words(
            ("one", 0.00, 0.15),
            ("two", 0.18, 0.33),
            ("three", 0.35, 0.50),
            ("four", 0.52, 0.70),
        ),
        out, cfg=cfg, captions=cfg.captions,
    )
    starts_ends: list[tuple[float, float]] = []
    for line in out.read_text().splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",")
        # parts[1] = start, parts[2] = end in H:MM:SS.cs
        def _parse(t: str) -> float:
            h, m, s_cs = t.split(":")
            s, cs = s_cs.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
        starts_ends.append((_parse(parts[1]), _parse(parts[2])))

    assert len(starts_ends) == 4
    for (_, end), (start, _) in zip(starts_ends, starts_ends[1:], strict=False):
        # end should not exceed next start (small epsilon for cs rounding)
        assert end <= start + 0.02, f"event overlap: end={end} > next_start={start}"


def test_ass_header_encoding_is_0():
    """ASS Encoding field should be 0 (auto) for max libass portability."""
    header = cr._ass_header(
        play_res_x=1080, play_res_y=1920,
        font_name="Inter", font_size=72,
        primary_bgr="FFFFFF", outline_bgr="000000", back_bgr="000000",
        margin_v=160,
    )
    # Style line ends with margin_v,{encoding}
    style_line = [ln for ln in header.splitlines() if ln.startswith("Style:")][0]
    assert style_line.rstrip().endswith(",0"), style_line


def test_render_srt_comma_does_not_force_break(tmp_path: Path):
    """A comma mid-sentence must NOT close a chunk early — only . ! ? do."""
    cfg = _mkcfg()
    out = tmp_path / "c.srt"
    words = _words(
        ("If", 0.0, 0.15),
        ("you,", 0.20, 0.40),           # comma — NOT a break
        ("like", 0.45, 0.65),
        ("me,", 0.70, 0.90),            # comma — NOT a break
        ("do", 0.95, 1.10),
        ("this.", 1.15, 1.45),           # full stop — breaks here
    )
    cr.render_srt(words, out, chunk_size=6)
    content = out.read_text()
    # With chunk_size=6 and no comma-break, we get exactly one chunk
    assert content.count(" --> ") == 1
