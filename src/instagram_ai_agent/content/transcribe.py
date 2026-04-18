"""Unified word-level transcription.

Prefers WhisperX (more accurate forced alignment via wav2vec2) when the
package is importable. Falls back to ``faster-whisper`` with
``word_timestamps=True`` which we already ship.

One public surface: :func:`transcribe_words` returns a list of
:class:`Word` dataclasses. Callers pick an emitter (SRT chunked or ASS
karaoke) separately — see :mod:`src.content.captions_render`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


def _whisperx_available() -> bool:
    """Cheap probe — module import only, no model load."""
    if os.environ.get("WHISPERX_DISABLE") == "1":
        return False
    try:
        import whisperx  # noqa: F401
        return True
    except Exception:
        return False


def transcribe_words(audio: Path, *, prefer_whisperx: bool = True, model_size: str | None = None) -> list[Word]:
    """Return a flat list of words with accurate timestamps.

    ``model_size`` defaults to the ``WHISPER_MODEL`` env var or ``"tiny.en"``.
    """
    model = model_size or os.environ.get("WHISPER_MODEL", "tiny.en")

    if prefer_whisperx and _whisperx_available():
        try:
            return _whisperx_transcribe(audio, model=model)
        except Exception as e:
            log.warning("WhisperX failed, falling back to faster-whisper: %s", e)

    return _faster_whisper_transcribe(audio, model=model)


# ───── faster-whisper backend ─────
def _faster_whisper_transcribe(audio: Path, *, model: str) -> list[Word]:
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = "int8" if device == "cpu" else "float16"
    m = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _info = m.transcribe(
        str(audio),
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            if w.start is None or w.end is None:
                continue
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(w.start), end=float(w.end)))
    return words


# ───── WhisperX backend ─────
def _whisperx_transcribe(audio: Path, *, model: str) -> list[Word]:
    import whisperx

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = "int8" if device == "cpu" else "float16"
    batch_size = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))

    # 1. Base transcription
    whisper_model = whisperx.load_model(model, device, compute_type=compute_type)
    audio_arr = whisperx.load_audio(str(audio))
    result = whisper_model.transcribe(audio_arr, batch_size=batch_size)

    # 2. Forced alignment (wav2vec2) — the reason we're here at all
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", "en"), device=device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio_arr,
        device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words") or []:
            start = w.get("start")
            end = w.get("end")
            text = (w.get("word") or "").strip()
            if start is None or end is None or not text:
                continue
            words.append(Word(text=text, start=float(start), end=float(end)))
    return words
