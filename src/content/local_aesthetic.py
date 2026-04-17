"""Local aesthetic / quality scoring.

Cheap pre-filter that runs before the vision LLM. Every candidate image
gets a 0-1 score from one or more CPU/GPU-local scorers; the top-K then
go to the more expensive vision-LLM pass.

Design: `BaseLocalScorer` is the stable interface. We ship one default
implementation (`AestheticsPredictorScorer`, MIT-licensed LAION port via
the `simple-aesthetics-predictor` pip package) and an
`EnsembleLocalScorer` that averages N scorers. Users who want NIMA or
PickScore drop their own `BaseLocalScorer` subclass in and pass it at
call time — no core code changes needed.

Every import is lazy; the module imports cleanly with zero extras
installed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LocalScore:
    """Per-candidate output from a local scorer (or an ensemble)."""

    score: float                  # 0.0-1.0 normalised
    raw: dict[str, float] = field(default_factory=dict)  # per-scorer raw values
    model_used: str = ""          # "ensemble:..." or single scorer name


class BaseLocalScorer(Protocol):
    """Lightweight protocol — any object with ``name`` + ``is_available`` +
    async ``score_one(path) -> float`` can plug in."""

    name: str

    def is_available(self) -> bool: ...
    async def score_one(self, image_path: Path) -> float | None: ...


# ─── simple-aesthetics-predictor (LAION CLIP-based, MIT) ───
class AestheticsPredictorScorer:
    """Wraps `simple-aesthetics-predictor` (CLIP-based, MIT).

    Raw output is on a 1-10 scale; we normalise to 0-1 by `(raw - 1) / 9`
    and clamp.
    """

    name = "laion_aesthetic"
    _predictor = None

    def is_available(self) -> bool:
        try:
            import aesthetics_predictor  # noqa: F401
            return True
        except Exception:
            try:
                import simple_aesthetics_predictor  # noqa: F401
                return True
            except Exception:
                return False

    def _load(self):
        if self.__class__._predictor is not None:
            return self.__class__._predictor
        # Package name has been published under both names; try both.
        try:
            from aesthetics_predictor import AestheticsPredictorV1 as _APV1
        except Exception:
            from simple_aesthetics_predictor import AestheticsPredictorV1 as _APV1  # type: ignore
        try:
            from transformers import CLIPProcessor
        except Exception as e:
            raise RuntimeError(f"transformers not installed: {e}") from e

        model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
        self.__class__._predictor = {
            "model": _APV1.from_pretrained(model_id).eval(),
            "processor": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
        }
        return self.__class__._predictor

    async def score_one(self, image_path: Path) -> float | None:
        def _run() -> float:
            import torch
            from PIL import Image

            bundle = self._load()
            model = bundle["model"]
            processor = bundle["processor"]
            with Image.open(image_path).convert("RGB") as img:
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                # Prediction is a scalar on a 1-10 scale
                val = float(out.logits.squeeze().item())
            normalised = max(0.0, min(1.0, (val - 1.0) / 9.0))
            return normalised

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            log.warning("aesthetic scorer failed on %s: %s", image_path.name, e)
            return None


# ─── PyIQA-based quality scorer (OPT-IN — LICENCE WARNING) ───
class PyIQAQualityScorer:
    """Wraps ``pyiqa`` (**PolyForm Noncommercial** — NOT commercial-safe).

    Provides CLIP-IQA / NIMA / MUSIQ quality metrics. Useful for personal
    or research use cases; **do NOT ship this on a monetised Instagram
    page**. It is deliberately NOT part of :func:`default_scorers` and
    NOT pulled by the ``[aesthetic]`` pip extra. Users who accept the
    licence constraints can install it via the separate ``[aesthetic-nc]``
    extra and pass an instance of this class explicitly into an
    :class:`EnsembleLocalScorer`.

    Complementary to the LAION aesthetic predictor: LAION captures
    "pretty", PyIQA captures "technical quality".
    """

    name = "pyiqa_quality"
    _model: Any = None

    def __init__(self, metric: str = "clipiqa"):
        self.metric = metric

    def is_available(self) -> bool:
        try:
            import pyiqa  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self.__class__._model is not None:
            return self.__class__._model
        import pyiqa
        self.__class__._model = pyiqa.create_metric(self.metric, as_loss=False)
        return self.__class__._model

    async def score_one(self, image_path: Path) -> float | None:
        def _run() -> float:
            metric = self._load()
            val = float(metric(str(image_path)).item())
            return max(0.0, min(1.0, val))

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            log.warning("pyiqa scorer failed on %s: %s", image_path.name, e)
            return None


# ─── Ensemble ───
class EnsembleLocalScorer:
    """Averages N scorers. Ignores missing ones."""

    def __init__(self, scorers: list[BaseLocalScorer] | None = None):
        self.scorers = scorers or default_scorers()

    def is_available(self) -> bool:
        return any(s.is_available() for s in self.scorers)

    async def score(self, image_path: Path) -> LocalScore | None:
        active = [s for s in self.scorers if s.is_available()]
        if not active:
            return None

        raw: dict[str, float] = {}
        results = await asyncio.gather(
            *[s.score_one(image_path) for s in active], return_exceptions=True,
        )
        for s, res in zip(active, results, strict=True):
            if isinstance(res, float) and 0.0 <= res <= 1.0:
                raw[s.name] = res
        if not raw:
            return None
        mean = sum(raw.values()) / len(raw)
        return LocalScore(
            score=mean,
            raw=raw,
            model_used="ensemble:" + "+".join(sorted(raw)),
        )


def default_scorers() -> list[BaseLocalScorer]:
    """Commercial-safe default ensemble.

    Only ``AestheticsPredictorScorer`` (LAION, MIT) is included. PyIQA is
    PolyForm Noncommercial and is NOT part of this list. Users on a
    personal / research setup who accept the NC terms can build their own
    ensemble explicitly::

        from src.content.local_aesthetic import (
            AestheticsPredictorScorer, PyIQAQualityScorer,
            EnsembleLocalScorer,
        )
        ensemble = EnsembleLocalScorer([
            AestheticsPredictorScorer(),
            PyIQAQualityScorer(metric="nima"),
        ])
    """
    return [AestheticsPredictorScorer()]


# Cached module-level singleton for the default ensemble
_default_ensemble: EnsembleLocalScorer | None = None


def get_ensemble() -> EnsembleLocalScorer:
    global _default_ensemble
    if _default_ensemble is None:
        _default_ensemble = EnsembleLocalScorer()
    return _default_ensemble


async def score_local(image_path: str | Path) -> LocalScore | None:
    """Convenience: score a path with the default ensemble. Returns None
    when no local backend is installed."""
    ens = get_ensemble()
    if not ens.is_available():
        return None
    return await ens.score(Path(image_path))


def reset_cache() -> None:
    """Drop cached models/ensembles — used by tests."""
    global _default_ensemble
    _default_ensemble = None
    AestheticsPredictorScorer._predictor = None
    PyIQAQualityScorer._model = None
