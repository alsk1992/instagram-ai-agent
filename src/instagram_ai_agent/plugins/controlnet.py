"""ControlNet reference-image conditioning for the ComfyUI workflow.

Three modes — pose, depth, canny — let a user point at a brand
reference image (athlete stance, product shot, edge sketch) and have
every AI-generated image respect its composition. Wired in at
``comfyui.generate()`` time, same pattern as the LoRA injection.

Commercial safety:
  * Default preprocessors are ALL commercial-safe (DWPose Apache-2.0,
    Depth-Anything v2 Apache-2.0, OpenCV Canny Apache-2.0).
  * ``preprocessor_override`` lets power users pick a custom node class;
    a naive OpenPose override is blocked at config-load by
    ``NicheConfig._controlnet_commercial_gate``.
  * We do NOT ship or download any model weights — the user points the
    ComfyUI server at its own ControlNet model directory.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from instagram_ai_agent.core.config import (
    CONTROLNET_DIR,
    CONTROLNET_MODELS_DIR,
    NICHE_PATH,
    NicheConfig,
    save_niche,
)
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


# ─── Safe preprocessor defaults ───
# Each tuple: (ComfyUI node class_type, licence label)
_DEFAULT_PREPROCESSORS: dict[str, tuple[str, str]] = {
    "pose":  ("DWPreprocessor",               "Apache-2.0"),
    "depth": ("DepthAnythingV2Preprocessor",  "Apache-2.0"),
    "canny": ("CannyEdgePreprocessor",        "Apache-2.0"),
}

# Required preprocessor inputs beyond the `image` pipe. Pulled from
# comfyui_controlnet_aux's node declarations: nodes without defaults
# must receive these values at prompt-submission time or ComfyUI's
# validator rejects the job.
_PREPROCESSOR_REQUIRED_INPUTS: dict[str, dict[str, object]] = {
    "DWPreprocessor": {
        "detect_hand":       "enable",
        "detect_body":       "enable",
        "detect_face":       "enable",
        "resolution":        512,
        "bbox_detector":     "yolox_l.onnx",
        "pose_estimator":    "dw-ll_ucoco_384.onnx",
    },
    "DepthAnythingV2Preprocessor": {
        "ckpt_name":         "depth_anything_v2_vitl.pth",
        "resolution":        512,
    },
    "CannyEdgePreprocessor": {
        "low_threshold":     100,
        "high_threshold":    200,
        "resolution":        512,
    },
    # MiDaS fallback for installs that don't have Depth-Anything.
    "MiDasDepthMapPreprocessor": {
        "a":                 6.283185307179586,  # 2π — package default
        "bg_threshold":      0.1,
        "resolution":        512,
    },
    # Canny alt node name on some builds
    "Canny": {
        "low_threshold":     0.4,
        "high_threshold":    0.8,
    },
}

# Substrings that mark a preprocessor as non-commercial. Checked after
# lowercasing — catches OpenPose/DWopenPoseHybrid/AnimalPose etc.
# DensePose (Facebook AI) is CC-BY-NC-4.0 — also non-commercial, also
# blocked. DWPose does NOT trigger this because its substring is "dwpose"
# not "openpose"; callers who override with DWPose-named nodes pass
# through cleanly.
_COMMERCIAL_BLOCK_SUBSTRINGS: tuple[str, ...] = (
    "openpose",   # CMU Academic non-commercial
    "animalpose", # CMU derivative, same licence
    "densepose",  # Facebook CC-BY-NC-4.0
)


def _is_noncommercial_preprocessor(name: str) -> bool:
    key = (name or "").lower()
    return any(bad in key for bad in _COMMERCIAL_BLOCK_SUBSTRINGS)

_SUPPORTED_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ControlNetMode:
    name: str
    preprocessor: str
    license: str


def mode_for(cfg: NicheConfig) -> ControlNetMode:
    """Return the resolved mode + preprocessor + licence for the current
    config. Honours ``preprocessor_override`` when set."""
    m = cfg.controlnet.mode
    default_preproc, default_licence = _DEFAULT_PREPROCESSORS.get(
        m, _DEFAULT_PREPROCESSORS["pose"],
    )
    chosen = cfg.controlnet.preprocessor_override.strip() or default_preproc
    # When the user overrode, we don't know the licence — label as "user-declared".
    licence = default_licence if chosen == default_preproc else "user-declared"
    return ControlNetMode(name=m, preprocessor=chosen, license=licence)


# ─── CLI-surface helpers ───
def _validate_filename(name: str) -> str:
    n = (name or "").strip()
    if not _NAME_RE.match(n):
        raise ValueError(
            f"Filename {name!r} invalid — use 1–128 chars: letters, digits, ., _ or -, "
            "starting with a letter or digit."
        )
    return n


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` atomically: writes to a sibling temp
    file first then ``os.replace``s into place. Protects against half-
    written files if an interrupted multi-GB copy corrupts the dest
    (common with ControlNet .safetensors at 1+ GB)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        shutil.copy2(src, staging)
        os.replace(staging, dest)
    except BaseException:
        # Best-effort cleanup if copy died mid-write
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def set_reference(
    source: Path,
    *,
    mode: str,
    cfg: NicheConfig,
    niche_path: Path | None = None,
) -> Path:
    """Copy a reference image into ``data/controlnet/`` and activate
    it in niche.yaml for the given mode. Returns the stored path."""
    if mode not in _DEFAULT_PREPROCESSORS:
        raise ValueError(f"Unknown mode {mode!r}. Pick one of: pose, depth, canny.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Reference image not found: {source}")
    if source.suffix.lower() not in _SUPPORTED_IMG_EXT:
        raise ValueError(
            f"Unsupported image format {source.suffix}. Use JPG, PNG, or WebP."
        )

    CONTROLNET_DIR.mkdir(parents=True, exist_ok=True)
    dest = CONTROLNET_DIR / f"{mode}{source.suffix.lower()}"
    # Remove any prior reference at a different extension (pose.jpg → pose.png)
    for ext in _SUPPORTED_IMG_EXT:
        prior = CONTROLNET_DIR / f"{mode}{ext}"
        if prior.exists() and prior != dest:
            prior.unlink(missing_ok=True)
    _atomic_copy(source, dest)

    # Persist to niche.yaml, honouring the existing model_name if set
    new_cn = cfg.controlnet.model_copy(update={
        "enabled": True,
        "mode": mode,
        "reference_image": dest.name,
    })
    new_cfg = cfg.model_copy(update={"controlnet": new_cn})
    # Re-validate through the root so the commercial gate fires
    new_cfg = NicheConfig.model_validate(new_cfg.model_dump(mode="python"))
    save_niche(new_cfg, path=niche_path or NICHE_PATH)
    log.info("ControlNet reference set — mode=%s ref=%s", mode, dest.name)
    return dest


def clear_reference(
    cfg: NicheConfig, *, niche_path: Path | None = None,
) -> NicheConfig:
    """Disable ControlNet in niche.yaml and remove the reference images
    for every mode from data/controlnet/."""
    for ext in _SUPPORTED_IMG_EXT:
        for mode in _DEFAULT_PREPROCESSORS:
            p = CONTROLNET_DIR / f"{mode}{ext}"
            p.unlink(missing_ok=True)
    from instagram_ai_agent.core.config import ControlNetConfig
    new_cfg = cfg.model_copy(update={"controlnet": ControlNetConfig()})
    save_niche(new_cfg, path=niche_path or NICHE_PATH)
    return new_cfg


def set_model(
    source: Path,
    *,
    cfg: NicheConfig,
    niche_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy a ControlNet .safetensors into data/controlnet_models/ and
    record its filename on niche.yaml's controlnet.model_name."""
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"ControlNet model not found: {source}")
    if source.suffix.lower() != ".safetensors":
        raise ValueError(f"Only .safetensors is supported (got {source.suffix}).")
    name = _validate_filename(source.name)
    CONTROLNET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CONTROLNET_MODELS_DIR / name
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"A ControlNet model named {name!r} already exists at {dest}. "
            "Pass --overwrite to replace it."
        )
    _atomic_copy(source, dest)
    new_cn = cfg.controlnet.model_copy(update={"model_name": dest.name})
    new_cfg = cfg.model_copy(update={"controlnet": new_cn})
    new_cfg = NicheConfig.model_validate(new_cfg.model_dump(mode="python"))
    save_niche(new_cfg, path=niche_path or NICHE_PATH)
    log.info("ControlNet model imported: %s", dest.name)
    return dest


def list_models() -> list[Path]:
    if not CONTROLNET_MODELS_DIR.exists():
        return []
    return sorted(CONTROLNET_MODELS_DIR.glob("*.safetensors"))


# ─── Readiness ───
def is_active(cfg: NicheConfig) -> bool:
    """True when ControlNet is enabled AND the reference image + model
    file both exist on disk."""
    cn = cfg.controlnet
    if not cn.enabled or not cn.reference_image or not cn.model_name:
        return False
    ref = CONTROLNET_DIR / cn.reference_image
    mdl = CONTROLNET_MODELS_DIR / cn.model_name
    return ref.exists() and mdl.exists()


# ─── ComfyUI workflow injection ───
def _find_by_class(workflow: dict, class_types: tuple[str, ...]) -> list[str]:
    return [nid for nid, node in workflow.items() if node.get("class_type") in class_types]


def _fresh_node_id(workflow: dict, start: int = 20) -> str:
    taken = {str(k) for k in workflow.keys()}
    n = start
    while str(n) in taken:
        n += 1
    return str(n)


def inject_into_workflow(workflow: dict, cfg: NicheConfig) -> dict:
    """Insert the LoadImage → Preprocessor → ControlNetLoader →
    ControlNetApplyAdvanced chain and rewire the KSampler's positive
    and negative inputs to read from the ControlNet conditioning pipes.

    No-op when:
      * ``cfg.controlnet.enabled=False``
      * reference image or model file is missing on disk
      * the workflow has no KSampler (sanity guard)
      * the workflow already contains a ControlNetApplyAdvanced (caller's
        hand-authored graph — we respect it)

    Commercial safety: if the resolved preprocessor is in the blocklist
    AND commercial=True, refuse and log.
    """
    if not is_active(cfg):
        return workflow

    resolved = mode_for(cfg)
    if cfg.commercial and _is_noncommercial_preprocessor(resolved.preprocessor):
        log.warning(
            "ControlNet: refusing injection — preprocessor %r is non-commercial "
            "under commercial=True. Clear preprocessor_override to use the safe "
            "default.", resolved.preprocessor,
        )
        return workflow

    samplers = _find_by_class(workflow, ("KSampler", "KSamplerAdvanced"))
    if not samplers:
        log.warning("ControlNet: workflow has no KSampler — skipping injection.")
        return workflow
    # Use the FIRST sampler's positive/negative refs as the source for
    # ControlNetApplyAdvanced's inputs — in multi-sampler pipelines
    # (SDXL base + refiner) both samplers typically share the same
    # conditioning pair, so one wrap node serves both.
    first_sampler = workflow[samplers[0]]
    positive_ref = first_sampler.get("inputs", {}).get("positive")
    negative_ref = first_sampler.get("inputs", {}).get("negative")
    if not (isinstance(positive_ref, list) and isinstance(negative_ref, list)):
        log.warning("ControlNet: sampler positive/negative refs malformed — skipping.")
        return workflow

    # If the user's workflow already includes ControlNet, leave it alone
    existing_apply = _find_by_class(
        workflow, ("ControlNetApplyAdvanced", "ControlNetApply", "ApplyFluxControlNet"),
    )
    if existing_apply:
        log.info(
            "ControlNet: workflow already has %s — leaving wiring untouched.",
            workflow[existing_apply[0]].get("class_type"),
        )
        return workflow

    ref_path = CONTROLNET_DIR / cfg.controlnet.reference_image
    model_name = cfg.controlnet.model_name

    # Allocate fresh node IDs — start at 20 to leave plenty of room for
    # existing SDXL / LoRA nodes (10..15 range).
    load_id = _fresh_node_id(workflow, start=20)
    prep_id = _fresh_node_id(workflow, start=int(load_id) + 1)
    cn_load_id = _fresh_node_id(workflow, start=int(prep_id) + 1)
    cn_apply_id = _fresh_node_id(workflow, start=int(cn_load_id) + 1)

    # 1. LoadImage — ComfyUI resolves bare filenames against its input dir.
    #    We use the absolute path so users don't have to copy files into
    #    ComfyUI's input/ directory manually.
    workflow[load_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": str(ref_path)},
    }

    # 2. Preprocessor — mode-specific class_type + its required inputs
    prep_inputs: dict[str, object] = {"image": [load_id, 0]}
    prep_inputs.update(_PREPROCESSOR_REQUIRED_INPUTS.get(resolved.preprocessor, {}))
    workflow[prep_id] = {
        "class_type": resolved.preprocessor,
        "inputs": prep_inputs,
    }

    # 3. ControlNetLoader — resolves model_name against ComfyUI's
    #    models/controlnet/ directory.
    workflow[cn_load_id] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": model_name},
    }

    # 4. ControlNetApplyAdvanced — wraps positive + negative
    workflow[cn_apply_id] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": positive_ref,
            "negative": negative_ref,
            "control_net": [cn_load_id, 0],
            "image": [prep_id, 0],
            "strength": float(cfg.controlnet.strength),
            "start_percent": float(cfg.controlnet.start_percent),
            "end_percent": float(cfg.controlnet.end_percent),
        },
    }

    # 5. Rewire EVERY sampler. Refiner pipelines (SDXL base + refiner)
    # have two samplers in series — both should consume the ControlNet-
    # wrapped conditioning, otherwise half the generation runs with the
    # unguided positive/negative pair and the composition drifts.
    for sid in samplers:
        node_inputs = workflow[sid].get("inputs", {})
        node_inputs["positive"] = [cn_apply_id, 0]
        node_inputs["negative"] = [cn_apply_id, 1]

    log.info(
        "ControlNet: injected mode=%s preproc=%s strength=%.2f (%s), "
        "rewired %d sampler(s)",
        resolved.name, resolved.preprocessor,
        cfg.controlnet.strength, resolved.license, len(samplers),
    )
    return workflow
