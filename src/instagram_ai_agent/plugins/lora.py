"""Brand LoRA plumbing — dataset prep, import, activation.

This module does NOT train LoRAs. Training requires 12+ GB VRAM plus
the full kohya-ss/FluxGym stack (torch, xformers, diffusers, bitsandbytes)
which we refuse to add to the base install. Users train externally (FluxGym,
kohya, RunPod, Replicate) and then ``import`` the resulting .safetensors
file here — at which point every ComfyUI image route automatically
chains the LoRA via LoraLoader.

Commercial-safety: the base-model choice is gated at niche.yaml load time
(``NicheConfig._lora_commercial_gate``). FLUX.1-dev is rejected under
``commercial=True``; FLUX.1-schnell (Apache-2.0) and SDXL (OpenRAIL++-M)
are allowed.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from instagram_ai_agent.core.config import LORA_DATASETS_DIR, LORAS_DIR, NICHE_PATH, NicheConfig, load_niche, save_niche
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)


_SUPPORTED_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_TRIGGER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}$")


@dataclass(frozen=True)
class LoRAInfo:
    name: str
    path: Path
    size_mb: float
    base_model_hint: str | None  # best-effort from safetensors metadata


# ─── Validation helpers ───
def validate_name(name: str) -> str:
    """Reject names with path traversal characters or spaces."""
    n = (name or "").strip()
    if not _NAME_RE.match(n):
        raise ValueError(
            f"LoRA name {name!r} invalid — use 1–64 chars: letters, digits, _ or -, "
            "starting with a letter or digit."
        )
    return n


def validate_trigger(trigger: str) -> str:
    t = (trigger or "").strip()
    if not _TRIGGER_RE.match(t):
        raise ValueError(
            f"Trigger word {trigger!r} invalid — use 1–32 chars: letters, digits, _ or -. "
            "Short unique tokens train best (e.g. 'mascxyz', 'brandv1')."
        )
    return t


def _is_safetensors(path: Path) -> bool:
    """Best-effort detection: safetensors files start with a u64 header
    length followed by JSON metadata. We read the first 8 bytes, then
    the header, and check it parses as JSON."""
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            header_len = struct.unpack("<Q", raw)[0]
            # Sane header size: 100 bytes to 100 MB
            if header_len < 10 or header_len > 100 * 1024 * 1024:
                return False
            header = f.read(header_len)
            json.loads(header)
        return True
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _base_model_hint_from_safetensors(path: Path) -> str | None:
    """Inspect the safetensors metadata header for a base-model hint.

    Kohya-sd-scripts writes ``ss_base_model_version`` into metadata
    and FluxGym writes ``modelspec.architecture``. We surface whatever
    we can find; return None if we can't confidently tell."""
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            header_len = struct.unpack("<Q", raw)[0]
            if header_len < 10 or header_len > 100 * 1024 * 1024:
                return None
            header_bytes = f.read(header_len)
        meta = json.loads(header_bytes).get("__metadata__") or {}
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return None

    # Kohya sd-scripts convention. Probe the stricter marker first
    # ("dev") — a string like "flux_dev_schnell_merged" should NOT be
    # resolved as schnell (the safer licence) when "dev" is also present.
    ss_base = (meta.get("ss_base_model_version") or "").lower()
    if "flux" in ss_base and "dev" in ss_base:
        return "flux-dev"
    if "flux" in ss_base and "schnell" in ss_base:
        return "flux-schnell"
    if "sdxl" in ss_base or "sd_xl" in ss_base:
        return "sdxl"

    # modelspec fallback — same dev-before-schnell ordering.
    arch = (meta.get("modelspec.architecture") or "").lower()
    if "flux" in arch and "dev" in arch:
        return "flux-dev"
    if "flux" in arch:
        return "flux-schnell"
    if "stable-diffusion-xl" in arch or "sdxl" in arch:
        return "sdxl"
    return None


# ─── Dataset prep ───
@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    image_count: int
    captions_written: int


def _discover_images(src: Path) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"Image directory not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Not a directory: {src}")
    images = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMG_EXT
    )
    return images


def _write_caption(dest: Path, trigger: str, auto_caption: str | None) -> None:
    """Write a Kohya-compatible .txt caption file next to the image.

    Trigger word goes FIRST (what the LoRA attaches to), then the auto-
    caption describing everything else in the image (so the LoRA learns
    to attach the trigger to the invariant subject, not the background)."""
    body = (auto_caption or "").strip()
    if body:
        text = f"{trigger}, {body}"
    else:
        text = trigger
    dest.write_text(text, encoding="utf-8")


def _fluxgym_dataset_toml(name: str, trigger: str, image_count: int) -> str:
    """A minimal dataset.toml the user can hand straight to FluxGym or
    kohya-ss — one concept folder, 1024px bucket, repeats=10.

    We don't generate the training config (optimizer/lr/epochs) here —
    the user's GPU/VRAM dictates that and FluxGym's UI handles it far
    better than any defaults we'd guess."""
    return (
        f'# Generated by ig-agent lora prepare for LoRA "{name}"\n'
        f'# Trigger word: {trigger}\n'
        f'# Image count:  {image_count}\n'
        '\n'
        '[general]\n'
        'shuffle_caption = false\n'
        'caption_extension = ".txt"\n'
        'keep_tokens = 1\n'
        '\n'
        '[[datasets]]\n'
        'resolution = [1024, 1024]\n'
        'batch_size = 1\n'
        '\n'
        '  [[datasets.subsets]]\n'
        f'  image_dir = "./{name}"\n'
        f'  class_tokens = "{trigger}"\n'
        '  num_repeats = 10\n'
    )


async def prepare_dataset(
    src_dir: Path,
    *,
    name: str,
    trigger: str,
    min_images: int = 10,
    auto_caption: bool = True,
) -> DatasetSummary:
    """Build a Kohya/FluxGym-ready dataset folder under data/lora_datasets/.

    * Copies every supported image from ``src_dir`` into the folder.
    * Writes a ``.txt`` caption file for each image. When ``auto_caption``
      is True, uses the existing vision LLM to describe the image; the
      trigger word is always placed first so Kohya's ``keep_tokens=1``
      keeps it anchored during caption shuffling.
    * Writes ``dataset.toml`` + ``README.md`` with the exact FluxGym /
      kohya-ss invocation the user should run.
    """
    name = validate_name(name)
    trigger = validate_trigger(trigger)
    images = _discover_images(src_dir)
    if len(images) < min_images:
        raise ValueError(
            f"Only {len(images)} image(s) under {src_dir} — LoRA training "
            f"needs at least {min_images} for stable results."
        )

    dataset_dir = LORA_DATASETS_DIR / name
    image_subdir = dataset_dir / name   # Kohya expects a subfolder by concept name
    image_subdir.mkdir(parents=True, exist_ok=True)

    # Clean out any prior run's files so stale captions can't leak in
    for p in image_subdir.iterdir():
        if p.is_file():
            p.unlink()

    # Lazy vision import so offline tests pass
    vision_fn = None
    if auto_caption:
        try:
            from instagram_ai_agent.core.llm import describe_image
            vision_fn = describe_image
        except Exception as e:
            log.warning("vision LLM unavailable, writing trigger-only captions: %s", e)

    captions_written = 0
    import base64
    for idx, img in enumerate(images):
        # Preserve order + give deterministic names so reruns are idempotent
        dest_img = image_subdir / f"{name}_{idx:03d}{img.suffix.lower()}"
        shutil.copy2(img, dest_img)

        caption_text = None
        if vision_fn is not None:
            try:
                mime = "image/jpeg" if dest_img.suffix.lower() in (".jpg", ".jpeg") else \
                       "image/png" if dest_img.suffix.lower() == ".png" else "image/webp"
                b64 = base64.b64encode(dest_img.read_bytes()).decode("ascii")
                data_url = f"data:{mime};base64,{b64}"
                caption_text = await asyncio.wait_for(
                    vision_fn(
                        data_url,
                        question=(
                            "Describe this image in one sentence — focus on background, "
                            "lighting, clothing, pose, setting. Do NOT invent a name. "
                            "Do NOT use quote marks. Keep it factual and brief."
                        ),
                    ),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                log.warning("auto-caption timed out for %s — trigger-only fallback", dest_img.name)
            except Exception as e:
                log.warning("auto-caption failed for %s: %s", dest_img.name, e)

        _write_caption(dest_img.with_suffix(".txt"), trigger, caption_text)
        captions_written += 1

    (dataset_dir / "dataset.toml").write_text(
        _fluxgym_dataset_toml(name, trigger, len(images)), encoding="utf-8",
    )
    (dataset_dir / "README.md").write_text(_readme(name, trigger, len(images)), encoding="utf-8")

    log.info(
        "LoRA dataset prepared — %d images, %d captions in %s",
        len(images), captions_written, dataset_dir,
    )
    return DatasetSummary(
        dataset_dir=dataset_dir,
        image_count=len(images),
        captions_written=captions_written,
    )


def _readme(name: str, trigger: str, count: int) -> str:
    return (
        f"# LoRA dataset — {name}\n\n"
        f"**Trigger word:** `{trigger}`\n"
        f"**Images:** {count}\n\n"
        "## Train with FluxGym\n\n"
        "1. Clone https://github.com/cocktailpeanut/fluxgym and follow its README to install.\n"
        "2. In the FluxGym UI, point the dataset at this folder's `./"
        f"{name}` subdirectory and set the trigger to `{trigger}`.\n"
        "3. For **commercial** use, pick base model `flux-schnell` (Apache-2.0) or `sdxl`.\n"
        "   FLUX.1-dev is non-commercial research licence and ig-agent blocks it when\n"
        "   `commercial=True` in niche.yaml.\n"
        "4. Start training. FluxGym writes a `.safetensors` file per epoch to its\n"
        "   output directory.\n\n"
        "## Train with kohya-ss directly\n\n"
        "```bash\n"
        "accelerate launch flux_train_network.py \\\n"
        "  --dataset_config dataset.toml \\\n"
        f"  --output_dir ./out --output_name {name} \\\n"
        "  --pretrained_model_name_or_path <path-to-schnell-unet.safetensors> \\\n"
        "  --save_model_as safetensors --network_module networks.lora_flux \\\n"
        "  --network_dim 16 --network_alpha 16 --learning_rate 1e-4 \\\n"
        "  --max_train_epochs 16 --mixed_precision bf16 \\\n"
        "  --network_train_unet_only\n"
        "```\n\n"
        "**Why `--network_train_unet_only`:** FLUX's CLIP text encoders (T5-XXL, CLIP-L)\n"
        "are frozen in the standard training recipe — attempting to train them\n"
        "typically destabilises the LoRA and wastes VRAM. Drop the flag only if\n"
        "you know you want a text-encoder-aware LoRA.\n\n"
        "## Import back into ig-agent\n\n"
        "```bash\n"
        f"ig-agent lora import <path-to>/{name}.safetensors \\\n"
        f"  --name {name} --trigger {trigger} --base-model flux-schnell\n"
        "ig-agent lora activate " + name + "\n"
        "```\n"
    )


# ─── Import / activation ───
def import_lora(
    source: Path,
    *,
    name: str,
    overwrite: bool = False,
) -> LoRAInfo:
    """Copy a trained .safetensors file into data/loras/ under the given
    name. Validates the file format (safetensors magic + JSON header)
    and returns a LoRAInfo with a best-effort base-model hint.
    """
    name = validate_name(name)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"LoRA file not found: {source}")
    if source.suffix.lower() != ".safetensors":
        raise ValueError(f"Only .safetensors is supported (got {source.suffix}).")
    if not _is_safetensors(source):
        raise ValueError(
            f"{source.name} does not parse as a safetensors file — "
            "is the download complete?"
        )

    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORAS_DIR / f"{name}.safetensors"
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"A LoRA named {name!r} already exists at {dest}. "
            "Pass --overwrite to replace it."
        )
    shutil.copy2(source, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    hint = _base_model_hint_from_safetensors(dest)
    log.info("LoRA imported: %s (%.1f MB, base hint=%s)", dest.name, size_mb, hint)
    return LoRAInfo(name=name, path=dest, size_mb=size_mb, base_model_hint=hint)


def list_loras() -> list[LoRAInfo]:
    """Enumerate every LoRA under data/loras/."""
    if not LORAS_DIR.exists():
        return []
    out: list[LoRAInfo] = []
    for p in sorted(LORAS_DIR.glob("*.safetensors")):
        size_mb = p.stat().st_size / (1024 * 1024)
        hint = _base_model_hint_from_safetensors(p)
        out.append(LoRAInfo(name=p.stem, path=p, size_mb=size_mb, base_model_hint=hint))
    return out


def remove_lora(name: str) -> bool:
    """Delete the LoRA file from data/loras/. Does NOT touch niche.yaml —
    if this is the active one, the caller should deactivate first."""
    name = validate_name(name)
    target = LORAS_DIR / f"{name}.safetensors"
    if not target.exists():
        return False
    target.unlink()
    log.info("LoRA removed: %s", target.name)
    return True


def activate_in_niche(
    cfg: NicheConfig,
    *,
    name: str,
    trigger: str,
    base_model: str | None = None,
    strength_model: float | None = None,
    strength_clip: float | None = None,
    niche_path: Path | None = None,
) -> NicheConfig:
    """Mark a LoRA as the active one in niche.yaml. Persists through save_niche.

    Raises if the LoRA file is missing or if base_model=flux-dev would
    collide with commercial=True (via NicheConfig's model validator)."""
    name = validate_name(name)
    trigger = validate_trigger(trigger)
    target = LORAS_DIR / f"{name}.safetensors"
    if not target.exists():
        raise FileNotFoundError(
            f"LoRA {name!r} not found at {target}. Import it first: "
            f"`ig-agent lora import <file> --name {name} --trigger {trigger}`."
        )

    from instagram_ai_agent.core.config import LoRAConfig

    lora_cfg = LoRAConfig(
        enabled=True,
        name=name,
        trigger_word=trigger,
        base_model=base_model or cfg.lora.base_model,
        strength_model=strength_model if strength_model is not None else cfg.lora.strength_model,
        strength_clip=strength_clip if strength_clip is not None else cfg.lora.strength_clip,
    )
    new_cfg = cfg.model_copy(update={"lora": lora_cfg})
    # Re-validate through the root so the commercial gate fires
    new_cfg = NicheConfig.model_validate(new_cfg.model_dump(mode="python"))
    save_niche(new_cfg, path=niche_path or NICHE_PATH)
    return new_cfg


def deactivate_in_niche(
    cfg: NicheConfig, *, niche_path: Path | None = None,
) -> NicheConfig:
    from instagram_ai_agent.core.config import LoRAConfig
    new_cfg = cfg.model_copy(update={"lora": LoRAConfig()})
    save_niche(new_cfg, path=niche_path or NICHE_PATH)
    return new_cfg


# ─── ComfyUI integration ───
def is_active(cfg: NicheConfig) -> bool:
    """True when niche.yaml has a LoRA enabled AND its file exists on disk."""
    if not cfg.lora.enabled or not cfg.lora.name or not cfg.lora.trigger_word:
        return False
    return (LORAS_DIR / f"{cfg.lora.name}.safetensors").exists()


def prepend_trigger(prompt: str, cfg: NicheConfig) -> str:
    """Prepend the LoRA trigger word to a positive image prompt.

    When the LoRA isn't active, returns the prompt unchanged."""
    if not is_active(cfg):
        return prompt
    trigger = cfg.lora.trigger_word.strip()
    if not trigger:
        return prompt
    # Avoid double-prepending if the caller already added it
    lower = prompt.lower().lstrip().split(",")[0].strip()
    if trigger.lower() == lower:
        return prompt
    return f"{trigger}, {prompt}" if prompt.strip() else trigger


# Node classes that expose the model pipe at slot 0.
_MODEL_LOADER_CLASSES = ("CheckpointLoaderSimple", "UNETLoader")
# Node classes that expose a CLIP pipe at slot 0. On SDXL via
# CheckpointLoaderSimple, clip is slot 1 of the same node; we handle
# that case separately below.
_CLIP_LOADER_CLASSES = ("DualCLIPLoader", "CLIPLoader")


def _find_first(workflow: dict, classes: tuple[str, ...]) -> str | None:
    for node_id, node in workflow.items():
        if node.get("class_type") in classes:
            return node_id
    return None


def inject_into_workflow(workflow: dict, cfg: NicheConfig) -> dict:
    """Insert a LoraLoader node into a ComfyUI workflow, rewiring the
    KSampler's model and every CLIPTextEncode's clip input to flow
    through the LoRA.

    Two anchor patterns are supported:
      * **SDXL / SD1.5** — ``CheckpointLoaderSimple`` (slot 0 = model,
        slot 1 = clip, slot 2 = vae).
      * **FLUX.1 native** — ``UNETLoader`` (slot 0 = model) + a
        ``DualCLIPLoader`` or ``CLIPLoader`` (slot 0 = clip). VAE comes
        from a separate ``VAELoader`` in this topology.

    Safe to call on workflows that already contain a LoraLoader:
      * same lora_name → update strength in place
      * different lora_name → stack: chain our LoraLoader off the
        existing one (brand LoRA + style LoRA is a common pattern).

    Returns the mutated workflow dict (same object)."""
    if not is_active(cfg):
        return workflow

    # Find anchor nodes. CheckpointLoaderSimple wins when both patterns
    # are present (defensive: a workflow wouldn't normally mix them).
    ckpt_id = _find_first(workflow, ("CheckpointLoaderSimple",))
    unet_id = None
    clip_id = None
    if ckpt_id is None:
        unet_id = _find_first(workflow, ("UNETLoader",))
        clip_id = _find_first(workflow, _CLIP_LOADER_CLASSES)

    if ckpt_id is not None:
        model_src = (ckpt_id, 0)
        clip_src = (ckpt_id, 1)
    elif unet_id is not None and clip_id is not None:
        model_src = (unet_id, 0)
        clip_src = (clip_id, 0)
    else:
        log.warning(
            "LoRA: workflow has no CheckpointLoaderSimple or UNETLoader+CLIPLoader "
            "pair — skipping LoRA injection. Edit your workflow JSON so the model "
            "and clip pipes are loaded from one of those node classes."
        )
        return workflow

    lora_name_file = f"{cfg.lora.name}.safetensors"

    # Existing-LoraLoader handling — only update in place when the
    # already-present LoRA has the SAME filename as ours. Different
    # filename means the user stacked another LoRA; we chain through.
    same_name_existing: str | None = None
    other_loaders: list[str] = []
    for node_id, node in workflow.items():
        if node.get("class_type") != "LoraLoader":
            continue
        if node.get("inputs", {}).get("lora_name") == lora_name_file:
            same_name_existing = node_id
        else:
            other_loaders.append(node_id)

    if same_name_existing is not None:
        workflow[same_name_existing]["inputs"].update({
            "lora_name": lora_name_file,
            "strength_model": cfg.lora.strength_model,
            "strength_clip": cfg.lora.strength_clip,
            "model": list(model_src),
            "clip": list(clip_src),
        })
        lora_id = same_name_existing
    else:
        # Fresh node id that won't collide with existing keys
        taken = {str(k) for k in workflow.keys()}
        candidate = 10
        while str(candidate) in taken:
            candidate += 1
        lora_id = str(candidate)
        # If other LoRAs are already present, chain off the last one so
        # strengths compose rather than clobbering. If not, chain off
        # the raw anchors.
        if other_loaders:
            tail = other_loaders[-1]
            model_src = (tail, 0)
            clip_src = (tail, 1)
        workflow[lora_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name_file,
                "strength_model": cfg.lora.strength_model,
                "strength_clip": cfg.lora.strength_clip,
                "model": list(model_src),
                "clip": list(clip_src),
            },
        }

    # Rewire every downstream consumer. The anchor whose outputs we
    # want to redirect to our LoRA is:
    #   * the last existing LoRA we chained off, when stacking, OR
    #   * the raw checkpoint / unet+clip pair otherwise.
    # (Nodes sitting BETWEEN the anchor and a sampler — ModelSamplingFlux,
    # ModelMergeSimple — require manual edits; we only rewire direct refs.)
    if other_loaders and same_name_existing is None:
        tail = other_loaders[-1]
        model_anchor_id, model_anchor_slot = tail, 0
        clip_anchor_id, clip_anchor_slot = tail, 1
    elif ckpt_id is not None:
        model_anchor_id, model_anchor_slot = ckpt_id, 0
        clip_anchor_id, clip_anchor_slot = ckpt_id, 1
    else:
        model_anchor_id, model_anchor_slot = unet_id, 0
        clip_anchor_id, clip_anchor_slot = clip_id, 0

    for node_id, node in workflow.items():
        if node_id == lora_id:
            continue
        # Don't rewire the OTHER LoRA loaders either — they must keep
        # pointing at the raw checkpoint so our LoRA can chain off them.
        if node_id in other_loaders:
            continue
        inputs = node.get("inputs") or {}
        for key, val in list(inputs.items()):
            if not (isinstance(val, list) and len(val) == 2):
                continue
            ref_node, ref_slot = val
            if str(ref_node) == str(model_anchor_id) and ref_slot == model_anchor_slot:
                inputs[key] = [lora_id, 0]
            elif str(ref_node) == str(clip_anchor_id) and ref_slot == clip_anchor_slot:
                inputs[key] = [lora_id, 1]
    return workflow
