"""ComfyUI HTTP client for local GPU image generation.

ComfyUI exposes a ``/prompt`` endpoint that accepts a workflow JSON. We ship
a minimal text-to-image workflow skeleton keyed by placeholders; users drop
the workflow exported from their ComfyUI graph into ``data/comfyui_workflows/``.

Activation: when ``COMFYUI_URL`` is set (e.g. ``http://localhost:8188``) the
photo / human_photo generators route through here *first* and fall back to
Pollinations only on error. Unset env → no-op, behaviour identical to before.

Recommended commercial-safe workflow pairings:
  * SDXL + RealVisXL LoRA  (CreativeML OpenRAIL++) — safe for monetised pages
  * Flux.1 schnell         (Apache-2.0)            — safe, and the best free quality
  * Flux.1 dev             (non-commercial)        — explicit opt-in only
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from pathlib import Path

import httpx

from src.content.generators.base import staging_path
from src.core.config import DATA_DIR, NicheConfig
from src.core.logging_setup import get_logger

log = get_logger(__name__)

WORKFLOW_DIR = DATA_DIR / "comfyui_workflows"

# Minimal, self-contained SDXL text-to-image workflow.
# Users can replace this with an export from their own ComfyUI graph.
_DEFAULT_WORKFLOW: dict = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,              # filled per call
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad anatomy, lowres, watermark, text", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ig_agent", "images": ["8", 0]},
    },
}


def configured() -> bool:
    return bool(os.environ.get("COMFYUI_URL"))


def _base_url() -> str:
    return os.environ["COMFYUI_URL"].rstrip("/")


def _load_workflow(name: str | None = None) -> dict:
    """Return a workflow dict — from disk if present, else the default."""
    if name:
        candidate = WORKFLOW_DIR / name
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    # Look for any single workflow on disk
    if WORKFLOW_DIR.exists():
        for p in sorted(WORKFLOW_DIR.glob("*.json")):
            return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(json.dumps(_DEFAULT_WORKFLOW))  # deep copy


def _apply_params(
    workflow: dict,
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
) -> dict:
    """Walk workflow nodes and fill standard text-to-image slots.

    Supports both the default workflow above and reasonable user-exported
    graphs. We match by class_type rather than node id so users can
    rearrange freely.
    """
    for node in workflow.values():
        cls = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if cls == "KSampler":
            inputs["seed"] = seed
        elif cls == "EmptyLatentImage":
            inputs["width"] = width
            inputs["height"] = height
        elif cls == "CLIPTextEncode":
            # First CLIPTextEncode == positive, second == negative.
            # Heuristic: detect by presence of "text" starting with lowercase verbs vs
            # "bad anatomy" pattern. Simpler: mark first-seen as positive.
            if not getattr(_apply_params, "_pos_set", False):
                inputs["text"] = prompt
                setattr(_apply_params, "_pos_set", True)
            else:
                inputs["text"] = negative or inputs.get("text", "")
    # Reset the flag for the next call
    if hasattr(_apply_params, "_pos_set"):
        delattr(_apply_params, "_pos_set")
    return workflow


async def generate(
    prompt: str,
    *,
    negative: str = "",
    width: int = 1080,
    height: int = 1350,
    seed: int | None = None,
    workflow_name: str | None = None,
    timeout_s: float = 240.0,
    cfg: NicheConfig | None = None,
) -> Path:
    """Submit a text-to-image job to ComfyUI and return the resulting file path.

    When ``cfg`` is supplied and a LoRA is active, the workflow is
    patched with a LoraLoader node and the trigger word is prepended
    to the positive prompt before submission.
    """
    if not configured():
        raise RuntimeError("COMFYUI_URL not configured")

    workflow = _load_workflow(workflow_name)
    seed = seed if seed is not None else random.randint(1, 10**9)

    # Brand LoRA: prepend the trigger word + inject LoraLoader if active.
    if cfg is not None:
        from src.plugins import controlnet as _cn
        from src.plugins import lora as _lora
        prompt = _lora.prepend_trigger(prompt, cfg)
        workflow = _lora.inject_into_workflow(workflow, cfg)
        # ControlNet goes AFTER LoRA so the LoRA-adjusted positive/
        # negative conditionings are the inputs ControlNet wraps.
        workflow = _cn.inject_into_workflow(workflow, cfg)

    workflow = _apply_params(
        workflow,
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
    )

    client_id = uuid.uuid4().hex
    payload = {"prompt": workflow, "client_id": client_id}
    base = _base_url()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        r = await client.post(f"{base}/prompt", json=payload)
        r.raise_for_status()
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {r.text[:200]}")

        # Poll /history/{id} until the job finishes.
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout_s}s")
            history_r = await client.get(f"{base}/history/{prompt_id}")
            if history_r.status_code == 200:
                data = history_r.json()
                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs") or {}
                    image_refs = _collect_image_refs(outputs)
                    if image_refs:
                        ref = image_refs[0]
                        img_r = await client.get(
                            f"{base}/view",
                            params={
                                "filename": ref["filename"],
                                "subfolder": ref.get("subfolder", ""),
                                "type": ref.get("type", "output"),
                            },
                        )
                        img_r.raise_for_status()
                        dest = staging_path("comfy", ".png")
                        dest.write_bytes(img_r.content)
                        return dest
            await asyncio.sleep(1.5)


def _collect_image_refs(outputs: dict) -> list[dict]:
    refs: list[dict] = []
    for node_id, node_out in outputs.items():
        for img in node_out.get("images") or []:
            if img.get("filename"):
                refs.append(img)
    return refs
