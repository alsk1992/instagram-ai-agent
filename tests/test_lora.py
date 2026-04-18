"""Brand LoRA CLI — config gate, validators, dataset prep, import,
activate/deactivate, safetensors detection, ComfyUI workflow injection."""
from __future__ import annotations

import asyncio
import copy
import json
import struct
from pathlib import Path

import pytest

from instagram_ai_agent.core import config as cfg_mod
from instagram_ai_agent.plugins import comfyui, lora


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


def _fake_safetensors(path: Path, *, metadata: dict | None = None, payload_size: int = 128) -> Path:
    """Write a minimally-valid safetensors file to ``path``.

    Header is a JSON object with an optional __metadata__ block plus one
    fake tensor descriptor so the file parses cleanly."""
    hdr = {
        "fake.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
    }
    if metadata:
        hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    header_bytes = json.dumps(hdr).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * payload_size)
    return path


@pytest.fixture()
def patched_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect LORAS_DIR / LORA_DATASETS_DIR / NICHE_PATH into tmp_path so
    tests can't touch the real filesystem."""
    loras = tmp_path / "loras"
    datasets = tmp_path / "lora_datasets"
    niche_path = tmp_path / "niche.yaml"
    loras.mkdir()
    datasets.mkdir()
    monkeypatch.setattr(cfg_mod, "LORAS_DIR", loras)
    monkeypatch.setattr(cfg_mod, "LORA_DATASETS_DIR", datasets)
    monkeypatch.setattr(cfg_mod, "NICHE_PATH", niche_path)
    monkeypatch.setattr(lora, "LORAS_DIR", loras)
    monkeypatch.setattr(lora, "LORA_DATASETS_DIR", datasets)
    monkeypatch.setattr(lora, "NICHE_PATH", niche_path)
    yield {"loras": loras, "datasets": datasets, "niche": niche_path}


# ─── Config ───
def test_lora_config_defaults_sane():
    cfg = _mkcfg()
    assert cfg.lora.enabled is False
    assert cfg.lora.name == ""
    assert cfg.lora.trigger_word == ""
    assert cfg.lora.base_model == "flux-schnell"
    assert 0.5 <= cfg.lora.strength_model <= 1.5


def test_lora_config_roundtrips():
    import yaml
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand_v1", trigger_word="mascxyz", base_model="flux-dev"),
    )
    dumped = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(dumped)))
    assert loaded.lora.enabled is True
    assert loaded.lora.name == "brand_v1"
    assert loaded.lora.base_model == "flux-dev"


def test_lora_commercial_gate_blocks_flux_dev():
    """Commercial=True with flux-dev must fail loudly at config load."""
    with pytest.raises(Exception) as exc:
        _mkcfg(
            commercial=True,
            lora=cfg_mod.LoRAConfig(enabled=True, name="x", trigger_word="t", base_model="flux-dev"),
        )
    assert "flux-dev" in str(exc.value).lower() or "non-commercial" in str(exc.value).lower()


def test_lora_commercial_gate_allows_flux_schnell_under_commercial():
    cfg = _mkcfg(
        commercial=True,
        lora=cfg_mod.LoRAConfig(enabled=True, name="x", trigger_word="t", base_model="flux-schnell"),
    )
    assert cfg.lora.base_model == "flux-schnell"


def test_lora_commercial_gate_allows_sdxl_under_commercial():
    cfg = _mkcfg(
        commercial=True,
        lora=cfg_mod.LoRAConfig(enabled=True, name="x", trigger_word="t", base_model="sdxl"),
    )
    assert cfg.lora.base_model == "sdxl"


def test_lora_commercial_gate_allows_flux_dev_when_noncommercial():
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="x", trigger_word="t", base_model="flux-dev"),
    )
    assert cfg.lora.base_model == "flux-dev"


def test_lora_gate_ignores_disabled_lora_even_on_commercial():
    """A flux-dev base on a DISABLED LoRA shouldn't block load — the
    field is dormant. Only enabled LoRAs are enforced."""
    cfg = _mkcfg(
        commercial=True,
        lora=cfg_mod.LoRAConfig(enabled=False, base_model="flux-dev"),
    )
    assert cfg.lora.enabled is False


# ─── Name / trigger validation ───
@pytest.mark.parametrize("name,ok", [
    ("brand_v1", True),
    ("brand-v1", True),
    ("abc", True),
    ("A", True),
    ("1brand", True),
    ("", False),
    ("   ", False),
    ("bad/path", False),
    ("has spaces", False),
    ("../traversal", False),
    ("weird.name", False),
    ("_leading_underscore", False),  # must start with alnum
    ("-leading-dash", False),
    ("x" * 65, False),   # too long
])
def test_validate_name(name, ok):
    if ok:
        assert lora.validate_name(name) == name.strip()
    else:
        with pytest.raises(ValueError):
            lora.validate_name(name)


@pytest.mark.parametrize("trigger,ok", [
    ("mascxyz", True),
    ("brand_1", True),
    ("b-v1", True),
    ("", False),
    ("has space", False),
    ("bad.char", False),
    ("x" * 33, False),
])
def test_validate_trigger(trigger, ok):
    if ok:
        assert lora.validate_trigger(trigger) == trigger.strip()
    else:
        with pytest.raises(ValueError):
            lora.validate_trigger(trigger)


# ─── Safetensors detection ───
def test_is_safetensors_accepts_valid_file(tmp_path: Path):
    p = _fake_safetensors(tmp_path / "good.safetensors")
    assert lora._is_safetensors(p) is True


def test_is_safetensors_rejects_random_garbage(tmp_path: Path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"\x00" * 500)   # all zeros → header_len=0 → rejected
    assert lora._is_safetensors(p) is False


def test_is_safetensors_rejects_short_file(tmp_path: Path):
    p = tmp_path / "tiny.safetensors"
    p.write_bytes(b"\x00\x00")
    assert lora._is_safetensors(p) is False


def test_is_safetensors_rejects_absurd_header_len(tmp_path: Path):
    p = tmp_path / "huge.safetensors"
    # Header length claim of 1 TB → rejected
    p.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    assert lora._is_safetensors(p) is False


def test_is_safetensors_rejects_invalid_json(tmp_path: Path):
    p = tmp_path / "junk.safetensors"
    header = b"not-json-content"
    p.write_bytes(struct.pack("<Q", len(header)) + header)
    assert lora._is_safetensors(p) is False


# ─── Base model hint ───
def test_base_model_hint_detects_flux_schnell(tmp_path: Path):
    p = _fake_safetensors(
        tmp_path / "s.safetensors",
        metadata={"ss_base_model_version": "flux1_schnell"},
    )
    assert lora._base_model_hint_from_safetensors(p) == "flux-schnell"


def test_base_model_hint_detects_flux_dev(tmp_path: Path):
    p = _fake_safetensors(
        tmp_path / "d.safetensors",
        metadata={"ss_base_model_version": "flux_dev_v1"},
    )
    assert lora._base_model_hint_from_safetensors(p) == "flux-dev"


def test_base_model_hint_detects_sdxl(tmp_path: Path):
    p = _fake_safetensors(
        tmp_path / "x.safetensors",
        metadata={"ss_base_model_version": "sdxl_base_1_0"},
    )
    assert lora._base_model_hint_from_safetensors(p) == "sdxl"


def test_base_model_hint_returns_none_for_empty_metadata(tmp_path: Path):
    p = _fake_safetensors(tmp_path / "empty.safetensors")
    assert lora._base_model_hint_from_safetensors(p) is None


# ─── Import ───
def test_import_lora_copies_and_validates(patched_dirs, tmp_path: Path):
    src = _fake_safetensors(tmp_path / "src.safetensors")
    info = lora.import_lora(src, name="brand_v1")
    assert info.path == patched_dirs["loras"] / "brand_v1.safetensors"
    assert info.path.exists()
    assert info.size_mb > 0


def test_import_lora_rejects_non_safetensors_suffix(patched_dirs, tmp_path: Path):
    src = tmp_path / "wrong.bin"
    src.write_bytes(b"x" * 1000)
    with pytest.raises(ValueError):
        lora.import_lora(src, name="x")


def test_import_lora_rejects_bad_magic(patched_dirs, tmp_path: Path):
    src = tmp_path / "fake.safetensors"
    src.write_bytes(b"not-a-real-safetensors" * 10)
    with pytest.raises(ValueError):
        lora.import_lora(src, name="x")


def test_import_lora_missing_source(patched_dirs, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        lora.import_lora(tmp_path / "nope.safetensors", name="x")


def test_import_lora_refuses_overwrite_unless_flag(patched_dirs, tmp_path: Path):
    src = _fake_safetensors(tmp_path / "s.safetensors")
    lora.import_lora(src, name="dup")
    with pytest.raises(FileExistsError):
        lora.import_lora(src, name="dup")
    # --overwrite lets it through
    info = lora.import_lora(src, name="dup", overwrite=True)
    assert info.path.exists()


# ─── list / remove ───
def test_list_loras_enumerates_dir(patched_dirs, tmp_path: Path):
    for n in ("a", "b", "c"):
        _fake_safetensors(patched_dirs["loras"] / f"{n}.safetensors")
    got = lora.list_loras()
    assert [i.name for i in got] == ["a", "b", "c"]


def test_list_loras_empty(patched_dirs):
    assert lora.list_loras() == []


def test_remove_lora_unlinks_file(patched_dirs, tmp_path: Path):
    _fake_safetensors(patched_dirs["loras"] / "x.safetensors")
    assert lora.remove_lora("x") is True
    assert not (patched_dirs["loras"] / "x.safetensors").exists()


def test_remove_lora_returns_false_when_missing(patched_dirs):
    assert lora.remove_lora("ghost") is False


# ─── activate / deactivate ───
def test_activate_in_niche_requires_existing_file(patched_dirs):
    cfg = _mkcfg()
    with pytest.raises(FileNotFoundError):
        lora.activate_in_niche(cfg, name="does_not_exist", trigger="t")


def test_activate_in_niche_writes_to_yaml(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg()
    new_cfg = lora.activate_in_niche(
        cfg, name="brand", trigger="mascxyz", base_model="flux-schnell",
    )
    assert new_cfg.lora.enabled is True
    assert new_cfg.lora.name == "brand"
    assert new_cfg.lora.trigger_word == "mascxyz"
    # File must be persisted
    assert patched_dirs["niche"].exists()


def test_activate_in_niche_blocks_flux_dev_on_commercial(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(commercial=True)
    with pytest.raises(Exception) as exc:
        lora.activate_in_niche(cfg, name="brand", trigger="mascxyz", base_model="flux-dev")
    assert "flux-dev" in str(exc.value).lower() or "non-commercial" in str(exc.value).lower()


def test_deactivate_returns_to_defaults(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg()
    cfg = lora.activate_in_niche(cfg, name="brand", trigger="mascxyz")
    assert cfg.lora.enabled is True
    cfg = lora.deactivate_in_niche(cfg)
    assert cfg.lora.enabled is False
    assert cfg.lora.name == ""


# ─── is_active / prepend_trigger ───
def test_is_active_requires_file_on_disk(patched_dirs):
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="ghost", trigger_word="t"),
    )
    # File doesn't exist → inactive despite enabled=True
    assert lora.is_active(cfg) is False


def test_is_active_true_when_file_and_config_match(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    assert lora.is_active(cfg) is True


def test_prepend_trigger_noop_when_inactive(patched_dirs):
    cfg = _mkcfg()
    assert lora.prepend_trigger("a man running", cfg) == "a man running"


def test_prepend_trigger_prepends_when_active(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="mascxyz"),
    )
    assert lora.prepend_trigger("a man running", cfg) == "mascxyz, a man running"


def test_prepend_trigger_no_double_prefix(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="mascxyz"),
    )
    already = "mascxyz, a man running"
    assert lora.prepend_trigger(already, cfg) == already


# ─── ComfyUI workflow injection ───
def _default_wf() -> dict:
    return copy.deepcopy(comfyui._DEFAULT_WORKFLOW)


def test_inject_noop_when_inactive(patched_dirs):
    cfg = _mkcfg()
    wf = _default_wf()
    out = lora.inject_into_workflow(wf, cfg)
    assert "LoraLoader" not in {n.get("class_type") for n in out.values()}


def test_inject_adds_loraloader_node_when_active(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    out = lora.inject_into_workflow(wf, cfg)
    loaders = [n for n in out.values() if n.get("class_type") == "LoraLoader"]
    assert len(loaders) == 1
    assert loaders[0]["inputs"]["lora_name"] == "brand.safetensors"


def test_inject_rewires_ksampler_model_through_lora(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    out = lora.inject_into_workflow(wf, cfg)
    # Find the LoraLoader's node_id and the KSampler
    lora_id = next(nid for nid, n in out.items() if n.get("class_type") == "LoraLoader")
    ks = next(n for n in out.values() if n.get("class_type") == "KSampler")
    assert ks["inputs"]["model"] == [lora_id, 0]


def test_inject_rewires_cliptextencode_clip_through_lora(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    out = lora.inject_into_workflow(wf, cfg)
    lora_id = next(nid for nid, n in out.items() if n.get("class_type") == "LoraLoader")
    for n in out.values():
        if n.get("class_type") == "CLIPTextEncode":
            assert n["inputs"]["clip"] == [lora_id, 1]


def test_inject_does_not_rewire_vae_pipe(patched_dirs):
    """VAE (slot 2 on CheckpointLoaderSimple) must NOT route through the
    LoRA — only model (0) and clip (1) go through it."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    out = lora.inject_into_workflow(wf, cfg)
    # VAEDecode should still point at the checkpoint node, slot 2
    ckpt_id = next(nid for nid, n in out.items() if n.get("class_type") == "CheckpointLoaderSimple")
    vae = next(n for n in out.values() if n.get("class_type") == "VAEDecode")
    assert vae["inputs"]["vae"] == [ckpt_id, 2]


def test_inject_idempotent_updates_existing_loraloader(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(
            enabled=True, name="brand", trigger_word="t",
            strength_model=0.70, strength_clip=0.80,
        ),
    )
    wf = _default_wf()
    lora.inject_into_workflow(wf, cfg)
    # Inject again — must NOT stack a second LoraLoader
    lora.inject_into_workflow(wf, cfg)
    loaders = [n for n in wf.values() if n.get("class_type") == "LoraLoader"]
    assert len(loaders) == 1
    assert loaders[0]["inputs"]["strength_model"] == 0.70
    assert loaders[0]["inputs"]["strength_clip"] == 0.80


def test_inject_skips_gracefully_when_no_checkpoint_node(patched_dirs):
    """A weird user workflow without CheckpointLoaderSimple should not crash."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    weird_wf = {"1": {"class_type": "SomeFluxModelLoader", "inputs": {"ckpt": "flux-schnell"}}}
    out = lora.inject_into_workflow(weird_wf, cfg)
    assert "LoraLoader" not in {n.get("class_type") for n in out.values()}


def _flux_native_wf() -> dict:
    """Minimal FLUX-native workflow: UNETLoader + DualCLIPLoader + VAELoader,
    representative of how FluxGym-trained LoRAs are actually loaded."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-schnell.safetensors"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5xxl.safetensors", "clip_name2": "clip_l.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad", "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": 0, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "flux"}},
    }


def test_inject_supports_flux_native_unetloader_workflow(patched_dirs):
    """Audit fix: FLUX-native workflows use UNETLoader + DualCLIPLoader
    instead of CheckpointLoaderSimple. We must wire the LoRA onto those
    two pipes, not silently no-op."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=True,  # flux-schnell is commercial-OK
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t", base_model="flux-schnell"),
    )
    wf = _flux_native_wf()
    out = lora.inject_into_workflow(wf, cfg)
    loaders = [nid for nid, n in out.items() if n.get("class_type") == "LoraLoader"]
    assert len(loaders) == 1
    lora_id = loaders[0]
    # LoraLoader's OWN inputs point at UNETLoader (model) + DualCLIPLoader (clip)
    assert out[lora_id]["inputs"]["model"] == [1, 0] or out[lora_id]["inputs"]["model"] == ["1", 0]
    assert out[lora_id]["inputs"]["clip"] == [2, 0] or out[lora_id]["inputs"]["clip"] == ["2", 0]


def test_inject_rewires_flux_native_ksampler_and_clip(patched_dirs):
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=True,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t", base_model="flux-schnell"),
    )
    wf = _flux_native_wf()
    out = lora.inject_into_workflow(wf, cfg)
    lora_id = next(nid for nid, n in out.items() if n.get("class_type") == "LoraLoader")
    ks = out["7"]
    assert ks["inputs"]["model"] == [lora_id, 0]
    # Both CLIPTextEncodes flipped onto LoRA.clip
    for nid in ("4", "5"):
        assert out[nid]["inputs"]["clip"] == [lora_id, 1]
    # VAE stays on the VAELoader — LoRA does NOT touch VAE pipe
    assert out["8"]["inputs"]["vae"] == ["3", 0]


def test_inject_flux_native_fails_gracefully_without_clip_loader(patched_dirs):
    """UNETLoader alone without a CLIP loader → skip injection."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=True,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t", base_model="flux-schnell"),
    )
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    out = lora.inject_into_workflow(wf, cfg)
    assert "LoraLoader" not in {n.get("class_type") for n in out.values()}


def test_inject_stacks_with_different_named_lora(patched_dirs):
    """Audit fix: when a DIFFERENT LoRA is already in the workflow, we
    must chain off it rather than overwriting — so brand + style LoRAs
    can compose."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    # Pre-insert a user "style" LoRA wired off the checkpoint
    wf["15"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "style_painterly.safetensors",
            "strength_model": 0.6,
            "strength_clip": 0.6,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }
    # Rewire KSampler/CLIPTextEncode to pass through the existing style LoRA
    wf["3"]["inputs"]["model"] = ["15", 0]
    for nid in ("6", "7"):
        wf[nid]["inputs"]["clip"] = ["15", 1]

    out = lora.inject_into_workflow(wf, cfg)
    loaders = [nid for nid, n in out.items() if n.get("class_type") == "LoraLoader"]
    # Must now have TWO LoRAs — style + our brand
    assert len(loaders) == 2
    our_id = next(nid for nid in loaders if out[nid]["inputs"]["lora_name"] == "brand.safetensors")
    # Our LoRA chains off the style LoRA, not the checkpoint
    assert out[our_id]["inputs"]["model"] == ["15", 0]
    assert out[our_id]["inputs"]["clip"] == ["15", 1]
    # KSampler ends up behind OUR LoRA
    assert out["3"]["inputs"]["model"] == [our_id, 0]


def test_inject_updates_in_place_when_same_name_lora_already_present(patched_dirs):
    """Same-name LoRA → update strengths + wiring in place, don't stack
    multiple copies of the same concept."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(
            enabled=True, name="brand", trigger_word="t",
            strength_model=0.95, strength_clip=0.60,
        ),
    )
    wf = _default_wf()
    wf["15"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "brand.safetensors",
            "strength_model": 0.4,
            "strength_clip": 0.4,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }
    out = lora.inject_into_workflow(wf, cfg)
    loaders = [nid for nid, n in out.items() if n.get("class_type") == "LoraLoader"]
    assert len(loaders) == 1
    assert out[loaders[0]]["inputs"]["strength_model"] == 0.95
    assert out[loaders[0]]["inputs"]["strength_clip"] == 0.60


def test_base_model_hint_prefers_dev_over_schnell_when_ambiguous(tmp_path: Path):
    """Audit fix: a mixed string like 'flux_dev_schnell_merged' must
    resolve to the STRICTER licence (flux-dev) so the commercial gate
    can catch it on import."""
    p = _fake_safetensors(
        tmp_path / "ambig.safetensors",
        metadata={"ss_base_model_version": "flux_dev_schnell_merged"},
    )
    assert lora._base_model_hint_from_safetensors(p) == "flux-dev"


def test_inject_chooses_fresh_node_id(patched_dirs):
    """If workflow already uses node IDs 10, 11, we must pick a higher free one."""
    _fake_safetensors(patched_dirs["loras"] / "brand.safetensors")
    cfg = _mkcfg(
        commercial=False,
        lora=cfg_mod.LoRAConfig(enabled=True, name="brand", trigger_word="t"),
    )
    wf = _default_wf()
    wf["10"] = {"class_type": "SomethingElse", "inputs": {}}
    wf["11"] = {"class_type": "SomethingElse", "inputs": {}}
    out = lora.inject_into_workflow(wf, cfg)
    lora_ids = [nid for nid, n in out.items() if n.get("class_type") == "LoraLoader"]
    assert len(lora_ids) == 1
    assert lora_ids[0] not in ("10", "11")


# ─── Dataset prep ───
def _mkjpeg(path: Path, color: tuple[int, int, int] = (128, 64, 200)) -> Path:
    """Write a tiny valid JPEG via PIL so discover_images + copies work."""
    from PIL import Image
    img = Image.new("RGB", (16, 16), color)
    img.save(path, "JPEG")
    return path


@pytest.mark.asyncio
async def test_prepare_dataset_fails_with_too_few_images(patched_dirs, tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _mkjpeg(src / f"img{i}.jpg")
    with pytest.raises(ValueError):
        await lora.prepare_dataset(src, name="brand", trigger="mascxyz", auto_caption=False)


@pytest.mark.asyncio
async def test_prepare_dataset_happy_path_no_auto_caption(patched_dirs, tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        _mkjpeg(src / f"img{i}.jpg")
    summary = await lora.prepare_dataset(
        src, name="brand", trigger="mascxyz", min_images=10, auto_caption=False,
    )
    assert summary.image_count == 12
    assert summary.captions_written == 12
    # Check folder layout
    assert (summary.dataset_dir / "dataset.toml").exists()
    assert (summary.dataset_dir / "README.md").exists()
    concept = summary.dataset_dir / "brand"
    assert concept.is_dir()
    jpegs = sorted(concept.glob("*.jpg"))
    captions = sorted(concept.glob("*.txt"))
    assert len(jpegs) == 12
    assert len(captions) == 12
    # Every caption must lead with the trigger word
    for c in captions:
        assert c.read_text().startswith("mascxyz")


@pytest.mark.asyncio
async def test_prepare_dataset_rejects_non_directory(patched_dirs, tmp_path: Path):
    p = tmp_path / "not_a_dir"
    p.write_text("x")
    with pytest.raises(NotADirectoryError):
        await lora.prepare_dataset(p, name="x", trigger="t", auto_caption=False)


@pytest.mark.asyncio
async def test_prepare_dataset_ignores_non_image_files(patched_dirs, tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        _mkjpeg(src / f"ok{i}.jpg")
    (src / "README.md").write_text("not an image")
    (src / "random.bin").write_bytes(b"\x00" * 100)
    summary = await lora.prepare_dataset(
        src, name="brand", trigger="mascxyz", min_images=10, auto_caption=False,
    )
    assert summary.image_count == 10


@pytest.mark.asyncio
async def test_prepare_dataset_auto_caption_uses_vision(patched_dirs, tmp_path: Path, monkeypatch):
    """When auto_caption=True, vision LLM is called and its text goes
    after the trigger word."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        _mkjpeg(src / f"img{i}.jpg")

    calls = {"n": 0}

    async def fake_describe(url, question=""):
        calls["n"] += 1
        return "a man standing outside at dusk with soft orange backlight"

    # Patch the late-imported describe_image symbol on the llm module
    import instagram_ai_agent.core.llm as llm_mod
    monkeypatch.setattr(llm_mod, "describe_image", fake_describe)

    summary = await lora.prepare_dataset(
        src, name="brand", trigger="mascxyz", auto_caption=True,
    )
    assert calls["n"] == 10
    # First caption should have the LLM output appended after the trigger
    first_caption = next((summary.dataset_dir / "brand").glob("*_000.txt"))
    text = first_caption.read_text()
    assert text.startswith("mascxyz,")
    assert "dusk" in text


@pytest.mark.asyncio
async def test_prepare_dataset_survives_vision_failure(patched_dirs, tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        _mkjpeg(src / f"img{i}.jpg")

    async def broken(url, question=""):
        raise RuntimeError("LLM down")

    import instagram_ai_agent.core.llm as llm_mod
    monkeypatch.setattr(llm_mod, "describe_image", broken)

    summary = await lora.prepare_dataset(
        src, name="brand", trigger="mascxyz", auto_caption=True,
    )
    # Still wrote captions (trigger-only fallback)
    assert summary.captions_written == 10
    first = next((summary.dataset_dir / "brand").glob("*_000.txt"))
    assert first.read_text().strip() == "mascxyz"


@pytest.mark.asyncio
async def test_prepare_dataset_rerun_overwrites_old_captions(patched_dirs, tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        _mkjpeg(src / f"img{i}.jpg")
    await lora.prepare_dataset(src, name="brand", trigger="v1", min_images=10, auto_caption=False)
    # Second run with a different trigger word — captions must be rewritten,
    # old stale files must not linger.
    summary = await lora.prepare_dataset(
        src, name="brand", trigger="v2new", min_images=10, auto_caption=False,
    )
    captions = list((summary.dataset_dir / "brand").glob("*.txt"))
    assert len(captions) == 10
    for c in captions:
        assert c.read_text().startswith("v2new")
        assert "v1" not in c.read_text()
