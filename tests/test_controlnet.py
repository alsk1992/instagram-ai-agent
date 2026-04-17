"""ControlNet pose/depth/canny — config gate, reference storage,
workflow injection, commercial guards."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PIL import Image

from src.core import config as cfg_mod
from src.plugins import comfyui, controlnet


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
def patched_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cn_ref = tmp_path / "controlnet"
    cn_mod = tmp_path / "controlnet_models"
    niche_yaml = tmp_path / "niche.yaml"
    cn_ref.mkdir()
    cn_mod.mkdir()
    monkeypatch.setattr(cfg_mod, "CONTROLNET_DIR", cn_ref)
    monkeypatch.setattr(cfg_mod, "CONTROLNET_MODELS_DIR", cn_mod)
    monkeypatch.setattr(cfg_mod, "NICHE_PATH", niche_yaml)
    monkeypatch.setattr(controlnet, "CONTROLNET_DIR", cn_ref)
    monkeypatch.setattr(controlnet, "CONTROLNET_MODELS_DIR", cn_mod)
    monkeypatch.setattr(controlnet, "NICHE_PATH", niche_yaml)
    yield {"ref": cn_ref, "models": cn_mod, "niche": niche_yaml}


def _mkimage(path: Path, size=(32, 32)) -> Path:
    Image.new("RGB", size, (0, 200, 100)).save(path, "PNG")
    return path


def _mksafetensors(path: Path) -> Path:
    """Minimal safetensors file layout — reused from test_lora helper."""
    import json
    import struct
    header = json.dumps({"fake.w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return path


# ─── Config ───
def test_controlnet_config_defaults_sane():
    cfg = _mkcfg()
    assert cfg.controlnet.enabled is False
    assert cfg.controlnet.mode == "pose"
    assert cfg.controlnet.reference_image == ""
    assert cfg.controlnet.model_name == ""
    assert 0.0 <= cfg.controlnet.strength <= 2.0


def test_controlnet_config_roundtrips():
    import yaml
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="depth", reference_image="depth.png",
        model_name="cn_depth.safetensors", strength=0.6,
    ))
    d = cfg.model_dump(mode="json")
    loaded = cfg_mod.NicheConfig.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert loaded.controlnet.mode == "depth"
    assert loaded.controlnet.reference_image == "depth.png"
    assert loaded.controlnet.strength == 0.6


def test_controlnet_end_before_start_rejected():
    with pytest.raises(Exception):
        cfg_mod.ControlNetConfig(start_percent=0.8, end_percent=0.3)


def test_controlnet_commercial_gate_blocks_openpose_override():
    with pytest.raises(Exception) as exc:
        _mkcfg(
            commercial=True,
            controlnet=cfg_mod.ControlNetConfig(
                enabled=True, mode="pose", preprocessor_override="OpenposePreprocessor",
            ),
        )
    assert "openpose" in str(exc.value).lower() or "non-commercial" in str(exc.value).lower()


def test_controlnet_commercial_gate_allows_openpose_when_noncommercial():
    cfg = _mkcfg(
        commercial=False,
        controlnet=cfg_mod.ControlNetConfig(
            enabled=True, mode="pose", preprocessor_override="OpenposePreprocessor",
        ),
    )
    assert cfg.controlnet.preprocessor_override == "OpenposePreprocessor"


def test_controlnet_commercial_gate_skips_when_disabled():
    cfg = _mkcfg(
        commercial=True,
        controlnet=cfg_mod.ControlNetConfig(
            enabled=False, preprocessor_override="OpenposePreprocessor",
        ),
    )
    assert cfg.controlnet.enabled is False


# ─── mode_for ───
def test_mode_for_returns_default_preprocessor_per_mode():
    for mode, (preproc, lic) in controlnet._DEFAULT_PREPROCESSORS.items():
        cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(enabled=True, mode=mode))
        got = controlnet.mode_for(cfg)
        assert got.preprocessor == preproc
        assert got.license == lic


def test_mode_for_honours_override():
    cfg = _mkcfg(
        commercial=False,
        controlnet=cfg_mod.ControlNetConfig(
            enabled=True, mode="pose", preprocessor_override="MyCustomPreprocessor",
        ),
    )
    got = controlnet.mode_for(cfg)
    assert got.preprocessor == "MyCustomPreprocessor"
    assert got.license == "user-declared"


# ─── Reference-image CLI-surface ───
def test_set_reference_copies_and_persists(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = _mkimage(tmp_path / "src.png")
    dest = controlnet.set_reference(src, mode="pose", cfg=cfg)
    assert dest == patched_dirs["ref"] / "pose.png"
    assert dest.exists()
    # Persisted to niche.yaml
    assert patched_dirs["niche"].exists()


def test_set_reference_rejects_unknown_mode(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = _mkimage(tmp_path / "src.png")
    with pytest.raises(ValueError):
        controlnet.set_reference(src, mode="bogus", cfg=cfg)


def test_set_reference_rejects_non_image_extension(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = tmp_path / "not_an_image.bin"
    src.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError):
        controlnet.set_reference(src, mode="pose", cfg=cfg)


def test_set_reference_replaces_prior_mode_file(patched_dirs, tmp_path: Path):
    """Setting pose.jpg then pose.png must remove the prior pose.jpg."""
    cfg = _mkcfg()
    old = _mkimage(tmp_path / "old.jpg")
    # Write the JPG-extension first so the prune-loop has something to remove
    dest_jpg = patched_dirs["ref"] / "pose.jpg"
    import shutil as sh
    sh.copy2(old, dest_jpg)
    # Now set a PNG — old JPG must be deleted
    new = _mkimage(tmp_path / "new.png")
    dest = controlnet.set_reference(new, mode="pose", cfg=cfg)
    assert dest.suffix == ".png"
    assert not dest_jpg.exists()


def test_set_reference_blocks_openpose_override_under_commercial(patched_dirs, tmp_path: Path):
    """Re-validation through NicheConfig should trip the commercial gate."""
    cfg = _mkcfg(
        commercial=True,
        controlnet=cfg_mod.ControlNetConfig(
            enabled=False, preprocessor_override="OpenposePreprocessor",
        ),
    )
    # The config itself validated fine (enabled=False). But calling
    # set_reference flips enabled=True which triggers the gate on re-
    # validation. Must raise.
    src = _mkimage(tmp_path / "src.png")
    with pytest.raises(Exception):
        controlnet.set_reference(src, mode="pose", cfg=cfg)


# ─── model import ───
def test_set_model_copies_and_persists(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = _mksafetensors(tmp_path / "cn.safetensors")
    dest = controlnet.set_model(src, cfg=cfg)
    assert dest.exists()
    assert dest.name == "cn.safetensors"


def test_set_model_rejects_non_safetensors(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = tmp_path / "wrong.bin"
    src.write_bytes(b"x" * 500)
    with pytest.raises(ValueError):
        controlnet.set_model(src, cfg=cfg)


def test_set_model_refuses_overwrite_unless_flag(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    src = _mksafetensors(tmp_path / "dup.safetensors")
    controlnet.set_model(src, cfg=cfg)
    with pytest.raises(FileExistsError):
        controlnet.set_model(src, cfg=cfg)
    controlnet.set_model(src, cfg=cfg, overwrite=True)


# ─── clear_reference ───
def test_clear_reference_wipes_images_and_disables(patched_dirs, tmp_path: Path):
    cfg = _mkcfg()
    for mode in ("pose", "depth"):
        _mkimage(patched_dirs["ref"] / f"{mode}.png")
    # Simulate prior active config
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="pose", reference_image="pose.png",
        model_name="cn.safetensors",
    ))
    new = controlnet.clear_reference(cfg)
    assert new.controlnet.enabled is False
    for mode in ("pose", "depth", "canny"):
        assert not (patched_dirs["ref"] / f"{mode}.png").exists()


# ─── is_active ───
def test_is_active_false_when_disabled(patched_dirs):
    cfg = _mkcfg()
    assert controlnet.is_active(cfg) is False


def test_is_active_false_when_reference_missing(patched_dirs):
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, reference_image="ghost.png", model_name="cn.safetensors",
    ))
    _mksafetensors(patched_dirs["models"] / "cn.safetensors")
    assert controlnet.is_active(cfg) is False


def test_is_active_false_when_model_missing(patched_dirs):
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, reference_image="pose.png", model_name="ghost.safetensors",
    ))
    _mkimage(patched_dirs["ref"] / "pose.png")
    assert controlnet.is_active(cfg) is False


def test_is_active_true_when_all_present(patched_dirs):
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, reference_image="pose.png", model_name="cn.safetensors",
    ))
    _mkimage(patched_dirs["ref"] / "pose.png")
    _mksafetensors(patched_dirs["models"] / "cn.safetensors")
    assert controlnet.is_active(cfg) is True


# ─── workflow injection ───
def _default_wf() -> dict:
    return copy.deepcopy(comfyui._DEFAULT_WORKFLOW)


def _active_cfg(patched_dirs, **overrides) -> cfg_mod.NicheConfig:
    _mkimage(patched_dirs["ref"] / "pose.png")
    _mksafetensors(patched_dirs["models"] / "cn.safetensors")
    defaults = dict(
        enabled=True, mode="pose",
        reference_image="pose.png", model_name="cn.safetensors",
        strength=0.75,
    )
    defaults.update(overrides)
    return _mkcfg(controlnet=cfg_mod.ControlNetConfig(**defaults))


def test_inject_noop_when_inactive(patched_dirs):
    cfg = _mkcfg()
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    assert "ControlNetApplyAdvanced" not in {n.get("class_type") for n in out.values()}


def test_inject_adds_four_nodes(patched_dirs):
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    classes = {n.get("class_type") for n in out.values()}
    for expected in ("LoadImage", "DWPreprocessor", "ControlNetLoader", "ControlNetApplyAdvanced"):
        assert expected in classes, f"missing {expected}"


def test_inject_rewires_ksampler_conditioning(patched_dirs):
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    apply_id = next(nid for nid, n in out.items() if n.get("class_type") == "ControlNetApplyAdvanced")
    ks = next(n for n in out.values() if n.get("class_type") == "KSampler")
    assert ks["inputs"]["positive"] == [apply_id, 0]
    assert ks["inputs"]["negative"] == [apply_id, 1]


def test_inject_apply_node_preserves_original_conditioning(patched_dirs):
    """The positive/negative refs passed into ControlNetApplyAdvanced
    must match the sampler's ORIGINAL refs (before we overwrote them)."""
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    original_positive = wf["3"]["inputs"]["positive"]
    original_negative = wf["3"]["inputs"]["negative"]
    out = controlnet.inject_into_workflow(wf, cfg)
    apply_node = next(n for n in out.values() if n.get("class_type") == "ControlNetApplyAdvanced")
    assert apply_node["inputs"]["positive"] == original_positive
    assert apply_node["inputs"]["negative"] == original_negative


def test_inject_mode_depth_uses_depth_preprocessor(patched_dirs):
    _mkimage(patched_dirs["ref"] / "depth.png")
    _mksafetensors(patched_dirs["models"] / "cn_depth.safetensors")
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="depth",
        reference_image="depth.png", model_name="cn_depth.safetensors",
    ))
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    classes = {n.get("class_type") for n in out.values()}
    assert "DepthAnythingV2Preprocessor" in classes
    assert "DWPreprocessor" not in classes


def test_inject_mode_canny_uses_canny_preprocessor(patched_dirs):
    _mkimage(patched_dirs["ref"] / "canny.png")
    _mksafetensors(patched_dirs["models"] / "cn_canny.safetensors")
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="canny",
        reference_image="canny.png", model_name="cn_canny.safetensors",
    ))
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    classes = {n.get("class_type") for n in out.values()}
    assert "CannyEdgePreprocessor" in classes


def test_inject_strength_and_percent_pass_through(patched_dirs):
    cfg = _active_cfg(patched_dirs, strength=0.45, start_percent=0.1, end_percent=0.75)
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    apply_node = next(n for n in out.values() if n.get("class_type") == "ControlNetApplyAdvanced")
    assert apply_node["inputs"]["strength"] == 0.45
    assert apply_node["inputs"]["start_percent"] == 0.1
    assert apply_node["inputs"]["end_percent"] == 0.75


def test_inject_reference_path_is_absolute(patched_dirs):
    """LoadImage in ComfyUI resolves bare filenames against its own
    input/ directory. We use an absolute path so users don't have to
    copy files into ComfyUI's input folder."""
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    load = next(n for n in out.values() if n.get("class_type") == "LoadImage")
    image_path = load["inputs"]["image"]
    assert Path(image_path).is_absolute()
    assert Path(image_path).name == "pose.png"


def test_inject_model_name_passed_as_filename(patched_dirs):
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    loader = next(n for n in out.values() if n.get("class_type") == "ControlNetLoader")
    assert loader["inputs"]["control_net_name"] == "cn.safetensors"


def test_inject_respects_preexisting_applynode(patched_dirs):
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    # User's own pre-authored ControlNet
    wf["50"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {"positive": ["6", 0], "negative": ["7", 0]},
    }
    out = controlnet.inject_into_workflow(wf, cfg)
    # Exactly one ControlNetApplyAdvanced remains — the user's
    apply_ids = [nid for nid, n in out.items() if n.get("class_type") == "ControlNetApplyAdvanced"]
    assert apply_ids == ["50"]


def test_inject_skips_when_no_ksampler(patched_dirs):
    cfg = _active_cfg(patched_dirs)
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}
    out = controlnet.inject_into_workflow(wf, cfg)
    assert "ControlNetApplyAdvanced" not in {n.get("class_type") for n in out.values()}


def test_inject_allocates_fresh_node_ids(patched_dirs):
    """If the workflow already has nodes at IDs 20-24, we must pick
    higher ones so we don't clobber them."""
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    for occupied in ("20", "21", "22", "23"):
        wf[occupied] = {"class_type": "DummyNode", "inputs": {}}
    out = controlnet.inject_into_workflow(wf, cfg)
    # Our four new nodes must all be at IDs not in the occupied set
    new_ids = [
        nid for nid, n in out.items()
        if n.get("class_type") in ("LoadImage", "DWPreprocessor", "ControlNetLoader", "ControlNetApplyAdvanced")
    ]
    for nid in new_ids:
        assert nid not in ("20", "21", "22", "23")


def test_inject_preprocessor_carries_required_inputs(patched_dirs):
    """Audit fix: DWPreprocessor/DepthAnythingV2/Canny all have required
    inputs beyond `image`. Without them ComfyUI rejects the prompt."""
    cfg = _active_cfg(patched_dirs)  # pose → DWPreprocessor
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    prep = next(n for n in out.values() if n.get("class_type") == "DWPreprocessor")
    for required in ("detect_hand", "detect_body", "detect_face", "resolution"):
        assert required in prep["inputs"], f"DWPreprocessor missing {required}"


def test_inject_depth_preprocessor_carries_ckpt_name(patched_dirs):
    _mkimage(patched_dirs["ref"] / "depth.png")
    _mksafetensors(patched_dirs["models"] / "cn_d.safetensors")
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="depth",
        reference_image="depth.png", model_name="cn_d.safetensors",
    ))
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    prep = next(n for n in out.values() if n.get("class_type") == "DepthAnythingV2Preprocessor")
    assert "ckpt_name" in prep["inputs"]
    assert "resolution" in prep["inputs"]


def test_inject_canny_preprocessor_carries_thresholds(patched_dirs):
    _mkimage(patched_dirs["ref"] / "canny.png")
    _mksafetensors(patched_dirs["models"] / "cn_c.safetensors")
    cfg = _mkcfg(controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="canny",
        reference_image="canny.png", model_name="cn_c.safetensors",
    ))
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    prep = next(n for n in out.values() if n.get("class_type") == "CannyEdgePreprocessor")
    assert "low_threshold" in prep["inputs"]
    assert "high_threshold" in prep["inputs"]


def test_inject_rewires_every_sampler_in_refiner_pipeline(patched_dirs):
    """Audit fix: SDXL base + refiner workflow has TWO KSamplers.
    Both must be rewired through ControlNet — otherwise the refiner
    pass runs unconditioned and the composition drifts."""
    cfg = _active_cfg(patched_dirs)
    wf = _default_wf()
    # Add a refiner sampler
    wf["100"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": 1, "steps": 10, "cfg": 7.0,
            "sampler_name": "euler", "scheduler": "karras", "denoise": 0.4,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["3", 0],
        },
    }
    out = controlnet.inject_into_workflow(wf, cfg)
    apply_id = next(nid for nid, n in out.items() if n.get("class_type") == "ControlNetApplyAdvanced")
    # Both samplers should end up reading from the ControlNet wrap
    for sid in ("3", "100"):
        assert out[sid]["inputs"]["positive"] == [apply_id, 0]
        assert out[sid]["inputs"]["negative"] == [apply_id, 1]


@pytest.mark.parametrize("bad_preproc", [
    "OpenPosePreprocessor",
    "OpenposePreprocessor_Preview",
    "openpose_full",
    "AnimalPosePreprocessor",
    "DenseposePreprocessor",
    "densepose_estimator",
])
def test_commercial_gate_catches_all_openpose_family(bad_preproc):
    """Audit fix: substring-match so any OpenPose / AnimalPose /
    DensePose variant is caught, not just the bare string."""
    with pytest.raises(Exception):
        _mkcfg(
            commercial=True,
            controlnet=cfg_mod.ControlNetConfig(
                enabled=True, mode="pose", preprocessor_override=bad_preproc,
            ),
        )


def test_commercial_gate_allows_dwpose_variants():
    """DWPose shares no substring with the blocklist — must pass."""
    for good in ("DWPreprocessor", "DWPose", "DWpose_estimator"):
        cfg = _mkcfg(
            commercial=True,
            controlnet=cfg_mod.ControlNetConfig(
                enabled=True, mode="pose", preprocessor_override=good,
            ),
        )
        assert cfg.controlnet.preprocessor_override == good


def test_atomic_copy_cleans_staging_on_failure(patched_dirs, tmp_path, monkeypatch):
    """_atomic_copy must not leave a .part file behind if os.replace fails."""
    src = tmp_path / "src.safetensors"
    import json as _json
    import struct as _struct
    header = _json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    src.write_bytes(_struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    dest = patched_dirs["models"] / "x.safetensors"

    def boom(src_p, dst_p):
        raise OSError("fake rename failure")

    monkeypatch.setattr("src.plugins.controlnet.os.replace", boom)
    with pytest.raises(OSError):
        controlnet._atomic_copy(src, dest)
    # No .part leftovers
    leftovers = list(patched_dirs["models"].glob(".*part*"))
    assert leftovers == []


def test_inject_runtime_block_on_commercial_openpose(patched_dirs, caplog):
    """Even if preprocessor_override smuggled OpenPose past a stale
    validator, the injection codepath must refuse under commercial=True."""
    _mkimage(patched_dirs["ref"] / "pose.png")
    _mksafetensors(patched_dirs["models"] / "cn.safetensors")
    # Build through noncommercial, then flip commercial on via model_copy
    cfg_nc = _mkcfg(commercial=False, controlnet=cfg_mod.ControlNetConfig(
        enabled=True, mode="pose",
        reference_image="pose.png", model_name="cn.safetensors",
        preprocessor_override="OpenposePreprocessor",
    ))
    cfg = cfg_nc.model_copy(update={"commercial": True})
    wf = _default_wf()
    out = controlnet.inject_into_workflow(wf, cfg)
    # Must have refused — no ControlNet nodes injected
    assert "ControlNetApplyAdvanced" not in {n.get("class_type") for n in out.values()}


# ─── integration with lora (both active) ───
def test_lora_and_controlnet_stack_cleanly(patched_dirs, monkeypatch, tmp_path: Path):
    """When both LoRA and ControlNet are active, the LoRA touches
    model/clip (pre-sampler), ControlNet wraps positive/negative
    conditioning (post-CLIPTextEncode, pre-sampler). Both can live in
    the same workflow without collision."""
    from src.plugins import lora as lora_mod

    # Stub LoRA paths
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "LORAS_DIR", loras_dir)
    monkeypatch.setattr(lora_mod, "LORAS_DIR", loras_dir)
    _mksafetensors(loras_dir / "brand.safetensors")

    # Enable both
    cfg = _active_cfg(
        patched_dirs,
    ).model_copy(update={
        "commercial": False,
        "lora": cfg_mod.LoRAConfig(
            enabled=True, name="brand", trigger_word="mascxyz",
        ),
    })

    wf = _default_wf()
    wf = lora_mod.inject_into_workflow(wf, cfg)
    wf = controlnet.inject_into_workflow(wf, cfg)

    classes = {n.get("class_type") for n in wf.values()}
    assert "LoraLoader" in classes
    assert "ControlNetApplyAdvanced" in classes
    # KSampler must end up reading positive/negative from ControlNet,
    # and model from the LoRA — orthogonal compositions.
    ks = next(n for n in wf.values() if n.get("class_type") == "KSampler")
    apply_id = next(nid for nid, n in wf.items() if n.get("class_type") == "ControlNetApplyAdvanced")
    lora_id = next(nid for nid, n in wf.items() if n.get("class_type") == "LoraLoader")
    assert ks["inputs"]["positive"] == [apply_id, 0]
    assert ks["inputs"]["negative"] == [apply_id, 1]
    assert ks["inputs"]["model"] == [lora_id, 0]
