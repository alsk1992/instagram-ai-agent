"""Persistent device fingerprint — generated once, never rotated.

Instagram pins the device+user pair; changing UUIDs triggers challenges.

Supports importing a seller-provided bundle via ``IG_DEVICE_IMPORT_PATH``
env var — critical for aged/bought accounts where the seller exported
the device off the account's original phone. Regenerating fresh UUIDs
breaks the ig_did / device-trust lineage IG built over months and
triggers a "device reset" review within 24-72h. Importing the seller's
bundle preserves that lineage.
"""
from __future__ import annotations

import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from instagram_ai_agent.core.config import DEVICE_PATH
from instagram_ai_agent.core.logging_setup import get_logger

log = get_logger(__name__)

# A realistic mid-range Android device profile. Pinned once per install.
_DEFAULT_DEVICE: dict[str, Any] = {
    "app_version": "302.0.0.23.114",
    "android_version": 30,
    "android_release": "11",
    "dpi": "420dpi",
    "resolution": "1080x2220",
    "manufacturer": "samsung",
    "device": "SM-A525F",
    "model": "a52q",
    "cpu": "qcom",
    "version_code": "521498971",
}


def _new_uuids() -> dict[str, str]:
    return {
        "phone_id": str(uuid.uuid4()),
        "uuid": str(uuid.uuid4()),
        "client_session_id": str(uuid.uuid4()),
        "advertising_id": str(uuid.uuid4()),
        "device_id": "android-" + secrets.token_hex(8),
    }


_UUID_KEYS = ("phone_id", "uuid", "client_session_id", "advertising_id", "device_id")


def _import_from_bundle(bundle_path: Path) -> dict[str, Any] | None:
    """Parse a seller-supplied bundle file. Accepts two shapes:

    * Raw device.json from instagrapi/our own format: flat dict with
      phone_id / uuid / device_id / etc. at the top level.
    * Full ``dump_settings()`` output: nested dict with top-level
      ``uuids`` sub-dict + ``device_settings`` sub-dict. We flatten both
      into our canonical shape.
    """
    if not bundle_path.exists():
        log.warning("device import: bundle not found at %s", bundle_path)
        return None
    try:
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("device import: couldn't parse %s — %s", bundle_path, e)
        return None

    if not isinstance(raw, dict):
        log.warning("device import: expected JSON object, got %s", type(raw).__name__)
        return None

    out: dict[str, Any] = dict(_DEFAULT_DEVICE)
    # Nested dump_settings shape
    if "uuids" in raw and isinstance(raw["uuids"], dict):
        for k in _UUID_KEYS:
            if raw["uuids"].get(k):
                out[k] = raw["uuids"][k]
    if "device_settings" in raw and isinstance(raw["device_settings"], dict):
        for k in _DEFAULT_DEVICE.keys():
            if raw["device_settings"].get(k):
                out[k] = raw["device_settings"][k]
    # Flat shape (our own device.json or bundler output)
    for k in (*_UUID_KEYS, *_DEFAULT_DEVICE.keys()):
        if k in raw and raw[k]:
            out[k] = raw[k]

    # Ensure all required UUID keys exist — fill any missing ones with
    # fresh UUIDs but LOG them prominently since partial imports are a
    # bigger risk than a full fresh generation.
    filled: list[str] = []
    generated_fresh = _new_uuids()
    for k in _UUID_KEYS:
        if k not in out or not out[k]:
            out[k] = generated_fresh[k]
            filled.append(k)
    if filled:
        log.warning(
            "device import: bundle missing %s — filled with fresh UUIDs. "
            "Risk: partial-device-match is detectable by Meta's fingerprint "
            "correlator. Get the full bundle from the seller if possible.",
            ", ".join(filled),
        )
    return out


def load_or_create(path: Path | None = None) -> dict[str, Any]:
    p = path or DEVICE_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    # First run: check for seller-supplied bundle. If present, import it;
    # otherwise generate fresh UUIDs (the fresh-account path).
    import_path = os.environ.get("IG_DEVICE_IMPORT_PATH", "").strip()
    if import_path:
        imported = _import_from_bundle(Path(import_path))
        if imported is not None:
            settings = imported
            log.info(
                "device import: loaded seller bundle from %s — "
                "preserving ig_did/device_id lineage (aged-account safe path)",
                import_path,
            )
        else:
            log.warning(
                "device import: bundle load failed — falling back to fresh UUIDs. "
                "This WILL trigger 'device reset' review on an aged account."
            )
            settings = {**_DEFAULT_DEVICE, **_new_uuids()}
    else:
        settings = {**_DEFAULT_DEVICE, **_new_uuids()}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return settings


def apply_to(cl: Any) -> dict[str, Any]:
    """Apply persistent device settings to an instagrapi Client."""
    settings = load_or_create()
    device = {k: settings[k] for k in _DEFAULT_DEVICE.keys() if k in settings}
    uuids = {k: settings[k] for k in _UUID_KEYS}
    cl.set_device(device)
    cl.set_uuids(uuids)
    # user_agent will be derived by instagrapi from the device dict
    return settings
