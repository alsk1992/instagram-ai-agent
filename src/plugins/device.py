"""Persistent device fingerprint — generated once, never rotated.

Instagram pins the device+user pair; changing UUIDs triggers challenges.
"""
from __future__ import annotations

import json
import secrets
import uuid
from pathlib import Path
from typing import Any

from src.core.config import DEVICE_PATH

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


def load_or_create(path: Path | None = None) -> dict[str, Any]:
    p = path or DEVICE_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    settings = {**_DEFAULT_DEVICE, **_new_uuids()}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return settings


def apply_to(cl: Any) -> dict[str, Any]:
    """Apply persistent device settings to an instagrapi Client."""
    settings = load_or_create()
    device = {k: settings[k] for k in _DEFAULT_DEVICE.keys() if k in settings}
    uuids = {k: settings[k] for k in ("phone_id", "uuid", "client_session_id", "advertising_id", "device_id")}
    cl.set_device(device)
    cl.set_uuids(uuids)
    # user_agent will be derived by instagrapi from the device dict
    return settings
