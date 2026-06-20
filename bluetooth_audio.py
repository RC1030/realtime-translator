"""Bluetooth audio input detection and device selection on macOS."""

from __future__ import annotations

import json
import subprocess
import unicodedata
from dataclasses import dataclass
from typing import Any

import sounddevice as sd

from config import INPUT_DEVICE

AUDIO_CAPABLE_MINOR_TYPES = {
    "headphones",
    "headset",
    "handsfree",
    "handset",
    "headphone",
    "car audio",
}

BUILTIN_NAME_TOKENS = ("macbook", "built-in", "internal")
BLUETOOTH_NAME_TOKENS = ("bluetooth", "airpods", "beats", "bose", "sony", "jabra", "buds")


@dataclass(frozen=True)
class BluetoothDevice:
    name: str
    minor_type: str
    address: str = ""


@dataclass(frozen=True)
class InputDeviceChoice:
    index: int | None
    sample_rate: int
    name: str
    source: str  # "bluetooth" | "builtin" | "configured" | "default"


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().lower()


def _names_match(left: str, right: str) -> bool:
    a = _normalize_name(left)
    b = _normalize_name(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _is_builtin_device_name(name: str) -> bool:
    lowered = _normalize_name(name)
    return any(token in lowered for token in BUILTIN_NAME_TOKENS)


def _looks_like_bluetooth_device_name(name: str) -> bool:
    lowered = _normalize_name(name)
    return any(token in lowered for token in BLUETOOTH_NAME_TOKENS)


def get_connected_bluetooth_devices() -> list[BluetoothDevice]:
    """Return Bluetooth devices currently connected to macOS."""
    try:
        proc = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    connected: list[BluetoothDevice] = []
    for block in payload.get("SPBluetoothDataType", []):
        if not isinstance(block, dict):
            continue
        for entry in block.get("device_connected", []):
            if not isinstance(entry, dict):
                continue
            for device_name, info in entry.items():
                if not isinstance(info, dict):
                    continue
                minor_type = str(info.get("device_minorType", "")).strip()
                connected.append(
                    BluetoothDevice(
                        name=str(device_name),
                        minor_type=minor_type,
                        address=str(info.get("device_address", "")),
                    )
                )
    return connected


def get_audio_capable_bluetooth_devices() -> list[BluetoothDevice]:
    """Connected Bluetooth devices that may expose a microphone."""
    devices = get_connected_bluetooth_devices()
    audio_devices = [
        device
        for device in devices
        if _normalize_name(device.minor_type) in AUDIO_CAPABLE_MINOR_TYPES
    ]
    if audio_devices:
        return audio_devices

    # Fall back to non-peripheral connected devices when minor type is missing.
    return [
        device
        for device in devices
        if _normalize_name(device.minor_type) not in {"keyboard", "mouse", "trackpad"}
        and (
            _looks_like_bluetooth_device_name(device.name)
            or bool(device.minor_type.strip())
        )
    ]


def list_input_devices() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    inputs: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            inputs.append(
                {
                    "index": index,
                    "name": str(device["name"]),
                    "sample_rate": int(device["default_samplerate"]),
                }
            )
    return inputs


def get_bluetooth_input_candidates() -> list[dict[str, Any]]:
    """Map connected Bluetooth audio devices to sounddevice input endpoints."""
    bt_devices = get_audio_capable_bluetooth_devices()
    if not bt_devices:
        return []

    candidates: list[dict[str, Any]] = []
    for input_device in list_input_devices():
        name = input_device["name"]
        if _is_builtin_device_name(name):
            continue

        matched_bt = next(
            (bt for bt in bt_devices if _names_match(name, bt.name)),
            None,
        )
        if matched_bt or _looks_like_bluetooth_device_name(name):
            candidates.append(
                {
                    "index": input_device["index"],
                    "name": name,
                    "sample_rate": input_device["sample_rate"],
                    "bluetooth_name": matched_bt.name if matched_bt else name,
                    "minor_type": matched_bt.minor_type if matched_bt else "",
                }
            )
    return candidates


def has_bluetooth_audio_input() -> bool:
    return bool(get_bluetooth_input_candidates())


def _pick_configured_device() -> InputDeviceChoice | None:
    if not INPUT_DEVICE:
        return None

    needle = INPUT_DEVICE.lower()
    for input_device in list_input_devices():
        if needle in input_device["name"].lower():
            return InputDeviceChoice(
                index=input_device["index"],
                sample_rate=input_device["sample_rate"],
                name=input_device["name"],
                source="configured",
            )
    return None


def _pick_builtin_input_device() -> InputDeviceChoice:
    configured = _pick_configured_device()
    if configured and _is_builtin_device_name(configured.name):
        return configured

    for input_device in list_input_devices():
        if _is_builtin_device_name(input_device["name"]):
            return InputDeviceChoice(
                index=input_device["index"],
                sample_rate=input_device["sample_rate"],
                name=input_device["name"],
                source="builtin",
            )

    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        device = sd.query_devices(default_in)
        return InputDeviceChoice(
            index=default_in,
            sample_rate=int(device["default_samplerate"]),
            name=str(device["name"]),
            source="default",
        )

    return InputDeviceChoice(index=None, sample_rate=48_000, name="default", source="default")


def _pick_bluetooth_input_device() -> InputDeviceChoice | None:
    candidates = get_bluetooth_input_candidates()
    if not candidates:
        return None

    chosen = candidates[0]
    return InputDeviceChoice(
        index=chosen["index"],
        sample_rate=chosen["sample_rate"],
        name=chosen["name"],
        source="bluetooth",
    )


def pick_input_device(*, prefer_bluetooth: bool) -> InputDeviceChoice:
    """Select the microphone source based on user preference."""
    configured = _pick_configured_device()
    if configured:
        return configured

    if prefer_bluetooth:
        bluetooth = _pick_bluetooth_input_device()
        if bluetooth is not None:
            return bluetooth

    return _pick_builtin_input_device()


def describe_input_choice(choice: InputDeviceChoice) -> str:
    return f"{choice.name} @ {choice.sample_rate} Hz ({choice.source})"
