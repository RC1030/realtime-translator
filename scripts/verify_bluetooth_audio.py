#!/usr/bin/env python3
"""Simulate and verify Bluetooth microphone source selection."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_listener import AudioListener
import bluetooth_audio as bt


class BluetoothAudioTests(unittest.TestCase):
    def test_parse_connected_bluetooth_devices(self) -> None:
        sample = {
            "SPBluetoothDataType": [
                {
                    "device_connected": [
                        {
                            "RC APPro3 - Find My": {
                                "device_minorType": "Headphones",
                                "device_address": "74:77:86:52:B5:93",
                            }
                        },
                        {
                            "Magic Keyboard": {
                                "device_minorType": "Keyboard",
                                "device_address": "9C:58:3C:E9:8A:06",
                            }
                        },
                    ]
                }
            ]
        }
        with patch.object(bt.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = __import__("json").dumps(sample)
            devices = bt.get_connected_bluetooth_devices()
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].name, "RC APPro3 - Find My")
        self.assertEqual(devices[0].minor_type, "Headphones")

    def test_audio_capable_filter_excludes_keyboard(self) -> None:
        devices = [
            bt.BluetoothDevice("RC APPro3 - Find My", "Headphones"),
            bt.BluetoothDevice("Magic Keyboard", "Keyboard"),
        ]
        with patch.object(bt, "get_connected_bluetooth_devices", return_value=devices):
            audio_devices = bt.get_audio_capable_bluetooth_devices()
        self.assertEqual([device.name for device in audio_devices], ["RC APPro3 - Find My"])

    def test_bluetooth_candidate_matching(self) -> None:
        bt_devices = [bt.BluetoothDevice("RC APPro3 - Find My", "Headphones")]
        inputs = [
            {"index": 0, "name": "RC APPro3 - Find My", "sample_rate": 24000},
            {"index": 2, "name": "MacBook Pro Microphone", "sample_rate": 48000},
        ]
        with patch.object(bt, "get_audio_capable_bluetooth_devices", return_value=bt_devices):
            with patch.object(bt, "list_input_devices", return_value=inputs):
                candidates = bt.get_bluetooth_input_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["index"], 0)
        self.assertEqual(candidates[0]["bluetooth_name"], "RC APPro3 - Find My")

    def test_pick_builtin_excludes_bluetooth_candidate(self) -> None:
        bt_devices = [bt.BluetoothDevice("RC APPro3 - Find My", "Headphones")]
        inputs = [
            {"index": 0, "name": "RC APPro3 - Find My", "sample_rate": 24000},
            {"index": 2, "name": "MacBook Pro Microphone", "sample_rate": 48000},
        ]
        with patch.object(bt, "INPUT_DEVICE", ""):
            with patch.object(bt, "get_audio_capable_bluetooth_devices", return_value=bt_devices):
                with patch.object(bt, "list_input_devices", return_value=inputs):
                    builtin = bt.pick_input_device(prefer_bluetooth=False)
                    bluetooth = bt.pick_input_device(prefer_bluetooth=True)
        self.assertEqual(builtin.source, "builtin")
        self.assertEqual(builtin.index, 2)
        self.assertEqual(bluetooth.source, "bluetooth")
        self.assertEqual(bluetooth.index, 0)
        self.assertNotEqual(builtin.index, bluetooth.index)


def simulate_live_selection() -> int:
    print("=== Live Bluetooth audio simulation ===")

    connected = bt.get_connected_bluetooth_devices()
    print(f"\nConnected Bluetooth devices: {len(connected)}")
    for device in connected:
        print(f"  - {device.name} ({device.minor_type or 'unknown type'})")

    candidates = bt.get_bluetooth_input_candidates()
    print(f"\nBluetooth microphone candidates: {len(candidates)}")
    for candidate in candidates:
        print(
            f"  - input[{candidate['index']}] {candidate['name']} "
            f"<= {candidate['bluetooth_name']}"
        )

    scenarios = [
        ("User accepts Bluetooth input", True),
        ("User keeps built-in microphone", False),
    ]
    for label, prefer_bluetooth in scenarios:
        choice = bt.pick_input_device(prefer_bluetooth=prefer_bluetooth)
        print(f"\n{label}")
        print(f"  Selected: {bt.describe_input_choice(choice)}")
        if prefer_bluetooth and candidates and choice.source != "bluetooth":
            print("  FAIL: expected Bluetooth source")
            return 1
        if not prefer_bluetooth and choice.source == "bluetooth":
            print("  FAIL: built-in path selected a Bluetooth device")
            return 1
        print("  PASS")

    print("\n=== AudioListener stream simulation ===")
    for prefer_bluetooth, label in ((True, "bluetooth"), (False, "builtin")):
        statuses: list[str] = []

        def on_status(msg: str) -> None:
            statuses.append(msg)

        listener = AudioListener(
            on_utterance=lambda _audio: None,
            on_status=on_status,
            prefer_bluetooth=prefer_bluetooth,
        )
        listener.start()
        time.sleep(1.2)
        listener.stop()
        mic_status = next((msg for msg in statuses if msg.startswith("Mic:")), "")
        print(f"{label}: {mic_status or statuses[:3]}")
        if not mic_status:
            print("  FAIL: listener did not report microphone selection")
            return 1
        if prefer_bluetooth and candidates and "(bluetooth)" not in mic_status:
            print("  FAIL: Bluetooth path did not open the Bluetooth microphone")
            return 1
        if not prefer_bluetooth and "(builtin)" not in mic_status and "(default)" not in mic_status:
            print("  FAIL: built-in path did not avoid Bluetooth selection label")
            return 1
        print("  PASS")

    if candidates:
        print("\n=== Exclusive source check ===")
        builtin = bt.pick_input_device(prefer_bluetooth=False)
        bluetooth = bt.pick_input_device(prefer_bluetooth=True)
        print(f"Built-in path: {bt.describe_input_choice(builtin)}")
        print(f"Bluetooth path: {bt.describe_input_choice(bluetooth)}")
        if builtin.index == bluetooth.index:
            print("FAIL: both paths selected the same input device")
            return 1
        print("PASS: macOS mic and Bluetooth mic resolve to different input endpoints")

    print("\nAll Bluetooth audio simulations passed.")
    return 0


def main() -> int:
    print("=== Unit tests ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BluetoothAudioTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    try:
        return simulate_live_selection()
    except sd.PortAudioError as exc:
        print(f"PortAudio unavailable for live simulation: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
