#!/usr/bin/env python3
"""Quick mic level test for debugging MacBook capture."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_listener import _resample_for_whisper
from bluetooth_audio import describe_input_choice, pick_input_device


def main() -> int:
    device = pick_input_device(prefer_bluetooth=False)
    name = device.name
    rate = device.sample_rate
    print(f"Device: {name}")
    print(f"Rate: {rate} Hz")
    print("Speak for 3 seconds...")

    samples: list[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print("status:", status)
        samples.append(indata[:, 0].copy())

    chunk = int(rate * 0.05)
    with sd.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        blocksize=chunk,
        device=device.index,
        callback=callback,
    ):
        time.sleep(3)

    audio = np.concatenate(samples)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    resampled = _resample_for_whisper(audio, rate)
    print(f"Captured {audio.size} samples ({audio.size / rate:.2f}s)")
    print(f"RMS: {rms:.6f}  Peak: {peak:.6f}")
    print(f"Whisper-ready samples: {resampled.size}")

    if rms < 0.0005:
        print("FAIL: almost no signal — check mic permission or input device.")
        return 1
    print("PASS: microphone is receiving audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
