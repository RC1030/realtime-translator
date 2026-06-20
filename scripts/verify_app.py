#!/usr/bin/env python3
"""Headless verification of translator pipeline components."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import WHISPER_SAMPLE_RATE
from speech_to_text import SpeechToText
from translator import translate


def synth_tone(seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(WHISPER_SAMPLE_RATE * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    print("=== Component verification ===")

    print("\n1) LM Studio translation")
    result = translate("Good morning.", "English", "Chinese (Mandarin)")
    print(f"   Input: Good morning.")
    print(f"   Output: {result}")
    if not result:
        print("   FAIL: empty translation")
        return 1
    print("   PASS")

    print("\n2) Whisper model load + empty audio handling")
    stt = SpeechToText()
    empty = stt.transcribe(np.array([], dtype=np.float32), "en")
    print(f"   Empty audio -> '{empty}'")
    if empty != "":
        print("   FAIL: expected empty string")
        return 1
    print("   PASS")

    print("\n3) Audio listener import + RMS path")
    from audio_listener import AudioListener

    captured: list[np.ndarray] = []

    def on_utterance(audio: np.ndarray) -> None:
        captured.append(audio)

    listener = AudioListener(on_utterance=on_utterance)
    listener.start()
    listener.pause()
    listener.stop()
    print("   PASS (listener starts/stops cleanly)")

    print("\n4) Tk app module import")
    import app as app_module

    print(f"   TranslatorApp: {app_module.TranslatorApp}")
    print("   PASS")

    print("\nAll automated checks passed.")
    print("Manual GUI test: speak, wait 2s silence, click Continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
