"""Application configuration."""

from __future__ import annotations

import os

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")

WHISPER_SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.05
SILENCE_SECONDS = 2.0
MIN_SPEECH_SECONDS = 0.35
MIN_RMS_THRESHOLD = 0.0015
NOISE_MULTIPLIER = 2.2
CALIBRATION_SECONDS = 1.5
PARTIAL_TRANSCRIBE_SECONDS = 1.2
TRANSLATE_DEBOUNCE_MS = 500
MIN_TRANSLATE_INTERVAL_SECONDS = 1.5

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
INPUT_DEVICE = os.getenv("INPUT_DEVICE")  # optional sounddevice name substring

LANGUAGES: dict[str, str] = {
    "English": "en",
    "Chinese (Mandarin)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Mongolian": "mn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Hindi": "hi",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
}

DEFAULT_SOURCE = "English"
DEFAULT_TARGET = "Chinese (Mandarin)"
