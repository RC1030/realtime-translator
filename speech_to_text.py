"""Local speech-to-text using faster-whisper."""

from __future__ import annotations

import threading

import numpy as np

from config import WHISPER_MODEL


class SpeechToText:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        self._model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio: np.ndarray, language_code: str | None) -> str:
        if audio.size == 0:
            return ""

        model = self._load_model()
        kwargs: dict = {
            "beam_size": 1,
            "vad_filter": False,
        }
        if language_code:
            kwargs["language"] = language_code

        with self._lock:
            segments, _info = model.transcribe(audio, **kwargs)

        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()
