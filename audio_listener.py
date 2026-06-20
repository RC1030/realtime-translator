"""Microphone capture with end-of-utterance detection."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from config import (
    CALIBRATION_SECONDS,
    CHUNK_SECONDS,
    INPUT_DEVICE,
    MIN_RMS_THRESHOLD,
    MIN_SPEECH_SECONDS,
    NOISE_MULTIPLIER,
    PARTIAL_TRANSCRIBE_SECONDS,
    SILENCE_SECONDS,
    WHISPER_SAMPLE_RATE,
)


def _pick_input_device() -> tuple[int | None, int]:
    """Return (device index, sample rate). Prefer the Mac built-in microphone."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]

    if INPUT_DEVICE:
        needle = INPUT_DEVICE.lower()
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
                rate = int(dev["default_samplerate"])
                return idx, rate

    preferred_names = ("macbook", "built-in", "internal")
    for idx, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and any(token in name for token in preferred_names):
            rate = int(dev["default_samplerate"])
            return idx, rate

    if default_in is not None and default_in >= 0:
        dev = devices[default_in]
        return default_in, int(dev["default_samplerate"])

    return None, 48_000


def _resample_for_whisper(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == WHISPER_SAMPLE_RATE or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    target_len = max(1, int(round(audio.size * WHISPER_SAMPLE_RATE / source_rate)))
    source_idx = np.arange(audio.size, dtype=np.float64)
    target_idx = np.linspace(0, audio.size - 1, target_len)
    return np.interp(target_idx, source_idx, audio).astype(np.float32)


class AudioListener:
    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        on_status: Callable[[str], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_progress: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.on_utterance = on_utterance
        self.on_status = on_status or (lambda _msg: None)
        self.on_level = on_level or (lambda _level: None)
        self.on_speech_start = on_speech_start or (lambda: None)
        self.on_progress = on_progress or (lambda _audio: None)
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._reset_state = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        self._utterance_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._capture_rate = 48_000
        self._device_index: int | None = None
        self._speech_threshold = MIN_RMS_THRESHOLD

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._pause.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-listener")
        self._thread.start()

    def pause(self) -> None:
        self._pause.set()
        self._reset_state.set()

    def resume(self) -> None:
        while True:
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                break
        self._pause.clear()
        self._reset_state.set()
        self.on_status("Listening...")

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def _run(self) -> None:
        self._device_index, self._capture_rate = _pick_input_device()
        chunk_size = max(256, int(self._capture_rate * CHUNK_SECONDS))
        chunk_seconds = chunk_size / self._capture_rate

        device_name = "default"
        if self._device_index is not None:
            device_name = sd.query_devices(self._device_index)["name"]

        self.on_status(f"Mic: {device_name} @ {self._capture_rate} Hz")

        buffer: list[np.ndarray] = []
        speech_started = False
        silence_elapsed = 0.0
        speech_seconds = 0.0
        progress_elapsed = 0.0
        calibration_samples: list[float] = []
        calibrated = False
        calibration_target = max(5, int(CALIBRATION_SECONDS / chunk_seconds))

        def callback(indata, _frames, _time, status):
            nonlocal speech_started, silence_elapsed, speech_seconds, buffer, calibrated, progress_elapsed

            if status:
                self.on_status(f"Audio status: {status}")
            if self._stop.is_set() or self._pause.is_set():
                return

            if self._reset_state.is_set():
                buffer = []
                speech_started = False
                silence_elapsed = 0.0
                speech_seconds = 0.0
                progress_elapsed = 0.0
                self._reset_state.clear()

            chunk = indata[:, 0].astype(np.float32, copy=False)
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            self.on_level(min(1.0, rms / max(self._speech_threshold * 4, 0.01)))

            if not calibrated:
                calibration_samples.append(rms)
                if len(calibration_samples) >= calibration_target:
                    noise_floor = float(np.percentile(calibration_samples, 75))
                    self._speech_threshold = max(MIN_RMS_THRESHOLD, noise_floor * NOISE_MULTIPLIER)
                    calibrated = True
                    self.on_status("Ready. Listening...")
                return

            if rms >= self._speech_threshold:
                if not speech_started:
                    speech_started = True
                    buffer = []
                    progress_elapsed = 0.0
                    self.on_speech_start()
                buffer.append(chunk.copy())
                speech_seconds += chunk_seconds
                silence_elapsed = 0.0
                progress_elapsed += chunk_seconds
                if progress_elapsed >= PARTIAL_TRANSCRIBE_SECONDS and speech_seconds >= MIN_SPEECH_SECONDS:
                    progress_elapsed = 0.0
                    audio = np.concatenate(buffer)
                    self.on_progress(_resample_for_whisper(audio, self._capture_rate))
                return

            if not speech_started:
                return

            buffer.append(chunk.copy())
            silence_elapsed += chunk_seconds
            if silence_elapsed >= SILENCE_SECONDS and speech_seconds >= MIN_SPEECH_SECONDS:
                audio = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
                speech_started = False
                silence_elapsed = 0.0
                speech_seconds = 0.0
                progress_elapsed = 0.0
                buffer = []
                self._pause.set()
                resampled = _resample_for_whisper(audio, self._capture_rate)
                try:
                    self._utterance_queue.put_nowait(resampled)
                except queue.Full:
                    pass

        try:
            stream = sd.InputStream(
                samplerate=self._capture_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                latency="high",
                device=self._device_index,
                callback=callback,
            )
        except Exception as exc:  # noqa: BLE001
            self.on_status(f"Microphone error: {exc}")
            return

        with stream:
            self._stream = stream
            while not self._stop.is_set():
                try:
                    audio = self._utterance_queue.get_nowait()
                except queue.Empty:
                    sd.sleep(50)
                    continue

                self.on_status("Finalizing speech...")
                self.on_utterance(audio)
                sd.sleep(50)

        self._stream = None
