#!/usr/bin/env python3
"""Real-time local translator: macOS mic -> Whisper -> LM Studio -> Tk UI."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from audio_listener import AudioListener
from config import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    LANGUAGES,
    MIN_TRANSLATE_INTERVAL_SECONDS,
    TRANSLATE_DEBOUNCE_MS,
)
from speech_to_text import SpeechToText
from translator import TranslationError, translate


class TranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Local Translator")
        self.root.minsize(420, 240)
        self.root.geometry("620x320")
        self.root.resizable(True, True)

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stt = SpeechToText()
        self.listener = AudioListener(
            on_utterance=self._on_utterance_captured,
            on_status=lambda msg: self.event_queue.put(("status", msg)),
            on_level=lambda level: self.event_queue.put(("level", level)),
            on_speech_start=lambda: self.event_queue.put(("speech_start", None)),
            on_progress=lambda audio: self.event_queue.put(("partial", audio.copy())),
        )

        self.models_ready = False
        self.transcribing = False
        self.translating = False
        self.awaiting_next = False
        self.auto_update_original = True
        self._partial_busy = False
        self._translate_timer: str | None = None
        self._translate_target_text = ""
        self._translate_pending_text = ""
        self._last_translate_at = 0.0
        self._last_translated_source = ""

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)
        threading.Thread(target=self._preload_models, daemon=True, name="model-preload").start()
        self.listener.start()

    def _preload_models(self) -> None:
        self.event_queue.put(("status", "Loading speech model..."))
        try:
            self.stt._load_model()
            self.models_ready = True
        except Exception as exc:  # noqa: BLE001
            self.event_queue.put(("error", str(exc)))

    def _build_ui(self) -> None:
        pad = 6
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.root, padding=pad)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="From").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.source_var = tk.StringVar(value=DEFAULT_SOURCE)
        ttk.Combobox(
            controls,
            textvariable=self.source_var,
            values=list(LANGUAGES.keys()),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="To").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.target_var = tk.StringVar(value=DEFAULT_TARGET)
        ttk.Combobox(
            controls,
            textvariable=self.target_var,
            values=list(LANGUAGES.keys()),
            state="readonly",
            width=12,
        ).grid(row=0, column=3, sticky="ew", padx=(0, 8))

        self.always_on_top = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Top",
            variable=self.always_on_top,
            command=self._toggle_always_on_top,
        ).grid(row=0, column=4, sticky="e")

        meter_row = ttk.Frame(self.root, padding=(pad, 0, pad, 0))
        meter_row.grid(row=1, column=0, sticky="ew")
        meter_row.columnconfigure(1, weight=1)
        ttk.Label(meter_row, text="Mic").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.level_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            meter_row,
            variable=self.level_var,
            maximum=100,
            mode="determinate",
            length=120,
        ).grid(row=0, column=1, sticky="ew")

        columns = ttk.Frame(self.root, padding=(pad, 4, pad, 0))
        columns.grid(row=2, column=0, sticky="nsew")
        columns.columnconfigure(0, weight=1, uniform="textcols")
        columns.columnconfigure(1, weight=1, uniform="textcols")
        columns.rowconfigure(1, weight=1)

        ttk.Label(columns, text="Original (editable)").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Label(columns, text="Translation").grid(row=0, column=1, sticky="w", padx=(4, 0))

        original_frame = ttk.Frame(columns)
        original_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)

        self.original_text = tk.Text(
            original_frame,
            wrap="word",
            font=("Helvetica", 12),
            height=6,
            padx=6,
            pady=6,
        )
        self.original_text.grid(row=0, column=0, sticky="nsew")
        self.original_text.bind("<KeyRelease>", self._on_original_edited)
        orig_scroll = ttk.Scrollbar(original_frame, orient="vertical", command=self.original_text.yview)
        orig_scroll.grid(row=0, column=1, sticky="ns")
        self.original_text.configure(yscrollcommand=orig_scroll.set)

        translation_frame = ttk.Frame(columns)
        translation_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        translation_frame.columnconfigure(0, weight=1)
        translation_frame.rowconfigure(0, weight=1)

        self.translation_text = tk.Text(
            translation_frame,
            wrap="word",
            font=("Helvetica", 12),
            height=6,
            padx=6,
            pady=6,
            foreground="#0b5cab",
            state="disabled",
        )
        self.translation_text.grid(row=0, column=0, sticky="nsew")
        trans_scroll = ttk.Scrollbar(
            translation_frame, orient="vertical", command=self.translation_text.yview
        )
        trans_scroll.grid(row=0, column=1, sticky="ns")
        self.translation_text.configure(yscrollcommand=trans_scroll.set)

        footer = ttk.Frame(self.root, padding=(pad, 4, pad, pad))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Starting...")
        ttk.Label(footer, textvariable=self.status_var, font=("Helvetica", 10)).grid(
            row=0, column=0, sticky="w"
        )

        self.continue_btn = ttk.Button(
            footer,
            text="Continue",
            command=self._on_continue,
            state="disabled",
        )
        self.continue_btn.grid(row=0, column=1, sticky="e")

    def _toggle_always_on_top(self) -> None:
        self.root.attributes("-topmost", self.always_on_top.get())

    def _on_original_edited(self, _event: tk.Event) -> None:
        self.auto_update_original = False
        text = self.original_text.get("1.0", "end").strip()
        self._schedule_translation(text, debounce_ms=TRANSLATE_DEBOUNCE_MS)

    def _set_original_text(self, text: str) -> None:
        if not self.auto_update_original:
            return
        self.original_text.delete("1.0", "end")
        self.original_text.insert("1.0", text)

    def _clear_translation(self) -> None:
        self.translation_text.configure(state="normal")
        self.translation_text.delete("1.0", "end")
        self.translation_text.configure(state="disabled")

    def _set_translation_text(self, text: str) -> None:
        self.translation_text.configure(state="normal")
        self.translation_text.delete("1.0", "end")
        self.translation_text.insert("1.0", text)
        self.translation_text.configure(state="disabled")

    def _reset_for_listening(self) -> None:
        self.auto_update_original = True
        self.awaiting_next = False
        self._last_translated_source = ""
        self.original_text.delete("1.0", "end")
        self._clear_translation()
        self.continue_btn.configure(state="disabled")
        self.status_var.set("Listening...")
        self.listener.resume()

    def _on_speech_start(self) -> None:
        if self.awaiting_next:
            return
        self.auto_update_original = True
        self._clear_translation()
        self._set_original_text("…")

    def _on_utterance_captured(self, audio: np.ndarray) -> None:
        self.event_queue.put(("utterance", audio.copy()))

    def _on_partial_audio(self, audio: np.ndarray) -> None:
        if self._partial_busy or not self.models_ready or self.awaiting_next:
            return
        self._partial_busy = True
        threading.Thread(
            target=self._transcribe_partial,
            args=(audio,),
            daemon=True,
            name="partial-stt",
        ).start()

    def _transcribe_partial(self, audio: np.ndarray) -> None:
        source_code = LANGUAGES.get(self.source_var.get())
        try:
            text = self.stt.transcribe(audio, source_code)
            if text:
                self.event_queue.put(("partial_text", text))
        finally:
            self._partial_busy = False

    def _begin_final_transcription(self, audio: np.ndarray) -> None:
        if self.transcribing or not self.models_ready:
            return
        self.transcribing = True
        self.listener.pause()
        threading.Thread(
            target=self._transcribe_final,
            args=(audio,),
            daemon=True,
            name="final-stt",
        ).start()

    def _transcribe_final(self, audio: np.ndarray) -> None:
        source_code = LANGUAGES.get(self.source_var.get())
        try:
            self.event_queue.put(("status", "Finalizing transcript..."))
            text = self.stt.transcribe(audio, source_code)
            if not text:
                raise TranslationError("No speech detected. Check the mic meter moves when you speak.")
            self.event_queue.put(("final_text", text))
        except Exception as exc:  # noqa: BLE001
            self.event_queue.put(("error", str(exc)))
        finally:
            self.transcribing = False

    def _schedule_translation(
        self,
        text: str,
        *,
        debounce_ms: int = 0,
        force: bool = False,
    ) -> None:
        text = text.strip()
        if not text or text == "…":
            return
        if not force:
            elapsed = time.monotonic() - self._last_translate_at
            if elapsed < MIN_TRANSLATE_INTERVAL_SECONDS and text == self._last_translated_source:
                return
        self._translate_target_text = text
        if self._translate_timer is not None:
            self.root.after_cancel(self._translate_timer)
        self._translate_timer = self.root.after(debounce_ms, self._flush_translation)

    def _flush_translation(self) -> None:
        self._translate_timer = None
        text = self._translate_target_text.strip()
        if not text or text == "…":
            return
        if self.translating:
            self._translate_pending_text = text
            return
        self.translating = True
        self.event_queue.put(("status", "Translating..."))
        threading.Thread(
            target=self._run_translation,
            args=(text, self.source_var.get(), self.target_var.get()),
            daemon=True,
            name="translate",
        ).start()

    def _run_translation(self, original: str, source_name: str, target_name: str) -> None:
        try:
            translated = translate(original, source_name, target_name)
            self.event_queue.put(("translation", (original, translated)))
        except Exception as exc:  # noqa: BLE001
            self.event_queue.put(("error", str(exc)))
        finally:
            self.translating = False

    def _on_continue(self) -> None:
        if not self.models_ready:
            return
        self._reset_for_listening()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "level":
                    self.level_var.set(float(payload) * 100)
                elif kind == "speech_start":
                    self._on_speech_start()
                elif kind == "partial":
                    self._on_partial_audio(payload)
                elif kind == "partial_text":
                    self._set_original_text(str(payload))
                    if self.auto_update_original:
                        self._schedule_translation(str(payload))
                    self.status_var.set("Listening and translating...")
                elif kind == "utterance":
                    self._begin_final_transcription(payload)
                elif kind == "final_text":
                    self.auto_update_original = True
                    text = str(payload)
                    self._set_original_text(text)
                    self._schedule_translation(text, force=True)
                    self.awaiting_next = True
                    self.continue_btn.configure(state="normal")
                    self.status_var.set("Done. Click Continue for the next sentence.")
                elif kind == "translation":
                    original, translated = payload
                    self._last_translate_at = time.monotonic()
                    self._last_translated_source = original
                    self._set_translation_text(translated)
                    if self.awaiting_next:
                        self.status_var.set("Translation ready. Click Continue for the next sentence.")
                    else:
                        self.status_var.set("Listening and translating...")
                    if self._translate_pending_text and self._translate_pending_text != original:
                        pending = self._translate_pending_text
                        self._translate_pending_text = ""
                        self._schedule_translation(pending, force=True)
                elif kind == "error":
                    messagebox.showerror("Error", str(payload))
                    self.awaiting_next = True
                    self.continue_btn.configure(state="normal")
                    self.status_var.set("Click Continue to listen again.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_close(self) -> None:
        self.listener.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("aqua")
    except tk.TclError:
        pass
    TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
