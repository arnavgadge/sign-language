"""
stt/listener.py — Return-channel speech-to-text listener.

Runs a microphone capture loop in a background thread so it never blocks
the main OpenCV loop.  The other person speaks; the deaf/mute user reads
the transcription on-screen.

Dependencies:
    pip install SpeechRecognition pyaudio

Falls back gracefully if either package is missing.
"""

import threading
from collections import deque
from typing import List, Optional

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class SpeechListener:
    """
    Background speech-to-text listener.

    Usage:
        listener = SpeechListener()
        listener.start()          # non-blocking
        text = listener.get_latest()
        history = listener.get_history()
        listener.stop()
    """

    def __init__(self, max_history: int = 8, language: str = "en-US"):
        self.max_history = max_history
        self.language = language

        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._stop_fn = None
        self._listening = False

        if not SR_AVAILABLE:
            print("[STT] SpeechRecognition not installed.")
            print("[STT] Run:  pip install SpeechRecognition pyaudio")
            return

        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold = 0.8

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start background microphone listener. Returns True on success."""
        if not SR_AVAILABLE:
            return False
        try:
            mic = sr.Microphone()
            print("[STT] Calibrating microphone for ambient noise (1 s)...")
            with mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)

            self._stop_fn = self._recognizer.listen_in_background(
                mic,
                self._callback,
                phrase_time_limit=10,
            )
            self._listening = True
            print("[STT] Microphone active. Other person can speak now.")
            return True
        except Exception as exc:
            print(f"[STT] Could not start microphone: {exc}")
            return False

    def stop(self) -> None:
        """Stop the background listener."""
        if self._stop_fn:
            self._stop_fn(wait_for_stop=False)
            self._listening = False

    @property
    def is_listening(self) -> bool:
        return self._listening

    def get_history(self) -> List[str]:
        """Return all transcribed utterances (oldest first)."""
        with self._lock:
            return list(self._history)

    def get_latest(self) -> Optional[str]:
        """Return the most recent transcription, or None."""
        with self._lock:
            return self._history[-1] if self._history else None

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    # ── Background callback ────────────────────────────────────────────────────

    def _callback(self, recognizer, audio) -> None:
        try:
            text = recognizer.recognize_google(audio, language=self.language)
            if text:
                with self._lock:
                    self._history.append(text)
                print(f"[STT] Heard: \"{text}\"")
        except Exception:
            # UnknownValueError  → silence or unintelligible — silently skip
            # RequestError       → network issue — silently skip
            pass
