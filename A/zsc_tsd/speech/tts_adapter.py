"""Stable TTS facade; the TensorFlow implementation stays in its own env."""

from __future__ import annotations

from typing import Protocol

from .transformertts import TransformerTTSClient, TransformerTTSResult


class TTSBackend(Protocol):
    def synthesize(self, text: str, output_path, *, report_path=None): ...


def default_tts_backend() -> TransformerTTSClient:
    """Return the frozen public TransformerTTS LJSpeech step-90000 backend."""

    return TransformerTTSClient()


__all__ = ["TTSBackend", "TransformerTTSClient", "TransformerTTSResult", "default_tts_backend"]
