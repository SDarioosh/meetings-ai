"""Speech-to-text with faster-whisper (CTranslate2 Whisper; CUDA on NVIDIA, CPU elsewhere)."""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .errors import TranscriptionError
from .schema import Segment, Transcript, Word

_cached: dict[tuple[str, str, str], Any] = {}


def resolve_device(settings: Settings) -> tuple[str, str]:
    """Pick (device, compute_type). auto -> CUDA with int8_float16 if available, else CPU int8."""
    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = settings.whisper_compute_type
    if compute == "auto":
        compute = "int8_float16" if device == "cuda" else "int8"
    return device, compute


def _load_model(settings: Settings, log: Callable[[str], None]) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed.",
            'Install the server package: pip install -e "server"',
        ) from exc

    device, compute = resolve_device(settings)
    key = (settings.whisper_model, device, compute)
    if key in _cached:
        return _cached[key]

    log(f"Loading Whisper model '{settings.whisper_model}' on {device} ({compute}) — first run downloads it")
    try:
        model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
    except Exception as exc:  # CTranslate2 raises plain RuntimeError/ValueError
        msg = str(exc)
        hint = "Check MEETINGS_WHISPER_MODEL (e.g. large-v3-turbo, medium, small, base)."
        if "cudnn" in msg.lower() or "cublas" in msg.lower() or "cuda" in msg.lower():
            hint = (
                "CUDA libraries failed to load. Use the provided Docker image (CUDA 12 + cuDNN 9), "
                "install cuDNN 9 / cuBLAS 12 on the host, or set MEETINGS_WHISPER_DEVICE=cpu."
            )
        elif "compute type" in msg.lower():
            hint = "This device doesn't support that compute type; set MEETINGS_WHISPER_COMPUTE_TYPE=auto."
        raise TranscriptionError(f"Could not load Whisper model: {msg}", hint) from exc
    _cached[key] = model
    return model


def unload_models() -> None:
    """Free Whisper's memory (VRAM) so the LLM sharing the GPU has room."""
    _cached.clear()
    gc.collect()


def _load_wav(wav_path: Path) -> Any:
    """Read the 16 kHz mono PCM WAV from `audio.to_wav16k` as float32 (bypasses PyAV decoding)."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe(
    wav_path: Path,
    settings: Settings,
    log: Callable[[str], None] = lambda _msg: None,
) -> Transcript:
    model = _load_model(settings, log)
    try:
        segments_iter, info = model.transcribe(
            _load_wav(wav_path),
            language=settings.language or None,
            word_timestamps=True,  # needed to split segments cleanly at speaker changes
            vad_filter=True,  # skip silences — big speedup on meetings
            beam_size=5,
        )
        segments: list[Segment] = []
        last_report = 0.0
        for s in segments_iter:
            words = [Word(start=w.start, end=w.end, text=w.word) for w in (s.words or [])]
            segments.append(Segment(start=s.start, end=s.end, text=s.text.strip(), words=words))
            if info.duration and s.end - last_report >= 60:
                last_report = s.end
                log(f"  transcribed {s.end / info.duration:5.1%} ({int(s.end // 60)} of {int(info.duration // 60)} min)")
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}", "If this mentions CUDA, try MEETINGS_WHISPER_DEVICE=cpu.") from exc

    log(f"Transcribed {len(segments)} segments (language: {info.language}, p={info.language_probability:.2f})")
    return Transcript(
        source_file=wav_path.name,
        language=info.language,
        duration=info.duration,
        model=settings.whisper_model,
        segments=segments,
    )
