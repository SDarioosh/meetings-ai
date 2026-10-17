"""Audio input checks and normalisation (any format -> 16 kHz mono WAV via ffmpeg)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .errors import AudioError

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mov", ".mkv", ".aiff", ".aif", ".wma",
}


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError(
            "ffmpeg is not installed (needed to decode audio).",
            "Install it: macOS `brew install ffmpeg`, Ubuntu `sudo apt install ffmpeg`, Windows `winget install ffmpeg`.",
        )
    return path


def check_audio_file(path: Path) -> None:
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}", "Check the path.")
    if not path.is_file():
        raise AudioError(f"Not a file: {path}", "Pass a single audio file, or use `meetings watch` for a folder.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported audio format: {path.suffix or '(no extension)'}",
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. Convert with: ffmpeg -i input -ar 16000 -ac 1 out.wav",
        )
    if path.stat().st_size == 0:
        raise AudioError(f"Audio file is empty: {path}", "The recording may have failed or still be writing.")


def probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0.0


def to_wav16k(src: Path, dest: Path) -> Path:
    """Decode `src` to 16 kHz mono 16-bit WAV (what Whisper and pyannote expect)."""
    ffmpeg = require_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise AudioError(
            f"ffmpeg could not decode {src.name}:\n{tail}",
            "The file may be corrupt or use an unusual codec. Try re-exporting it as WAV or M4A.",
        )
    return dest
