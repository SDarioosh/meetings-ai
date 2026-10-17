"""End-to-end pipeline: audio -> transcript (+ speakers) -> summary, written to an output folder."""

from __future__ import annotations

import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import audio, transcribe
from .config import Settings
from .errors import MeetingsError
from .render import summary_to_markdown, transcript_to_text
from .schema import Transcript
from .summarize import LLMClient, SummaryResult, summarize_transcript
from .templates import load_template

Log = Callable[[str], None]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "meeting"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_transcript(transcript: Transcript, out_dir: Path) -> None:
    _write_json(out_dir / "transcript.json", transcript.model_dump())
    (out_dir / "transcript.txt").write_text(transcript_to_text(transcript), encoding="utf-8")


def summarize_into(
    transcript: Transcript,
    out_dir: Path,
    settings: Settings,
    meeting_date: datetime | None = None,
    log: Log = print,
    client: LLMClient | None = None,
) -> SummaryResult:
    template = load_template(settings.template, settings.templates_dir)
    raw: list[dict] = []
    try:
        result = summarize_transcript(
            transcript, template, settings, client=client, meeting_date=meeting_date, log=log, raw=raw
        )
    finally:
        # Always keep the raw exchange for debugging, even if parsing failed.
        if raw:
            _write_json(out_dir / "raw_llm_response.json", raw)
    _write_json(out_dir / "summary.json", {"template": template.name, "mode": result.mode, **result.summary.model_dump()})
    (out_dir / "summary.md").write_text(
        summary_to_markdown(result.summary, template, transcript, meeting_date), encoding="utf-8"
    )
    log(f"Summary written ({result.mode} mode)")
    return result


def process_file(audio_path: Path, settings: Settings, log: Log = print) -> Path:
    """Run the whole pipeline for one recording. Returns the output folder."""
    audio_path = audio_path.expanduser().resolve()
    audio.check_audio_file(audio_path)
    load_template(settings.template, settings.templates_dir)  # fail fast on a bad template

    meeting_date = datetime.fromtimestamp(audio_path.stat().st_mtime)
    out_dir = settings.output_dir.expanduser().resolve() / f"{meeting_date:%Y-%m-%d_%H%M}_{_slug(audio_path.stem)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Processing {audio_path.name} -> {out_dir}")
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="meetings-") as tmp:
        wav = audio.to_wav16k(audio_path, Path(tmp) / "audio.wav")
        transcript = transcribe.transcribe(wav, settings, log)
        transcript.source_file = audio_path.name
        if settings.unload_whisper:
            transcribe.unload_models()

        if settings.diarize:
            from .diarize import assign_speakers, run_diarization

            transcript = assign_speakers(transcript, run_diarization(wav, settings, log))

    write_transcript(transcript, out_dir)
    log(f"Transcript written ({time.monotonic() - t0:.0f}s so far)")

    try:
        summarize_into(transcript, out_dir, settings, meeting_date, log)
    except MeetingsError as exc:
        exc.hint = (
            f"{exc.hint} The transcript was saved; once fixed, re-run only the summary with: "
            f"meetings summarize {out_dir / 'transcript.json'}"
        ).strip()
        raise
    log(f"Done in {time.monotonic() - t0:.0f}s")
    return out_dir
