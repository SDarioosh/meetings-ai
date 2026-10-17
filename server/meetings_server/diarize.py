"""Speaker detection with pyannote.audio, and merging speaker turns into the transcript."""

from __future__ import annotations

import gc
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings
from .errors import DiarizationError
from .schema import Segment, Transcript, Word


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def run_diarization(wav_path: Path, settings: Settings, log: Callable[[str], None] = lambda _m: None) -> list[Turn]:
    if not settings.hf_token:
        raise DiarizationError(
            "Speaker detection needs a Hugging Face token (HF_TOKEN), which is not set.",
            f"1) Create a read token at https://huggingface.co/settings/tokens  "
            f"2) Accept the terms at https://huggingface.co/{settings.diarization_model} "
            "(and pyannote/segmentation-3.0)  3) Put HF_TOKEN=... in .env. "
            "Or run without --diarize.",
        )
    try:
        import numpy as np
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio is not installed.",
            'Install the optional extra: pip install -e "server[diarize]"',
        ) from exc

    log(f"Loading speaker-detection model '{settings.diarization_model}'")
    try:
        try:
            pipeline = Pipeline.from_pretrained(settings.diarization_model, token=settings.hf_token)
        except TypeError:  # pyannote.audio 3.x
            pipeline = Pipeline.from_pretrained(settings.diarization_model, use_auth_token=settings.hf_token)
    except Exception as exc:
        raise DiarizationError(
            f"Could not load {settings.diarization_model}: {exc}",
            f"Make sure HF_TOKEN is valid and you accepted the model terms at "
            f"https://huggingface.co/{settings.diarization_model}. The first run needs internet to download it; "
            "after that it works offline from the Hugging Face cache.",
        ) from exc
    if pipeline is None:
        raise DiarizationError(
            f"Access to {settings.diarization_model} was denied.",
            f"Accept the model's terms at https://huggingface.co/{settings.diarization_model} with the account that owns HF_TOKEN.",
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Pass audio in memory: avoids pyannote's own decoder dependencies.
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    waveform = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)

    kwargs = {}
    if settings.num_speakers:
        kwargs["num_speakers"] = settings.num_speakers
    else:
        if settings.min_speakers:
            kwargs["min_speakers"] = settings.min_speakers
        if settings.max_speakers:
            kwargs["max_speakers"] = settings.max_speakers

    log("Detecting speakers")
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    except Exception as exc:
        raise DiarizationError(f"Speaker detection failed: {exc}", "Try again without --diarize to get a plain transcript.") from exc
    annotation = getattr(output, "speaker_diarization", output)  # pyannote 4.x wraps the result
    turns = [Turn(t.start, t.end, str(spk)) for t, _, spk in annotation.itertracks(yield_label=True)]

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Found {len({t.speaker for t in turns})} speakers")
    return turns


def _speaker_at(start: float, end: float, turns: list[Turn]) -> str | None:
    """Speaker with the most overlap with [start, end]; nearest turn if none overlaps."""
    if not turns:
        return None
    overlap: dict[str, float] = {}
    for t in turns:
        o = min(end, t.end) - max(start, t.start)
        if o > 0:
            overlap[t.speaker] = overlap.get(t.speaker, 0.0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def assign_speakers(transcript: Transcript, turns: list[Turn]) -> Transcript:
    """Label every word with a speaker and split segments where the speaker changes.

    Raw pyannote labels (SPEAKER_00, ...) are renamed "Speaker 1", "Speaker 2", ... in order of first appearance.
    """
    new_segments: list[Segment] = []
    for seg in transcript.segments:
        if not seg.words:
            new_segments.append(seg.model_copy(update={"speaker": _speaker_at(seg.start, seg.end, turns)}))
            continue
        words = [w.model_copy(update={"speaker": _speaker_at(w.start, w.end, turns)}) for w in seg.words]
        group: list[Word] = [words[0]]
        for w in words[1:]:
            if w.speaker != group[-1].speaker:
                new_segments.append(_segment_from(group))
                group = []
            group.append(w)
        new_segments.append(_segment_from(group))

    names: dict[str, str] = {}
    for seg in new_segments:
        if seg.speaker and seg.speaker not in names:
            names[seg.speaker] = f"Speaker {len(names) + 1}"
    for seg in new_segments:
        seg.speaker = names.get(seg.speaker or "", seg.speaker)
        for w in seg.words:
            w.speaker = names.get(w.speaker or "", w.speaker)

    return transcript.model_copy(update={"segments": new_segments, "diarized": True})


def _segment_from(words: list[Word]) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.text for w in words).strip(),
        speaker=words[0].speaker,
        words=words,
    )
