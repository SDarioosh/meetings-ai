# Architecture

```
┌──────────── macOS / Windows ───────────┐        ┌──────────── GPU server (Docker) ─────────────────┐
│  Meetings.ai desktop app (planned)     │  HTTP  │  meetings-server (Python)                        │
│  • records mic + system audio          │ ─────► │  1. ffmpeg → 16 kHz mono WAV                     │
│    as separate tracks                  │        │  2. faster-whisper → timestamped words/segments  │
│  • upload, status, browse results      │ ◄───── │  3. pyannote (optional) → speaker turns          │
│  • edit summary templates              │        │  4. llama-server /v1/chat/completions → summary  │
└────────────────────────────────────────┘        └──────────────────────────────────────────────────┘
```

## Pipeline (`meetings_server/pipeline.py`)

1. **Validate & normalise audio** (`audio.py`): check extension/size, decode anything ffmpeg can read
   to 16 kHz mono PCM WAV in a temp dir.
2. **Transcribe** (`transcribe.py`): faster-whisper with VAD and word timestamps. Device/compute type
   auto-detected (CUDA `int8_float16`, CPU `int8`). The model is unloaded afterwards (by default) so
   the LLM sharing the GPU gets its VRAM back.
3. **Speaker detection** (`diarize.py`, optional): pyannote produces speaker turns; every *word* is
   assigned the speaker it overlaps most, and segments are split where the speaker changes. Labels are
   renamed `Speaker 1..N` in order of appearance.
4. **Write transcript** (`transcript.json`, `transcript.txt`) — before the LLM step, so a summary
   failure never loses the transcript (`meetings summarize` re-runs just the summary).
5. **Summarize** (`summarize.py`):
   - The template (`templates.py`) decides which sections exist; a JSON schema containing exactly those
     sections is built and sent as `response_format: json_schema`, which llama-server compiles to a
     grammar → output is always parseable JSON.
   - `auto` mode falls back to a strict Markdown layout (parsed back into the same model) if the server
     rejects the schema or returns something unparseable.
   - Transcripts above `MEETINGS_MAX_TRANSCRIPT_TOKENS` are summarized in parts, then merged with a
     second "merge" prompt.
   - Missing owners / dates / criteria are normalised to `unresolved`.
6. **Render** (`render.py`) → `summary.md` + `summary.json`; every LLM exchange → `raw_llm_response.json`.

## Why faster-whisper instead of whisper.cpp?

Both run Whisper on CUDA at similar speed. faster-whisper is a Python library, so word timestamps,
VAD and speaker alignment happen in-process with no subprocess/JSON plumbing, and pyannote (also Python)
slots in directly. whisper.cpp remains a good option for a future *on-device* mode in the desktop app.

## Model choice for an RTX 3060 (12 GB)

| Model | VRAM (int8_float16) | Notes |
|---|---|---|
| `large-v3-turbo` (default) | ~2–3 GB | Near large-v3 accuracy, several times faster. Best fit. |
| `large-v3` | ~4–5 GB | Slightly more accurate on hard audio/accents; much slower. |
| `medium` / `small` | ~1–2 GB | Use when the LLM leaves little VRAM free, or for CPU tests. |

The 3060 is shared with llama.cpp, so steps run sequentially and Whisper/pyannote are freed before the
LLM call. If your loaded LLM fills the card, set `MEETINGS_WHISPER_DEVICE=cpu` or use a smaller model.

## Planned

- **HTTP API** (`meetings serve`): upload audio, poll job status, fetch results — used by the desktop app.
- **Desktop app** (Tauri 2 + React): mic + system audio capture (ScreenCaptureKit / Core Audio taps on
  macOS, WASAPI loopback on Windows) on separate tracks, so "me vs. others" is known without diarization.
- **Hermes integration**: expose meetings to the Hermes agent harness as tools (MCP) — search past
  meetings, list open action items, draft follow-ups.
