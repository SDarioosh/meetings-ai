# Meetings.ai

**Private, self-hosted meeting notes.** Record a meeting and get back a full timestamped transcript
with speaker labels, plus a detailed summary you can customize: executive summary, decisions,
action items (owner · task · due date · acceptance criteria), a plan, open questions and risks.

Everything runs on your own hardware: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for speech-to-text, [pyannote](https://github.com/pyannote/pyannote-audio) for speaker detection,
and your own [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` for summaries.
The default path makes no cloud API calls.

> **Status:** v0.1. The server pipeline and CLI work. The macOS/Windows desktop app and the HTTP API
> are next. See [docs/architecture.md](docs/architecture.md).

```
meetings process standup.m4a --diarize
  → meetings/2026-10-16_0930_standup/
      transcript.txt         [00:01:12] Speaker 2: CSV export is finished…
      transcript.json        segments + word timings + speakers
      summary.md             the minutes (see example below)
      summary.json           the same, structured
      raw_llm_response.json  exactly what the LLM returned (debugging)
```

---

## Setup

You need:

- **Python 3.10+** and **ffmpeg**
- A running **llama-server** with an OpenAI-compatible API (any instruct model; Qwen3/3.5,
  Llama 3.x, Mistral and Hermes all work). A context window of 16K+ tokens is recommended.
- Optional: an **NVIDIA GPU** for fast transcription (CPU and Apple Silicon work, just slower).
- Optional: a **Hugging Face account**. It's needed once to download the speaker-detection model.

### 1. Install

```bash
git clone https://github.com/SDarioosh/meetings-ai.git
cd meetings-ai
python3 -m venv server/.venv
source server/.venv/bin/activate          # Windows: server\.venv\Scripts\activate
pip install -e "server[dev]"              # add ,diarize for speaker detection: "server[dev,diarize]"
```

Install ffmpeg if you don't have it: `brew install ffmpeg` (macOS), `sudo apt install ffmpeg`
(Ubuntu), or `winget install ffmpeg` (Windows).

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`. This is the only place your server address goes, and `.env` is gitignored:

```ini
MEETINGS_LLM_BASE_URL=http://<your-llama-server-host>:<port>/v1
MEETINGS_LLM_MODEL=qwen3.5-35b        # a model id from your server's /v1/models
```

Every setting in `.env.example` can also be set as an environment variable, and the common ones
as CLI flags.

### 3. Check

```bash
meetings check
```

```
✓ ffmpeg             /opt/homebrew/bin/ffmpeg
✓ faster-whisper     model large-v3-turbo on cuda (int8_float16)
✓ llama-server       http://…/v1 — models: qwen3.5-35b, …
✓ template           default (7 sections, 0 custom)
- speaker detection  off (enable with --diarize or MEETINGS_DIARIZE=true)
```

Each failed check prints what to fix.

### 4. Run

```bash
meetings process path/to/meeting.m4a
```

The first run downloads the Whisper model (`large-v3-turbo` is about 1.6 GB) into the Hugging
Face cache. After that it works offline.

---

## Speaker detection (optional)

1. `pip install -e "server[diarize]"` (this pulls in PyTorch).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
4. Add `HF_TOKEN=hf_…` to `.env`, then run `meetings process meeting.m4a --diarize`.
   If you know how many people spoke, add `--speakers 4`.

The token is only used to download the model the first time. After that, speaker detection runs offline.

## Customizing summaries

Summaries are driven by YAML templates in [`server/templates/`](server/templates):

| Template | For |
|---|---|
| `default` | Detailed general minutes: all sections |
| `standup` | Updates by person, blockers |
| `one-on-one` | Feedback, career & growth, follow-ups |
| `sales-call` | Customer needs, objections, buying signals, next steps |

```bash
meetings templates                                   # list them
meetings process call.m4a -t sales-call              # pick one
meetings summarize meetings/…/transcript.json -t standup   # re-summarize without re-transcribing
```

To make your own, create `server/templates/my-team.yaml`:

```yaml
extends: default            # inherit prompts and guidance
name: my-team
detail: detailed            # brief | standard | detailed
sections: [executive_summary, decisions, action_items, plan, risks]
custom_sections:
  - key: customer_impact
    title: Customer Impact
    type: list              # list | text
    instructions: Anything that affects customers, with the customer named if mentioned.
guidance:
  action_items: Include the ticket number if one is mentioned.
extra_instructions: Write in British English. Refer to the product as "Falcon".
```

Every prompt is in the YAML (`system_prompt`, `user_prompt`, `merge_prompt`), so you can tune
wording without touching code. The JSON schema sent to the model is generated from `sections` and
`custom_sections`.

### How the LLM output stays reliable

- **`json_schema` mode:** the request includes a JSON schema, which llama-server turns into a
  grammar, so the model can only produce valid JSON in the right shape.
- **`markdown` mode:** a strict heading/table layout that is parsed back into the same structure.
- **`auto`** (the default) tries JSON first and falls back to Markdown if the server rejects the
  schema or the answer won't parse.
- Owners, due dates and acceptance criteria that weren't stated come out as `unresolved` rather
  than guessed.
- Long meetings (over `MEETINGS_MAX_TRANSCRIPT_TOKENS`) are summarized in parts, then merged.
- For Qwen3-family models, "thinking" is turned off for summaries (`MEETINGS_LLM_DISABLE_THINKING`).

## Watch a folder

```bash
meetings watch ~/Meetings/inbox --diarize
```

Drop a recording into the folder. Once the file has finished writing, it is processed and moved
to `inbox/processed/` (or to `inbox/failed/`, with the error printed).

## Running on a GPU server with Docker

For an Ubuntu host with an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
git clone https://github.com/SDarioosh/meetings-ai.git && cd meetings-ai
cp .env.example .env && nano .env                 # set MEETINGS_LLM_BASE_URL, HF_TOKEN, MEETINGS_DIARIZE=true
docker compose build
docker compose run --rm meetings check
docker compose up -d                              # watches ./data/inbox → ./data/meetings
```

The image uses CUDA 12 + cuDNN 9 and includes speaker detection. Models are cached in a Docker
volume, and `server/templates` is mounted so template edits apply without a rebuild. The
container uses host networking, so it reaches llama-server the same way the host does
(localhost, LAN or Tailscale).

### Choosing a Whisper model

| GPU / machine | Recommended `MEETINGS_WHISPER_MODEL` |
|---|---|
| NVIDIA with ≥ 6 GB free (e.g. RTX 3060 12 GB) | `large-v3-turbo` (default) |
| NVIDIA with little VRAM left after the LLM | `medium`, or `large-v3-turbo` with `MEETINGS_WHISPER_DEVICE=cpu` |
| Laptop CPU / Apple Silicon (testing) | `small` or `base` |

By default Whisper is unloaded before the LLM is called (`MEETINGS_UNLOAD_WHISPER=true`), so the
two can share one GPU.

## Troubleshooting

| Message | Fix |
|---|---|
| `Can't reach llama-server at …` | Is it running? `curl $MEETINGS_LLM_BASE_URL/models`. Check VPN/Tailscale, and that the URL ends in `/v1`. |
| `model 'x' isn't listed` | `meetings check` lists the available ids. Set `MEETINGS_LLM_MODEL`. |
| `did not answer within 600s` | Raise `MEETINGS_LLM_TIMEOUT`, or lower `MEETINGS_MAX_TRANSCRIPT_TOKENS` to summarize in smaller parts. |
| `Unsupported audio format` | Convert it: `ffmpeg -i in.xyz -ar 16000 -ac 1 out.wav` |
| `CUDA libraries failed to load` | Use the Docker image, install cuDNN 9 + cuBLAS 12, or set `MEETINGS_WHISPER_DEVICE=cpu`. |
| `needs a Hugging Face token` | See [Speaker detection](#speaker-detection-optional). |
| A summary looks wrong | Check `raw_llm_response.json`, tune the template, then `meetings summarize …/transcript.json`. |

## Development

```bash
pytest server/tests                                   # tests mock the LLM; no network or GPU needed
scripts/make-demo-meeting.sh demo.m4a                 # macOS: synthesize a 3-voice test meeting
meetings process demo.m4a --whisper-model small
```

## Example output

Generated from the synthetic `scripts/make-demo-meeting.sh` recording (Whisper `small` on an M2 Pro
CPU, summary by Qwen3.5-35B-A3B through llama-server; about 35 s end to end):

```markdown
# Falcon Dashboard Launch Weekly Sync

**Date:** 2026-10-16 21:19 · **Duration:** 1m 13s · **Template:** default

## Executive Summary

This weekly sync focused on the Falcon dashboard launch, covering updates on export features,
API performance, and the launch timeline. Priya confirmed that CSV export is complete and PDF
export is 70% done, with a target completion date of next Friday, the 24th. The team decided to
include PDF export in version 1 due to customer demand …

## Action Items

| # | Owner | Task | Due date | Acceptance criteria |
|---|-------|------|----------|---------------------|
| 1 | Priya | Finish PDF export. | Friday the 24th | PDF export is finished. |
| 2 | unresolved | Update the pricing page. | unresolved | Pricing page reflects that PDF is a paid feature before launch. |
| 3 | Marco | Add caching in front of the reporting API. | Wednesday | Every chart loads in under two seconds. |

## Open Questions

- The specific launch date has not been confirmed by Marketing.
…
```

## Roadmap

- [x] Transcription, speaker detection, template-driven summaries, folder watcher, Docker
- [ ] HTTP API (`meetings serve`) for remote clients
- [ ] Desktop app for macOS & Windows (Tauri): one-click recording of mic + system audio
- [ ] Hermes agent integration (MCP tools: search meetings, open action items, draft follow-ups)
- [ ] Speaker naming (map `Speaker 2` → "Priya" and remember voices)

## License

[MIT](LICENSE)
