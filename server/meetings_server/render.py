"""Turn transcripts and summaries into human-readable text / Markdown."""

from __future__ import annotations

from datetime import datetime

from .schema import MeetingSummary, Transcript
from .templates import Template


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds <= 0:
        return ""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _turns(transcript: Transcript) -> list[tuple[float, str | None, str]]:
    """Merge consecutive segments by the same (known) speaker into turns; unlabelled segments stay separate."""
    turns: list[tuple[float, str | None, str]] = []
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker and turns and turns[-1][1] == seg.speaker:
            start, speaker, prev = turns[-1]
            turns[-1] = (start, speaker, f"{prev} {text}")
        else:
            turns.append((seg.start, seg.speaker, text))
    return turns


def transcript_to_text(transcript: Transcript) -> str:
    """One line per speaker turn: `[HH:MM:SS] Speaker: text`."""
    lines = []
    for start, speaker, text in _turns(transcript):
        prefix = f"[{fmt_ts(start)}] "
        if speaker:
            prefix += f"{speaker}: "
        lines.append(prefix + text)
    return "\n".join(lines) + ("\n" if lines else "")


def _cell(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def summary_to_markdown(
    summary: MeetingSummary,
    template: Template,
    transcript: Transcript | None = None,
    meeting_date: datetime | None = None,
) -> str:
    out: list[str] = [f"# {summary.title or 'Meeting Summary'}", ""]

    meta = []
    if meeting_date:
        meta.append(f"**Date:** {meeting_date:%Y-%m-%d %H:%M}")
    if transcript and transcript.duration:
        meta.append(f"**Duration:** {fmt_duration(transcript.duration)}")
    if transcript and transcript.speakers:
        meta.append(f"**Speakers:** {', '.join(transcript.speakers)}")
    meta.append(f"**Template:** {template.name}")
    out += [" · ".join(meta), ""]

    for key in template.sections:
        title = template.section_title(key)
        if key == "executive_summary":
            if summary.executive_summary:
                out += [f"## {title}", "", summary.executive_summary.strip(), ""]
        elif key == "key_points":
            out += [f"## {title}", ""] + _bullets(summary.key_points)
        elif key == "decisions":
            items = [f"{d.decision}" + (f" — {d.context}" if d.context else "") for d in summary.decisions]
            out += [f"## {title}", ""] + _bullets(items)
        elif key == "action_items":
            out += [f"## {title}", ""]
            if summary.action_items:
                out += [
                    "| # | Owner | Task | Due date | Acceptance criteria |",
                    "|---|-------|------|----------|---------------------|",
                ]
                for i, a in enumerate(summary.action_items, 1):
                    out.append(
                        f"| {i} | {_cell(a.owner)} | {_cell(a.task)} | {_cell(a.due_date)} | {_cell(a.acceptance_criteria)} |"
                    )
                out.append("")
            else:
                out += ["_None identified._", ""]
        elif key == "plan":
            out += [f"## {title}", ""]
            if summary.plan:
                for i, p in enumerate(summary.plan, 1):
                    out.append(f"{i}. {p.step} — _owner: {p.owner}; timeframe: {p.timeframe}_")
                out.append("")
            else:
                out += ["_None identified._", ""]
        elif key == "open_questions":
            out += [f"## {title}", ""] + _bullets(summary.open_questions)
        elif key == "risks":
            items = [f"{r.risk}" + (f" — _mitigation: {r.mitigation}_" if r.mitigation else "") for r in summary.risks]
            out += [f"## {title}", ""] + _bullets(items)

    for cs in template.custom_sections:
        value = summary.custom_sections.get(cs.key)
        out += [f"## {cs.title}", ""]
        if isinstance(value, list):
            out += _bullets([str(v) for v in value])
        elif value:
            out += [str(value).strip(), ""]
        else:
            out += ["_None identified._", ""]

    return "\n".join(out).rstrip() + "\n"


def _bullets(items: list[str]) -> list[str]:
    if not items:
        return ["_None identified._", ""]
    return [f"- {i}" for i in items] + [""]
