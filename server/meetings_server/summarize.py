"""Summarize a transcript with an OpenAI-compatible llama.cpp server (`llama-server`)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from .config import Settings
from .errors import LLMError
from .render import fmt_duration, transcript_to_text
from .schema import BUILTIN_SECTIONS, UNRESOLVED, MeetingSummary, Transcript
from .templates import Template, build_json_schema, build_merge_messages, build_messages

CHARS_PER_TOKEN = 3.0  # conservative estimate; real tokenizers average ~3.5-4 for English
_UNRESOLVED_ALIASES = {"", "unknown", "n/a", "na", "none", "tbd", "not specified", "unspecified", "not stated"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class SummaryParseError(LLMError):
    pass


class LLMRequestRejected(LLMError):
    """The server answered with a 4xx — e.g. it does not support json_schema for this model."""


# --- HTTP client ------------------------------------------------------------


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.llm_timeout, connect=10.0),
            transport=transport,
        )

    def _unreachable(self, exc: Exception) -> LLMError:
        return LLMError(
            f"Can't reach llama-server at {self.base_url} ({type(exc).__name__}).",
            "Check that llama-server is running and reachable from this machine "
            f"(try: curl {self.base_url}/models). If it's on another host, make sure the VPN/Tailscale "
            "is up and MEETINGS_LLM_BASE_URL in .env points at it, including the /v1 suffix.",
        )

    def list_models(self) -> list[str]:
        try:
            resp = self._http.get("/models")
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unreachable(exc) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"llama-server /models failed: {exc}", "Check the server logs.") from exc
        return [m.get("id", "") for m in resp.json().get("data", [])]

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> tuple[str, dict]:
        """Send a chat completion. Returns (assistant content, raw response JSON)."""
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "stream": False,
        }
        if self.settings.llm_model:
            payload["model"] = self.settings.llm_model
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "meeting_summary", "strict": True, "schema": schema},
            }
        if self.settings.llm_disable_thinking:
            # Honoured by llama-server --jinja for Qwen3-family templates; ignored by others.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            resp = self._http.post("/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unreachable(exc) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(
                f"llama-server did not answer within {self.settings.llm_timeout:.0f}s.",
                "Long meetings take a while on local hardware. Raise MEETINGS_LLM_TIMEOUT, "
                "or lower MEETINGS_MAX_TRANSCRIPT_TOKENS so the transcript is summarized in smaller chunks.",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Request to llama-server failed: {exc}", "Check the server logs.") from exc

        if resp.status_code >= 400:
            body = resp.text[:500]
            hint = "Check the llama-server logs."
            if self.settings.llm_model and ("model" in body.lower() or resp.status_code == 404):
                hint = (
                    f"Is the model name {self.settings.llm_model!r} right? "
                    "Run `meetings check` to list the models your server offers, then set MEETINGS_LLM_MODEL."
                )
            elif "context" in body.lower() or "too long" in body.lower():
                hint = "The transcript exceeds the model's context. Lower MEETINGS_MAX_TRANSCRIPT_TOKENS."
            cls = LLMRequestRejected if 400 <= resp.status_code < 500 else LLMError
            raise cls(f"llama-server returned HTTP {resp.status_code}: {body}", hint)

        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("llama-server returned an unexpected response shape.", "See raw_llm_response.json.") from exc
        content = message.get("content") or ""
        if not content.strip() and message.get("reasoning_content"):
            raise SummaryParseError(
                "The model spent its whole answer 'thinking' and returned no content.",
                "Keep MEETINGS_LLM_DISABLE_THINKING=true, or raise the server's max tokens (-n).",
            )
        return content, data

    def close(self) -> None:
        self._http.close()


# --- Parsing ----------------------------------------------------------------


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    fence = re.match(r"^```(?:json|markdown|md)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def _unresolved(value: Any) -> str:
    text = str(value or "").strip()
    return UNRESOLVED if text.lower() in _UNRESOLVED_ALIASES else text


def _normalize(summary: MeetingSummary) -> MeetingSummary:
    for a in summary.action_items:
        a.owner, a.due_date, a.acceptance_criteria = (
            _unresolved(a.owner),
            _unresolved(a.due_date),
            _unresolved(a.acceptance_criteria),
        )
    for p in summary.plan:
        p.owner, p.timeframe = _unresolved(p.owner), _unresolved(p.timeframe)
    for r in summary.risks:
        if r.mitigation.strip().lower() in _UNRESOLVED_ALIASES:
            r.mitigation = ""
    return summary


def parse_json_summary(content: str, template: Template) -> MeetingSummary:
    text = _clean(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} in case the model wrapped it in prose.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise SummaryParseError("LLM response contained no JSON object.", "See raw_llm_response.json.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SummaryParseError(f"LLM returned invalid JSON: {exc}", "See raw_llm_response.json.") from exc
    if not isinstance(data, dict):
        raise SummaryParseError("LLM JSON was not an object.", "See raw_llm_response.json.")

    custom = data.get("custom_sections") or {}
    known = {cs.key for cs in template.custom_sections}
    # Tolerate models that put custom sections at the top level.
    for key in known:
        if key in data and key not in custom:
            custom[key] = data[key]
    payload = {k: v for k, v in data.items() if k == "title" or k in BUILTIN_SECTIONS}
    payload["custom_sections"] = {k: v for k, v in custom.items() if k in known}
    try:
        return _normalize(MeetingSummary.model_validate(payload))
    except ValidationError as exc:
        raise SummaryParseError(f"LLM JSON did not match the summary schema: {exc}", "See raw_llm_response.json.") from exc


def _split_cells(row: str) -> list[str]:
    row = row.strip().strip("|")
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", row)]


def _list_items(body: str) -> list[str]:
    items = []
    for line in body.splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if m and m.group(1).strip().lower() not in ("none", "none.", "n/a"):
            items.append(m.group(1).strip())
    return items


def _split_dash(item: str) -> tuple[str, str]:
    parts = re.split(r"\s+[—–-]\s+", item, maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (item.strip(), "")


def parse_markdown_summary(content: str, template: Template) -> MeetingSummary:
    """Parse the strict-markdown layout produced by `templates.markdown_format_spec`."""
    text = _clean(content)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    by_title = {template.section_title(k).lower(): k for k in template.sections}
    by_title.update({cs.title.lower(): f"custom:{cs.key}" for cs in template.custom_sections})

    bodies: dict[str, str] = {}
    for m in re.finditer(r"^##\s+(.+?)\s*\n(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL):
        key = by_title.get(m.group(1).strip().rstrip(":").lower())
        if key:
            bodies[key] = m.group(2).strip()
    if not bodies:
        raise SummaryParseError("LLM markdown had none of the expected section headings.", "See raw_llm_response.json.")

    data: dict[str, Any] = {"title": title_match.group(1).strip() if title_match else "", "custom_sections": {}}
    for key, body in bodies.items():
        if key == "executive_summary":
            data["executive_summary"] = "" if body.lower().rstrip(".") == "none" else body
        elif key in ("key_points", "open_questions"):
            data[key] = _list_items(body)
        elif key == "decisions":
            data["decisions"] = [{"decision": d, "context": c} for d, c in map(_split_dash, _list_items(body))]
        elif key == "risks":
            data["risks"] = [
                {"risk": r, "mitigation": re.sub(r"^mitigation:\s*", "", m, flags=re.I)}
                for r, m in map(_split_dash, _list_items(body))
            ]
        elif key == "plan":
            steps = []
            for item in _list_items(body):
                step, rest = _split_dash(item)
                owner = re.search(r"owner:\s*([^;]+)", rest, re.I)
                when = re.search(r"timeframe:\s*(.+)$", rest, re.I)
                steps.append(
                    {"step": step, "owner": owner.group(1).strip() if owner else "", "timeframe": when.group(1).strip() if when else ""}
                )
            data["plan"] = steps
        elif key == "action_items":
            rows = [l for l in body.splitlines() if l.strip().startswith("|")]
            items = []
            for row in rows[1:]:  # skip header
                cells = _split_cells(row)
                if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                    continue
                cells += [""] * (4 - len(cells))
                if cells[1]:
                    items.append({"owner": cells[0], "task": cells[1], "due_date": cells[2], "acceptance_criteria": cells[3]})
            data["action_items"] = items
        elif key.startswith("custom:"):
            cs_key = key.split(":", 1)[1]
            cs = next(c for c in template.custom_sections if c.key == cs_key)
            data["custom_sections"][cs_key] = _list_items(body) if cs.type == "list" else body
    return _normalize(MeetingSummary.model_validate(data))


# --- Orchestration ----------------------------------------------------------


@dataclass
class SummaryResult:
    summary: MeetingSummary
    mode: str
    raw_responses: list[dict] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def chunk_lines(text: str, max_tokens: int) -> list[str]:
    """Split transcript text on line boundaries into chunks under `max_tokens` (estimated)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        line_tokens = estimate_tokens(line)
        if current and size + line_tokens > max_tokens:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


def _run(
    client: LLMClient,
    template: Template,
    messages_for_mode: Callable[[str], list[dict[str, str]]],
    mode: str,
    raw: list[dict],
) -> tuple[MeetingSummary, str]:
    """One summarization request; in auto mode falls back from JSON schema to markdown."""
    if mode in ("auto", "json_schema"):
        try:
            content, data = client.chat(messages_for_mode("json_schema"), schema=build_json_schema(template))
            raw.append({"mode": "json_schema", "response": data})
            return parse_json_summary(content, template), "json_schema"
        except (LLMRequestRejected, SummaryParseError) as exc:
            raw.append({"mode": "json_schema", "error": str(exc)})
            if mode == "json_schema":
                raise
    content, data = client.chat(messages_for_mode("markdown"))
    raw.append({"mode": "markdown", "response": data})
    return parse_markdown_summary(content, template), "markdown"


def summarize_transcript(
    transcript: Transcript,
    template: Template,
    settings: Settings,
    client: LLMClient | None = None,
    meeting_date: datetime | None = None,
    log: Callable[[str], None] = lambda _msg: None,
    raw: list[dict] | None = None,
) -> SummaryResult:
    """Summarize `transcript`. Every request/response is appended to `raw` (if given), even on failure."""
    if not transcript.segments:
        raise LLMError("The transcript is empty, so there is nothing to summarize.", "Check the recording has audible speech.")

    own_client = client is None
    client = client or LLMClient(settings)
    raw = raw if raw is not None else []
    meeting = {
        "date": f"{meeting_date:%Y-%m-%d %H:%M}" if meeting_date else "",
        "duration": fmt_duration(transcript.duration),
        "speakers": transcript.speakers,
    }
    try:
        text = transcript_to_text(transcript)
        chunks = chunk_lines(text, settings.max_transcript_tokens)
        mode_used = settings.llm_output_mode

        if len(chunks) == 1:
            log(f"Summarizing ~{estimate_tokens(text):,} tokens of transcript with template '{template.name}'")
            summary, mode_used = _run(
                client, template, lambda m: build_messages(template, text, m, meeting), settings.llm_output_mode, raw
            )
            return SummaryResult(summary, mode_used, raw)

        log(f"Long transcript: summarizing in {len(chunks)} parts, then merging")
        partials = []
        for i, chunk in enumerate(chunks, 1):
            log(f"  part {i}/{len(chunks)}")
            info = {"index": i, "total": len(chunks)}
            part, mode_used = _run(
                client,
                template,
                lambda m, c=chunk, info=info: build_messages(template, c, m, meeting, chunk=info),
                mode_used,
                raw,
            )
            partials.append(part.model_dump())
        log("  merging parts")
        partials_json = json.dumps(partials, ensure_ascii=False, indent=1)
        summary, mode_used = _run(
            client, template, lambda m: build_merge_messages(template, partials_json, m), mode_used, raw
        )
        return SummaryResult(summary, mode_used, raw)
    finally:
        if own_client:
            client.close()
