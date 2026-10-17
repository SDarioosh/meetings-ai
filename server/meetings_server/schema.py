"""Data models for transcripts and meeting summaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

UNRESOLVED = "unresolved"


# --- Transcript -------------------------------------------------------------


class Word(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    source_file: str = ""
    language: str = ""
    duration: float = 0.0
    model: str = ""
    diarized: bool = False
    segments: list[Segment] = Field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for seg in self.segments:
            if seg.speaker:
                seen.setdefault(seg.speaker, None)
        return list(seen)


# --- Summary ----------------------------------------------------------------


class Decision(BaseModel):
    decision: str
    context: str = ""


class ActionItem(BaseModel):
    owner: str = UNRESOLVED
    task: str
    due_date: str = UNRESOLVED
    acceptance_criteria: str = UNRESOLVED


class PlanStep(BaseModel):
    step: str
    owner: str = UNRESOLVED
    timeframe: str = UNRESOLVED


class Risk(BaseModel):
    risk: str
    mitigation: str = ""


class MeetingSummary(BaseModel):
    title: str = ""
    executive_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    # Template-defined extra sections: key -> text or list of bullets
    custom_sections: dict[str, Any] = Field(default_factory=dict)


# JSON-schema fragments for each built-in section. The full schema sent to the
# LLM is assembled per-template so the grammar only asks for enabled sections.
_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


BUILTIN_SECTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "executive_summary": _STR,
    "key_points": _STR_LIST,
    "decisions": {
        "type": "array",
        "items": _obj({"decision": _STR, "context": _STR}, ["decision", "context"]),
    },
    "action_items": {
        "type": "array",
        "items": _obj(
            {"owner": _STR, "task": _STR, "due_date": _STR, "acceptance_criteria": _STR},
            ["owner", "task", "due_date", "acceptance_criteria"],
        ),
    },
    "plan": {
        "type": "array",
        "items": _obj({"step": _STR, "owner": _STR, "timeframe": _STR}, ["step", "owner", "timeframe"]),
    },
    "open_questions": _STR_LIST,
    "risks": {
        "type": "array",
        "items": _obj({"risk": _STR, "mitigation": _STR}, ["risk", "mitigation"]),
    },
}

BUILTIN_SECTIONS = tuple(BUILTIN_SECTION_SCHEMAS)
