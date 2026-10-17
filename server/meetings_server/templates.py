"""Load summary templates (YAML) and build LLM prompts + JSON schemas from them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from .errors import ConfigError
from .schema import BUILTIN_SECTION_SCHEMAS, BUILTIN_SECTIONS

SECTION_TITLES = {
    "executive_summary": "Executive Summary",
    "key_points": "Key Points",
    "decisions": "Key Decisions",
    "action_items": "Action Items",
    "plan": "Plan & Next Steps",
    "open_questions": "Open Questions",
    "risks": "Risks & Blockers",
}

DETAIL_LEVELS = {
    "brief": "Brief — only the essentials, short bullets, no more than ~5 items per section.",
    "standard": "Standard — cover every important point concisely.",
    "detailed": (
        "Detailed — be thorough and specific: capture every substantive point, figure, name, "
        "date and rationale mentioned. Prefer completeness over brevity."
    ),
}

_jinja = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, autoescape=False)


@dataclass(frozen=True)
class CustomSection:
    key: str
    title: str
    type: str = "list"  # "list" | "text"
    instructions: str = ""


@dataclass(frozen=True)
class Template:
    name: str
    system_prompt: str
    user_prompt: str
    merge_prompt: str
    description: str = ""
    detail: str = "detailed"
    sections: tuple[str, ...] = BUILTIN_SECTIONS
    custom_sections: tuple[CustomSection, ...] = ()
    guidance: dict[str, str] = field(default_factory=dict)
    extra_instructions: str = ""

    def section_title(self, key: str) -> str:
        return SECTION_TITLES.get(key, key.replace("_", " ").title())

    @property
    def detail_instructions(self) -> str:
        return DETAIL_LEVELS[self.detail]


# --- Loading ----------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Template {path} is not valid YAML: {exc}", "Fix the YAML syntax in that file.") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Template {path} must be a YAML mapping.", "See templates/default.yaml for the format.")
    return data


def _merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Child overrides base. Dict values (e.g. guidance) merge one level deep; lists replace."""
    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _resolve(name_or_path: str, templates_dir: Path, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = templates_dir / f"{name_or_path}.yaml"
    if not path.is_file():
        available = ", ".join(list_templates(templates_dir)) or "none found"
        raise ConfigError(
            f"Template {name_or_path!r} not found (looked for {path}).",
            f"Available templates: {available}. Templates live in {templates_dir}.",
        )
    key = str(path.resolve())
    if key in seen:
        raise ConfigError(f"Template {path} has a circular `extends`.", "Remove the loop in `extends:`.")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent:
        parent_path = parent if Path(parent).suffix else str(path.parent / f"{parent}.yaml")
        data = _merge(_resolve(parent_path, templates_dir, seen + (key,)), data)
    return data


def load_template(name_or_path: str, templates_dir: Path) -> Template:
    data = _resolve(name_or_path, templates_dir)

    for required in ("system_prompt", "user_prompt", "merge_prompt"):
        if not data.get(required):
            raise ConfigError(
                f"Template {name_or_path!r} is missing `{required}`.",
                "Add it, or add `extends: default` to inherit it.",
            )

    sections = tuple(data.get("sections") or ())
    unknown = [s for s in sections if s not in BUILTIN_SECTIONS]
    if unknown:
        raise ConfigError(
            f"Template {name_or_path!r} lists unknown sections: {', '.join(unknown)}",
            f"Built-in sections are: {', '.join(BUILTIN_SECTIONS)}. Use custom_sections for anything else.",
        )

    detail = data.get("detail", "detailed")
    if detail not in DETAIL_LEVELS:
        raise ConfigError(
            f"Template {name_or_path!r} has invalid detail {detail!r}.",
            f"Use one of: {', '.join(DETAIL_LEVELS)}",
        )

    customs = []
    for raw in data.get("custom_sections") or ():
        if not isinstance(raw, dict) or not raw.get("key"):
            raise ConfigError(f"Template {name_or_path!r}: each custom section needs a `key`.", "")
        cs_type = raw.get("type", "list")
        if cs_type not in ("list", "text"):
            raise ConfigError(f"Custom section {raw['key']!r}: type must be 'list' or 'text'.", "")
        if raw["key"] in BUILTIN_SECTIONS or raw["key"] == "title":
            raise ConfigError(f"Custom section key {raw['key']!r} clashes with a built-in section.", "Rename it.")
        customs.append(
            CustomSection(
                key=raw["key"],
                title=raw.get("title") or raw["key"].replace("_", " ").title(),
                type=cs_type,
                instructions=raw.get("instructions", ""),
            )
        )

    return Template(
        name=data.get("name") or Path(name_or_path).stem,
        description=data.get("description", ""),
        detail=detail,
        sections=sections,
        custom_sections=tuple(customs),
        guidance=dict(data.get("guidance") or {}),
        extra_instructions=data.get("extra_instructions") or "",
        system_prompt=data["system_prompt"],
        user_prompt=data["user_prompt"],
        merge_prompt=data["merge_prompt"],
    )


def list_templates(templates_dir: Path) -> list[str]:
    if not templates_dir.is_dir():
        return []
    return sorted(p.stem for p in templates_dir.glob("*.yaml"))


# --- Schema & prompt building ----------------------------------------------


def build_json_schema(template: Template) -> dict[str, Any]:
    """JSON schema for exactly the sections this template asks for (used for grammar-constrained output)."""
    props: dict[str, Any] = {"title": {"type": "string"}}
    for key in template.sections:
        props[key] = BUILTIN_SECTION_SCHEMAS[key]
    if template.custom_sections:
        custom_props = {
            cs.key: ({"type": "array", "items": {"type": "string"}} if cs.type == "list" else {"type": "string"})
            for cs in template.custom_sections
        }
        props["custom_sections"] = {
            "type": "object",
            "properties": custom_props,
            "required": list(custom_props),
            "additionalProperties": False,
        }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _section_list(template: Template) -> list[dict[str, str]]:
    items = [
        {"key": key, "title": template.section_title(key), "guidance": template.guidance.get(key, "")}
        for key in template.sections
    ]
    items += [
        {
            "key": f"custom_sections.{cs.key}",
            "title": cs.title,
            "guidance": (cs.instructions + (" (a list of bullet strings)" if cs.type == "list" else " (a paragraph)")).strip(),
        }
        for cs in template.custom_sections
    ]
    return items


def markdown_format_spec(template: Template) -> str:
    """Exact output layout for markdown mode, which `summarize.parse_markdown` can read back."""
    lines = [
        "Respond with ONLY Markdown in exactly this layout (keep the headings verbatim, in this order;",
        'write "None" under a heading if there is nothing for it):',
        "",
        "# <meeting title>",
    ]
    for key in template.sections:
        lines.append(f"## {template.section_title(key)}")
        if key == "executive_summary":
            lines.append("<one paragraph>")
        elif key == "action_items":
            lines += [
                "| Owner | Task | Due date | Acceptance criteria |",
                "|---|---|---|---|",
                "| <owner or unresolved> | <task> | <due date or unresolved> | <criteria or unresolved> |",
            ]
        elif key == "decisions":
            lines.append("- <decision> — <context>")
        elif key == "plan":
            lines.append("1. <step> — owner: <owner or unresolved>; timeframe: <timeframe or unresolved>")
        elif key == "risks":
            lines.append("- <risk> — mitigation: <mitigation or none>")
        else:
            lines.append("- <item>")
    for cs in template.custom_sections:
        lines.append(f"## {cs.title}")
        lines.append("- <item>" if cs.type == "list" else "<paragraph>")
    return "\n".join(lines)


def output_instructions(template: Template, mode: str) -> str:
    if mode == "json_schema":
        return (
            "Respond with ONLY a JSON object matching the provided schema "
            "(keys: " + ", ".join(build_json_schema(template)["properties"]) + "). No prose, no code fences."
        )
    return markdown_format_spec(template)


def build_messages(
    template: Template,
    transcript_text: str,
    mode: str,
    meeting: dict[str, Any] | None = None,
    chunk: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Chat messages asking the LLM to summarize one transcript (or one chunk of it)."""
    meeting = {"date": "", "duration": "", "speakers": [], **(meeting or {})}
    user = _jinja.from_string(template.user_prompt).render(
        transcript=transcript_text.strip(),
        meeting=meeting,
        chunk=chunk,
        detail_instructions=template.detail_instructions,
        sections=_section_list(template),
        extra_instructions=template.extra_instructions.strip(),
        output_instructions=output_instructions(template, mode),
    )
    return [
        {"role": "system", "content": template.system_prompt.strip()},
        {"role": "user", "content": user.strip()},
    ]


def build_merge_messages(template: Template, partials_json: str, mode: str) -> list[dict[str, str]]:
    user = _jinja.from_string(template.merge_prompt).render(
        partials=partials_json,
        detail_instructions=template.detail_instructions,
        extra_instructions=template.extra_instructions.strip(),
        output_instructions=output_instructions(template, mode),
    )
    return [
        {"role": "system", "content": template.system_prompt.strip()},
        {"role": "user", "content": user.strip()},
    ]
