import pytest

from meetings_server.errors import ConfigError
from meetings_server.render import transcript_to_text
from meetings_server.schema import BUILTIN_SECTIONS
from meetings_server.templates import (
    build_json_schema,
    build_merge_messages,
    build_messages,
    list_templates,
    load_template,
    markdown_format_spec,
)

from .conftest import TEMPLATES_DIR


def test_all_bundled_templates_load():
    names = list_templates(TEMPLATES_DIR)
    assert {"default", "standup", "one-on-one", "sales-call"} <= set(names)
    for name in names:
        load_template(name, TEMPLATES_DIR)


def test_extends_merges_guidance_and_replaces_lists(default_template, sales_template):
    # sales-call overrides risks guidance but inherits the rest from default
    assert "Deal risks" in sales_template.guidance["risks"]
    assert sales_template.guidance["action_items"] == default_template.guidance["action_items"]
    assert "decisions" not in sales_template.sections
    assert sales_template.system_prompt == default_template.system_prompt


def test_json_schema_has_only_enabled_sections(sales_template):
    schema = build_json_schema(sales_template)
    props = schema["properties"]
    assert set(props) == {"title", *sales_template.sections, "custom_sections"}
    assert "decisions" not in props
    assert set(props["custom_sections"]["required"]) == {"customer_needs", "objections", "buying_signals"}
    assert schema["required"] == list(props)
    assert props["action_items"]["items"]["required"] == ["owner", "task", "due_date", "acceptance_criteria"]


def test_default_schema_covers_every_builtin(default_template):
    props = build_json_schema(default_template)["properties"]
    assert set(BUILTIN_SECTIONS) <= set(props)
    assert "custom_sections" not in props


def test_build_messages_json_mode(default_template, transcript):
    text = transcript_to_text(transcript)
    msgs = build_messages(
        default_template, text, "json_schema", {"date": "2026-01-05 10:00", "duration": "5m", "speakers": ["Speaker 1"]}
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "Never invent" in system
    assert "<transcript>" in user and "[00:00:14] Speaker 2: Sure. CSV export is done." in user
    assert "2026-01-05 10:00" in user
    assert '"unresolved"' in user  # owner/due-date rule reaches the model
    assert "Action Items (`action_items`)" in user
    assert "Respond with ONLY a JSON object" in user
    assert "part 1 of" not in user  # no chunk note for single-pass


def test_build_messages_markdown_mode_and_chunk_note(sales_template):
    msgs = build_messages(sales_template, "[00:00:00] A: hi", "markdown", chunk={"index": 2, "total": 3})
    user = msgs[1]["content"]
    assert "part 2 of 3" in user
    assert "## Customer Needs & Pain Points" in user
    assert "| Owner | Task | Due date | Acceptance criteria |" in user
    assert "Respond with ONLY a JSON object" not in user


def test_extra_instructions_are_included(tmp_path):
    (tmp_path / "mine.yaml").write_text(
        f"extends: {TEMPLATES_DIR / 'default.yaml'}\nname: mine\nextra_instructions: Always write in British English.\n"
    )
    t = load_template(str(tmp_path / "mine.yaml"), TEMPLATES_DIR)
    assert "British English" in build_messages(t, "x", "json_schema")[1]["content"]
    assert "British English" in build_merge_messages(t, "[]", "json_schema")[1]["content"]


def test_markdown_spec_lists_headings_in_order(default_template):
    spec = markdown_format_spec(default_template)
    order = [spec.index(f"## {default_template.section_title(k)}") for k in default_template.sections]
    assert order == sorted(order)


def test_unknown_section_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(f"extends: {TEMPLATES_DIR / 'default.yaml'}\nsections: [executive_summary, vibes]\n")
    with pytest.raises(ConfigError, match="vibes"):
        load_template(str(tmp_path / "bad.yaml"), TEMPLATES_DIR)


def test_missing_template_lists_available():
    with pytest.raises(ConfigError) as exc:
        load_template("does-not-exist", TEMPLATES_DIR)
    assert "default" in exc.value.hint
