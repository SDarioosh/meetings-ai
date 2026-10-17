"""Summarization against a mocked llama-server (httpx.MockTransport — no network)."""

import json

import httpx
import pytest

from meetings_server.errors import LLMError
from meetings_server.pipeline import summarize_into
from meetings_server.schema import UNRESOLVED
from meetings_server.summarize import (
    LLMClient,
    SummaryParseError,
    chunk_lines,
    parse_markdown_summary,
    summarize_transcript,
)

JSON_ANSWER = {
    "title": "Falcon Dashboard Launch Planning",
    "executive_summary": "The team reviewed export status and agreed PDF export ships in v1.",
    "key_points": ["CSV export is done", "PDF export is ~70% complete"],
    "decisions": [{"decision": "PDF export ships in version one", "context": "Two customers asked for it"}],
    "action_items": [
        {"owner": "Priya", "task": "Finish PDF export", "due_date": "next Friday", "acceptance_criteria": ""},
        {"owner": "Marco", "task": "Add caching for the reporting API", "due_date": "", "acceptance_criteria": "Charts load under two seconds"},
        {"owner": "", "task": "Update the pricing page", "due_date": "before launch", "acceptance_criteria": "TBD"},
    ],
    "plan": [{"step": "Finish PDF export", "owner": "Priya", "timeframe": "next Friday"}],
    "open_questions": ["Launch date not confirmed by marketing"],
    "risks": [{"risk": "Reporting API rate limit during the demo", "mitigation": "Caching"}],
}

MARKDOWN_ANSWER = """# Falcon Dashboard Launch Planning

## Executive Summary
The team agreed PDF export ships in v1.

## Key Points
- CSV export is done

## Key Decisions
- PDF export ships in version one — two customers asked for it

## Action Items
| Owner | Task | Due date | Acceptance criteria |
|---|---|---|---|
| Priya | Finish PDF export | next Friday | unresolved |
| unresolved | Update the pricing page \\| copy | before launch | unresolved |

## Plan & Next Steps
1. Finish PDF export — owner: Priya; timeframe: next Friday

## Open Questions
- Launch date

## Risks & Blockers
- API rate limit — mitigation: caching
"""


def completion(content: str) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def mock_client(settings, handler) -> tuple[LLMClient, list[dict]]:
    requests: list[dict] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return handler(body, len(requests))

    return LLMClient(settings, transport=httpx.MockTransport(_handle)), requests


def test_json_mode_parses_into_schema(settings, default_template, transcript):
    client, requests = mock_client(settings, lambda body, n: httpx.Response(200, json=completion(json.dumps(JSON_ANSWER))))
    result = summarize_transcript(transcript, default_template, settings, client=client)

    assert result.mode == "json_schema"
    s = result.summary
    assert s.title == "Falcon Dashboard Launch Planning"
    assert s.decisions[0].decision == "PDF export ships in version one"
    assert [a.owner for a in s.action_items] == ["Priya", "Marco", UNRESOLVED]
    # blank / "TBD" fields are normalised to "unresolved" rather than guessed
    assert s.action_items[0].acceptance_criteria == UNRESOLVED
    assert s.action_items[1].due_date == UNRESOLVED
    assert s.action_items[2].acceptance_criteria == UNRESOLVED

    req = requests[0]
    assert req["model"] == "test-model"
    assert req["response_format"]["type"] == "json_schema"
    assert "action_items" in req["response_format"]["json_schema"]["schema"]["properties"]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Priya, can you give us the status" in req["messages"][1]["content"]


def test_think_tags_and_code_fences_are_stripped(settings, default_template, transcript):
    content = "<think>let me plan</think>\n```json\n" + json.dumps(JSON_ANSWER) + "\n```"
    client, _ = mock_client(settings, lambda b, n: httpx.Response(200, json=completion(content)))
    assert summarize_transcript(transcript, default_template, settings, client=client).summary.title


def test_custom_sections_parsed(settings, sales_template, transcript):
    answer = {
        "title": "Call",
        "executive_summary": "x",
        "key_points": [],
        "action_items": [],
        "plan": [],
        "risks": [],
        "custom_sections": {"customer_needs": ["PDF export"], "objections": [], "buying_signals": ["Q3 budget"]},
    }
    client, _ = mock_client(settings, lambda b, n: httpx.Response(200, json=completion(json.dumps(answer))))
    s = summarize_transcript(transcript, sales_template, settings, client=client).summary
    assert s.custom_sections == {"customer_needs": ["PDF export"], "objections": [], "buying_signals": ["Q3 budget"]}


def test_auto_mode_falls_back_to_markdown(settings, default_template, transcript):
    def handler(body, n):
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "json_schema not supported"}})
        return httpx.Response(200, json=completion(MARKDOWN_ANSWER))

    client, requests = mock_client(settings, handler)
    result = summarize_transcript(transcript, default_template, settings, client=client)
    assert result.mode == "markdown"
    assert len(requests) == 2 and "response_format" not in requests[1]
    assert "| Owner | Task |" in requests[1]["messages"][1]["content"]
    assert result.summary.action_items[1].task == "Update the pricing page | copy"
    assert result.summary.action_items[1].owner == UNRESOLVED


def test_auto_mode_falls_back_on_unparseable_json(settings, default_template, transcript):
    answers = iter([completion("Sorry, here are the minutes: not json"), completion(MARKDOWN_ANSWER)])
    client, _ = mock_client(settings, lambda b, n: httpx.Response(200, json=next(answers)))
    assert summarize_transcript(transcript, default_template, settings, client=client).mode == "markdown"


def test_strict_json_mode_does_not_fall_back(settings, default_template, transcript):
    strict = settings.with_overrides(llm_output_mode="json_schema")
    client, _ = mock_client(strict, lambda b, n: httpx.Response(200, json=completion("not json")))
    with pytest.raises(SummaryParseError):
        summarize_transcript(transcript, default_template, strict, client=client)


def test_markdown_parser_full(default_template):
    s = parse_markdown_summary(MARKDOWN_ANSWER, default_template)
    assert s.title == "Falcon Dashboard Launch Planning"
    assert s.decisions[0].context == "two customers asked for it"
    assert s.plan[0].owner == "Priya" and s.plan[0].timeframe == "next Friday"
    assert s.risks[0].mitigation == "caching"
    assert s.open_questions == ["Launch date"]


def test_server_unreachable_gives_actionable_error(settings, default_template, transcript):
    def handler(body, n):
        raise httpx.ConnectError("connection refused")

    client, _ = mock_client(settings, handler)
    with pytest.raises(LLMError) as exc:
        summarize_transcript(transcript, default_template, settings, client=client)
    assert "Can't reach llama-server" in str(exc.value)
    assert "MEETINGS_LLM_BASE_URL" in exc.value.hint


def test_unknown_model_hint(settings, default_template, transcript):
    strict = settings.with_overrides(llm_output_mode="markdown")
    client, _ = mock_client(strict, lambda b, n: httpx.Response(404, json={"error": {"message": "model not found"}}))
    with pytest.raises(LLMError) as exc:
        summarize_transcript(transcript, default_template, strict, client=client)
    assert "meetings check" in exc.value.hint


def test_long_transcript_is_chunked_then_merged(settings, default_template, transcript):
    small = settings.with_overrides(max_transcript_tokens=80)
    client, requests = mock_client(small, lambda b, n: httpx.Response(200, json=completion(json.dumps(JSON_ANSWER))))
    summarize_transcript(transcript, default_template, small, client=client)
    assert len(requests) >= 3  # >=2 parts + 1 merge
    assert "part 1 of" in requests[0]["messages"][1]["content"]
    assert "<partial_minutes>" in requests[-1]["messages"][1]["content"]


def test_chunk_lines_respects_budget():
    text = "\n".join(f"[00:00:{i:02d}] Speaker 1: " + "word " * 20 for i in range(30))
    chunks = chunk_lines(text, 200)
    assert len(chunks) > 1
    assert "\n".join(chunks) == text


def test_summarize_into_writes_files(settings, transcript, tmp_path):
    client, _ = mock_client(settings, lambda b, n: httpx.Response(200, json=completion(json.dumps(JSON_ANSWER))))
    summarize_into(transcript, tmp_path, settings, log=lambda m: None, client=client)
    md = (tmp_path / "summary.md").read_text()
    assert md.startswith("# Falcon Dashboard Launch Planning")
    assert "| 3 | unresolved | Update the pricing page | before launch | unresolved |" in md
    assert "**Speakers:** Speaker 1, Speaker 2, Speaker 3" in md
    assert json.loads((tmp_path / "summary.json").read_text())["mode"] == "json_schema"
    assert json.loads((tmp_path / "raw_llm_response.json").read_text())[0]["mode"] == "json_schema"


def test_raw_response_kept_when_parsing_fails(settings, transcript, tmp_path):
    strict = settings.with_overrides(llm_output_mode="json_schema")
    client, _ = mock_client(strict, lambda b, n: httpx.Response(200, json=completion("garbage")))
    with pytest.raises(SummaryParseError):
        summarize_into(transcript, tmp_path, strict, log=lambda m: None, client=client)
    assert (tmp_path / "raw_llm_response.json").exists()
    assert not (tmp_path / "summary.md").exists()
