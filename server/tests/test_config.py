import pytest

from meetings_server.config import load_settings
from meetings_server.errors import ConfigError


def test_env_vars_override_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("MEETINGS_LLM_BASE_URL", "http://llm.invalid:9000/v1")
    monkeypatch.setenv("MEETINGS_DIARIZE", "true")
    monkeypatch.setenv("MEETINGS_MAX_TRANSCRIPT_TOKENS", "1234")
    monkeypatch.setenv("HF_TOKEN", "placeholder")
    s = load_settings(tmp_path / "missing.env")
    assert s.llm_base_url == "http://llm.invalid:9000/v1"
    assert s.diarize is True
    assert s.max_transcript_tokens == 1234
    assert s.hf_token == "placeholder"
    assert "placeholder" not in repr(s)  # token never printed


def test_invalid_output_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("MEETINGS_LLM_OUTPUT_MODE", "yaml")
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "missing.env")


def test_cli_override_wins(settings):
    assert settings.with_overrides(template="standup", diarize=None).template == "standup"
