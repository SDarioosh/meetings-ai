"""Settings, loaded from (lowest to highest priority): defaults, .env, environment, CLI flags.

All settings use the MEETINGS_ prefix, except HF_TOKEN (the standard Hugging Face name).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .errors import ConfigError

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_MODES = ("auto", "json_schema", "markdown")


def _default_templates_dir() -> Path:
    installed = PACKAGE_DIR / "_templates"
    if installed.is_dir():
        return installed
    return PACKAGE_DIR.parent / "templates"


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_output_mode: str = "auto"
    llm_disable_thinking: bool = True
    llm_timeout: float = 600.0
    llm_temperature: float = 0.2
    max_transcript_tokens: int = 60000

    # Transcription
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    language: str = ""
    unload_whisper: bool = True

    # Speaker detection
    diarize: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = field(default="", repr=False)
    num_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0

    # Output
    output_dir: Path = Path("./meetings")
    template: str = "default"
    templates_dir: Path = field(default_factory=_default_templates_dir)

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy with non-None overrides applied (used for CLI flags)."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def validate(self) -> None:
        if self.llm_output_mode not in OUTPUT_MODES:
            raise ConfigError(
                f"Invalid MEETINGS_LLM_OUTPUT_MODE={self.llm_output_mode!r}",
                f"Use one of: {', '.join(OUTPUT_MODES)}",
            )
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"MEETINGS_LLM_BASE_URL must start with http:// or https:// (got {self.llm_base_url!r})",
                "Example: MEETINGS_LLM_BASE_URL=http://192.168.1.10:8080/v1",
            )


def _coerce(raw: str, target: Any) -> Any:
    if isinstance(target, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(target, int):
        return int(raw) if raw.strip() else 0
    if isinstance(target, float):
        return float(raw)
    if isinstance(target, Path):
        return Path(raw).expanduser()
    return raw.strip()


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    """Build Settings from .env + environment. Does not override variables already set."""
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        # Look in the current directory, then the repo root (one level above server/).
        for candidate in (Path.cwd() / ".env", PACKAGE_DIR.parent.parent / ".env"):
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                break

    defaults = Settings()
    values: dict[str, Any] = {}
    for f in fields(Settings):
        env_name = "HF_TOKEN" if f.name == "hf_token" else f"MEETINGS_{f.name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None or (raw == "" and not isinstance(getattr(defaults, f.name), str)):
            continue
        try:
            values[f.name] = _coerce(raw, getattr(defaults, f.name))
        except ValueError as exc:
            raise ConfigError(f"Invalid value for {env_name}: {raw!r}", str(exc)) from exc

    settings = replace(defaults, **values)
    settings.validate()
    return settings
