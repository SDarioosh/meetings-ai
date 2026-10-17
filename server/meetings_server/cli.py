"""`meetings` command-line interface."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .audio import SUPPORTED_EXTENSIONS, require_ffmpeg
from .config import Settings, load_settings
from .errors import MeetingsError
from .schema import Transcript
from .templates import list_templates, load_template

app = typer.Typer(
    help="Meetings.ai — local meeting transcription, speaker detection and summaries.",
    no_args_is_help=True,
    add_completion=False,
)


def _log(msg: str) -> None:
    typer.echo(msg, err=True)


def _fail(exc: MeetingsError) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    if exc.hint:
        typer.secho(f"fix:   {exc.hint}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(1)


def _settings(env_file: Optional[Path], **overrides) -> Settings:
    try:
        settings = load_settings(env_file).with_overrides(**overrides)
        settings.validate()
        return settings
    except MeetingsError as exc:
        _fail(exc)
        raise  # unreachable


EnvOpt = typer.Option(None, "--env-file", help="Path to a .env file (default: ./.env or the repo's .env).")
TemplateOpt = typer.Option(None, "--template", "-t", help="Summary template name or path to a .yaml file.")
OutOpt = typer.Option(None, "--output-dir", "-o", help="Where meeting folders are written.")


@app.command()
def process(
    audio_file: Path = typer.Argument(..., help="Recording to process (wav, m4a, mp3, ...)."),
    diarize: Optional[bool] = typer.Option(None, "--diarize/--no-diarize", help="Detect and label speakers."),
    output_dir: Optional[Path] = OutOpt,
    template: Optional[str] = TemplateOpt,
    whisper_model: Optional[str] = typer.Option(None, "--whisper-model", "-m", help="e.g. large-v3-turbo, small."),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Spoken language code, e.g. en (default: auto)."),
    num_speakers: Optional[int] = typer.Option(None, "--speakers", help="Exact number of speakers, if known."),
    print_summary: bool = typer.Option(True, "--print/--no-print", help="Print summary.md when done."),
    env_file: Optional[Path] = EnvOpt,
) -> None:
    """Transcribe, (optionally) label speakers, and summarize one recording."""
    from .pipeline import process_file

    settings = _settings(
        env_file,
        diarize=diarize,
        output_dir=output_dir,
        template=template,
        whisper_model=whisper_model,
        language=language,
        num_speakers=num_speakers,
    )
    try:
        out_dir = process_file(audio_file, settings, log=_log)
    except MeetingsError as exc:
        _fail(exc)
    _log(f"\nOutput: {out_dir}")
    if print_summary:
        typer.echo("\n" + (out_dir / "summary.md").read_text(encoding="utf-8"))


@app.command()
def summarize(
    transcript_json: Path = typer.Argument(..., help="A transcript.json written by `meetings process`."),
    template: Optional[str] = TemplateOpt,
    print_summary: bool = typer.Option(True, "--print/--no-print"),
    env_file: Optional[Path] = EnvOpt,
) -> None:
    """Re-run only the summary for an existing transcript (e.g. with a different template)."""
    from datetime import datetime

    from .pipeline import summarize_into

    settings = _settings(env_file, template=template)
    if not transcript_json.is_file():
        _fail(MeetingsError(f"Transcript not found: {transcript_json}", "Pass the path to a transcript.json."))
    try:
        transcript = Transcript.model_validate_json(transcript_json.read_text(encoding="utf-8"))
        out_dir = transcript_json.parent
        summarize_into(transcript, out_dir, settings, datetime.fromtimestamp(transcript_json.stat().st_mtime), _log)
    except MeetingsError as exc:
        _fail(exc)
    except ValueError as exc:
        _fail(MeetingsError(f"{transcript_json} is not a valid transcript: {exc}", ""))
    if print_summary:
        typer.echo("\n" + (out_dir / "summary.md").read_text(encoding="utf-8"))


@app.command()
def watch(
    inbox: Path = typer.Argument(..., help="Folder to watch for new recordings."),
    diarize: Optional[bool] = typer.Option(None, "--diarize/--no-diarize"),
    output_dir: Optional[Path] = OutOpt,
    template: Optional[str] = TemplateOpt,
    interval: float = typer.Option(5.0, help="Seconds between checks."),
    env_file: Optional[Path] = EnvOpt,
) -> None:
    """Process every recording dropped into INBOX. Done files move to INBOX/processed, failures to INBOX/failed."""
    from .pipeline import process_file

    settings = _settings(env_file, diarize=diarize, output_dir=output_dir, template=template)
    inbox = inbox.expanduser().resolve()
    done_dir, failed_dir = inbox / "processed", inbox / "failed"
    for d in (inbox, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    _log(f"Watching {inbox} (Ctrl+C to stop). Outputs go to {settings.output_dir.resolve()}")
    sizes: dict[Path, int] = {}
    try:
        while True:
            for f in sorted(inbox.iterdir()):
                if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                size = f.stat().st_size
                # Only process once the size has stopped changing (the recorder finished writing).
                if sizes.get(f) != size or size == 0:
                    sizes[f] = size
                    continue
                sizes.pop(f, None)
                try:
                    process_file(f, settings, log=_log)
                    shutil.move(str(f), done_dir / f.name)
                except MeetingsError as exc:
                    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
                    if exc.hint:
                        typer.secho(f"fix:   {exc.hint}", fg=typer.colors.YELLOW, err=True)
                    shutil.move(str(f), failed_dir / f.name)
            time.sleep(interval)
    except KeyboardInterrupt:
        _log("Stopped.")


@app.command()
def templates(env_file: Optional[Path] = EnvOpt) -> None:
    """List available summary templates."""
    settings = _settings(env_file)
    for name in list_templates(settings.templates_dir):
        try:
            desc = load_template(name, settings.templates_dir).description
        except MeetingsError as exc:
            desc = f"(invalid: {exc})"
        typer.echo(f"{name:<16} {desc}")
    typer.echo(f"\nTemplates folder: {settings.templates_dir}")


@app.command()
def check(env_file: Optional[Path] = EnvOpt) -> None:
    """Check that ffmpeg, Whisper, the LLM server and (if enabled) speaker detection are ready."""
    from .summarize import LLMClient
    from .transcribe import resolve_device

    settings = _settings(env_file)
    ok = True

    def report(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok &= good
        mark = typer.style("✓", fg=typer.colors.GREEN) if good else typer.style("✗", fg=typer.colors.RED)
        typer.echo(f"{mark} {name:<18} {detail}")

    typer.echo(f"meetings {__version__} (Python {sys.version.split()[0]})\n")

    try:
        report("ffmpeg", True, require_ffmpeg())
    except MeetingsError as exc:
        report("ffmpeg", False, exc.hint)

    try:
        import faster_whisper  # noqa: F401

        device, compute = resolve_device(settings)
        report("faster-whisper", True, f"model {settings.whisper_model} on {device} ({compute})")
    except ImportError:
        report("faster-whisper", False, 'not installed — pip install -e "server"')

    client = LLMClient(settings)
    try:
        models = client.list_models()
        if settings.llm_model and settings.llm_model not in models:
            report("llama-server", False, f"{settings.llm_base_url} is up, but model {settings.llm_model!r} isn't listed. Available: {', '.join(models)}")
        else:
            report("llama-server", True, f"{settings.llm_base_url} — models: {', '.join(models) or '(none listed)'}")
    except MeetingsError as exc:
        report("llama-server", False, f"{exc} {exc.hint}")
    finally:
        client.close()

    try:
        t = load_template(settings.template, settings.templates_dir)
        report("template", True, f"{t.name} ({len(t.sections)} sections, {len(t.custom_sections)} custom)")
    except MeetingsError as exc:
        report("template", False, f"{exc} {exc.hint}")

    if settings.diarize:
        try:
            import pyannote.audio  # noqa: F401

            report("speaker detection", bool(settings.hf_token), "ready" if settings.hf_token else "HF_TOKEN is not set")
        except ImportError:
            report("speaker detection", False, 'pyannote not installed — pip install -e "server[diarize]"')
    else:
        typer.echo("- speaker detection  off (enable with --diarize or MEETINGS_DIARIZE=true)")

    raise typer.Exit(0 if ok else 1)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
