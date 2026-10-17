import json
from pathlib import Path

import pytest

from meetings_server.config import Settings
from meetings_server.schema import Transcript
from meetings_server.templates import load_template

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.model_validate(json.loads((FIXTURES / "synthetic_transcript.json").read_text()))


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Placeholder host only — tests never touch the network (httpx.MockTransport).
    return Settings(
        llm_base_url="http://llm.invalid:8080/v1",
        llm_model="test-model",
        templates_dir=TEMPLATES_DIR,
        output_dir=tmp_path,
    )


@pytest.fixture
def default_template():
    return load_template("default", TEMPLATES_DIR)


@pytest.fixture
def sales_template():
    return load_template("sales-call", TEMPLATES_DIR)
