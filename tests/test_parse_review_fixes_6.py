"""Round 6 review on issue #3: replacing an ignored JSON object with a space
joined a key to a later quote and hid text cut out of a fallback value. The
cut now leaves a sentinel, and a value that would span it is not used."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TEMPLATE = '{"reference": "", "verse": "", "explanation": ""}'

SPANS_THE_CUT = [
    '{"reference": "A", "verse": "B", "explanation": ' + TEMPLATE + '\n"love"}',
    '"reference": "A", "verse": "B", "explanation": ' + TEMPLATE + '\n"love"',
    '{"reference": "A", "verse": "B", "explanation": "Use ' + TEMPLATE + ' always"}',
    '"reference": "A", "verse": "He said ' + TEMPLATE + '", "explanation": "C"',
]


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        yield importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def _parse(app_module, generated):
    return app_module.parse_model_output(generated, MODEL, 1.234)


@pytest.mark.parametrize("generated", SPANS_THE_CUT)
def test_value_spanning_an_ignored_object_is_parse_failure(app_module, generated):
    assert _parse(app_module, generated) == {"error": "parse_failed", "raw_output": generated}


@pytest.mark.parametrize("generated", SPANS_THE_CUT)
def test_value_spanning_an_ignored_object_is_502(app_module, generated):
    status, body = app_module.handle_verse_request(
        "Some statement", lambda text, request_id=None: _parse(app_module, generated)
    )
    assert status == 502
    assert body["error"] == app_module.ERROR_UNPARSEABLE
