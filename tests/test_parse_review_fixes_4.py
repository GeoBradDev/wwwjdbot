"""Round 4 review on issue #3: an empty or whitespace value fell through to a
later terminator tier and captured text up to a later quote, giving a 200
with garbage. An empty value now moves to the key's next occurrence, and a
valid JSON object with all three keys that does not qualify is a parse
failure without the regex fallback."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

EMPTY_LAST_VALUE = [
    '{"reference": "A", "verse": "B", "explanation": ""}\nHe said "love"',
    '{"reference": "A", "verse": "B", "explanation": "  "}\nHe said "love"',
    '{\n"reference": "A",\n"verse": "B",\n"explanation": ""\n}\nHe said "love"',
    '"reference": "A",\n"verse": "B",\n"explanation": ""\nHe said "love"',
]


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        yield importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def _ok(reference, verse, explanation):
    return {
        "reference": reference,
        "verse": verse,
        "explanation": explanation,
        "model": MODEL,
        "generation_time": "1.23 seconds",
    }


def _failed(generated):
    return {"error": "parse_failed", "raw_output": generated}


def _parse(app_module, generated):
    return app_module.parse_model_output(generated, MODEL, 1.234)


@pytest.mark.parametrize("generated", EMPTY_LAST_VALUE)
def test_empty_last_value_before_later_quote_is_parse_failure(app_module, generated):
    assert _parse(app_module, generated) == _failed(generated)


@pytest.mark.parametrize("generated", EMPTY_LAST_VALUE)
def test_empty_last_value_before_later_quote_is_502(app_module, generated):
    status, body = app_module.handle_verse_request(
        "Some statement", lambda text, request_id=None: _parse(app_module, generated)
    )
    assert status == 502
    assert body["error"] == app_module.ERROR_UNPARSEABLE


def test_empty_value_moves_to_next_occurrence_of_key(app_module):
    generated = '{"reference": "A", "verse": "B", "explanation": "", "explanation": "real",}'
    assert _parse(app_module, generated) == _ok("A", "B", "real")


def test_valid_json_with_all_keys_that_does_not_qualify_skips_fallback(app_module):
    generated = '{"reference": "A", "verse": "B", "explanation": 5}\n"explanation": "C"'
    assert _parse(app_module, generated) == _failed(generated)
