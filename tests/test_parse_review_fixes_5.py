"""Round 5 review on issue #3: a valid JSON object with all three keys that
does not qualify disabled the fallback for the whole output, so a later real
answer that is not valid JSON was lost. Such an object is now ignored and
the fallback reads only the rest of the output."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


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


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        (
            '{"reference": "A", "verse": "B", "explanation": ""}\n'
            '{"reference": "A", "verse": "B", "explanation": "real",}',
            ("A", "B", "real"),
        ),
        (
            'Format: {"reference": "", "verse": "", "explanation": ""}\nAnswer:\n'
            '{"reference": "Matthew 5:44", "verse": "Love your "enemies"", "explanation": "C"}',
            ("Matthew 5:44", 'Love your "enemies"', "C"),
        ),
        (
            '{"reference": null, "verse": null, "explanation": null}\n'
            '{"reference": "A", "verse": "B", "explanation": "C",}',
            ("A", "B", "C"),
        ),
        (
            '{"reference": "", "verse": "", "explanation": ""}\n'
            '"reference": "John 3:16",\n"verse": "For God",\n"explanation": "C"',
            ("John 3:16", "For God", "C"),
        ),
    ],
)
def test_non_qualifying_json_object_is_ignored_by_fallback(app_module, generated, expected):
    assert _parse(app_module, generated) == _ok(*expected)


def test_nested_non_qualifying_object_inside_invalid_outer_is_ignored(app_module):
    generated = (
        '{"reference": "A", "verse": "He said "love" them", "explanation": "C",'
        ' "example": {"reference": "x", "verse": "", "explanation": ""}}'
    )
    assert _parse(app_module, generated) == _ok("A", 'He said "love" them', "C")


def test_overlapping_non_qualifying_objects_are_both_ignored(app_module):
    generated = (
        '{"reference": "", "verse": "", "explanation": "",'
        ' "x": {"reference": "", "verse": "", "explanation": ""}}\n'
        '"reference": "A", "verse": "B", "explanation": "C"'
    )
    assert _parse(app_module, generated) == _ok("A", "B", "C")


@pytest.mark.parametrize(
    "generated",
    [
        '{"reference": "A", "verse": "B", "explanation": 5}\n"explanation": "C"',
        '{"reference": "A", "verse": "B", "explanation": ""}\nHe said "love"',
        '{\n"reference": "A",\n"verse": "B",\n"explanation": ""\n}\nHe said "love"',
    ],
)
def test_ignored_object_leaves_no_answer_is_parse_failure(app_module, generated):
    assert _parse(app_module, generated) == _failed(generated)
