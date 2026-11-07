"""Delta review on issue #3: the line-end terminator from review fix B cut a
value short at an unescaped inner quote before a raw newline. Strict
terminators must win for a candidate start; the line end is only a second
tier for the same start."""

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


def _parse(app_module, generated):
    return app_module.parse_model_output(generated, MODEL, 1.234)


def test_inner_quote_before_newline_keeps_full_value(app_module):
    generated = '{"reference": "A",\n"verse": "He said "Peace"\nand left",\n"explanation": "C",}'
    assert _parse(app_module, generated) == _ok("A", 'He said "Peace"\nand left', "C")


def test_braceless_inner_quote_before_newline_keeps_full_value(app_module):
    generated = '"reference": "A",\n"verse": "said "x"\nmore",\n"explanation": "C"'
    assert _parse(app_module, generated) == _ok("A", 'said "x"\nmore', "C")


def test_line_end_is_used_when_strict_value_contains_another_key(app_module):
    generated = '"verse": "B"\n\nThen "reference": "A", "explanation": "C"'
    assert _parse(app_module, generated) == _ok("A", "B", "C")


def test_prose_on_new_line_after_last_value_still_parses(app_module):
    generated = (
        '"reference": "Luke 6:31",\n"verse": "Do to others",\n'
        '"explanation": "Golden rule."\n\nI hope this helps!'
    )
    assert _parse(app_module, generated) == _ok("Luke 6:31", "Do to others", "Golden rule.")
