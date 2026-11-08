"""Round 3 review on issue #3: the strict end-of-text terminator ran the last
value into quoted prose on a later line. A value now ends at the next key or
a closing brace first, then at a line end, then at the end of the output."""

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


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        (
            '"reference": "A",\n"verse": "B",\n"explanation": "C"\n\nAs Paul wrote, "Love is patient."',
            ("A", "B", "C"),
        ),
        (
            '"reference": "Matthew 22:39",\n"verse": "Love your neighbor as yourself.",\n'
            '"explanation": "Be kind."\n\nRemember, Jesus said "love one another"',
            ("Matthew 22:39", "Love your neighbor as yourself.", "Be kind."),
        ),
        (
            '{"reference": "A",\n"verse": "B",\n"explanation": "C"\n\nHope this helps "friend"',
            ("A", "B", "C"),
        ),
    ],
)
def test_last_value_stops_before_quoted_prose_on_a_later_line(app_module, generated, expected):
    assert _parse(app_module, generated) == _ok(*expected)


def test_peace_inner_quote_before_newline_keeps_full_value(app_module):
    generated = '{"reference": "A",\n"verse": "He said "Peace"\nand left",\n"explanation": "C",}'
    assert _parse(app_module, generated) == _ok("A", 'He said "Peace"\nand left', "C")


def test_braceless_peace_inner_quote_before_newline_keeps_full_value(app_module):
    generated = '"reference": "A",\n"verse": "He said "Peace"\nand left",\n"explanation": "C"'
    assert _parse(app_module, generated) == _ok("A", 'He said "Peace"\nand left', "C")


def test_prose_on_new_line_after_last_value_still_parses(app_module):
    generated = (
        '"reference": "Luke 6:31",\n"verse": "Do to others",\n'
        '"explanation": "Golden rule."\n\nI hope this helps!'
    )
    assert _parse(app_module, generated) == _ok("Luke 6:31", "Do to others", "Golden rule.")


def test_braceless_output_ending_at_last_value_still_parses(app_module):
    generated = '"reference": "A",\n"verse": "B",\n"explanation": "C"'
    assert _parse(app_module, generated) == _ok("A", "B", "C")
