"""Review findings on issue #3: the regex fallback over-captured values,
missed fields followed by prose on a new line, decoded escapes only partly,
and accepted lone surrogates that Starlette cannot encode."""

import importlib
import json
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


def _starlette_render(content):
    # The exact call in starlette.responses.JSONResponse.render.
    return json.dumps(
        content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
    ).encode("utf-8")


# A: the fallback must not run a value into the next field


def test_fallback_fields_separated_by_newlines_without_commas(app_module):
    generated = '{"reference": "A"\n"verse": "B"\n"explanation": "C"}'
    assert _parse(app_module, generated) == _ok("A", "B", "C")


def test_fallback_fields_separated_by_spaces_without_commas(app_module):
    generated = '{"reference": "A" "verse": "B" "explanation": "C"}'
    assert _parse(app_module, generated) == _ok("A", "B", "C")


def test_fallback_stops_before_extra_key_with_digits(app_module):
    generated = '{"reference": "A", "verse": "B", "explanation": "C", "note2": "x",}'
    assert _parse(app_module, generated) == _ok("A", "B", "C")


def test_fallback_skips_key_mentioned_in_prose(app_module):
    generated = (
        'I will put the "reference": "x" style keys.'
        ' {"reference": "A", "verse": "B", "explanation": "C",}'
    )
    assert _parse(app_module, generated) == _ok("A", "B", "C")


# B: a value may end at a line end


def test_fallback_fields_followed_by_prose_on_a_new_line(app_module):
    generated = (
        '"reference": "Luke 6:31",\n"verse": "Do to others",\n'
        '"explanation": "Golden rule."\n\nI hope this helps!'
    )
    assert _parse(app_module, generated) == _ok("Luke 6:31", "Do to others", "Golden rule.")


# C: lone surrogates are never accepted


@pytest.mark.parametrize(
    "generated",
    [
        r'{"reference": "A", "verse": "lone \ud800", "explanation": "C",}',
        r'{"reference": "A", "verse": "lone \ud800", "explanation": "C"}',
    ],
)
def test_parse_rejects_lone_surrogate(app_module, generated):
    assert _parse(app_module, generated) == _failed(generated)


def test_handler_rejects_lone_surrogate_with_502(app_module):
    result = {"reference": "A", "verse": "lone \ud800", "explanation": "C"}
    status, body = app_module.handle_verse_request("Some statement", lambda text, request_id=None: result)
    assert status == 502
    assert body["error"] == app_module.ERROR_UNPARSEABLE


@pytest.mark.parametrize(
    ("generated", "expected_status"),
    [
        ('{"reference": "A", "verse": "B", "explanation": "C"}', 200),
        (r'{"reference": "A", "verse": "pair \ud834\udd1e", "explanation": "C"}', 200),
        (r'{"reference": "A", "verse": "He said "hi" \ud834\udd1e", "explanation": "C"}', 200),
        (r'{"reference": "A", "verse": "He said "hi" \u2019", "explanation": "C"}', 200),
        (r'{"reference": "A", "verse": "lone \ud800", "explanation": "C",}', 502),
        (r'{"reference": "A", "verse": "lone \ud800", "explanation": "C"}', 502),
        (r'{"reference": "A", "verse": "He said "hi" \udc00", "explanation": "C"}', 502),
    ],
)
def test_handler_200_payloads_render_with_starlette_json(app_module, generated, expected_status):
    status, body = app_module.handle_verse_request(
        "Some statement", lambda text, request_id=None: _parse(app_module, generated)
    )
    assert status == expected_status
    _starlette_render(body)


# D: escapes in a fallback value are decoded in one left-to-right pass


@pytest.mark.parametrize(
    ("verse_raw", "verse"),
    [
        (r'He said "hi" \u2019 \t x', 'He said "hi" \u2019 \t x'),
        (r'He said "hi" a\\nb', 'He said "hi" a\\nb'),
        (r'He said "hi" \ud834\udd1e', 'He said "hi" \U0001d11e'),
        (r'He said "hi" \x \q', 'He said "hi" \\x \\q'),
        (r'He said "hi" \"ok\" a\/b\n', 'He said "hi" "ok" a/b'),
    ],
)
def test_fallback_decodes_escapes_with_unescaped_quote(app_module, verse_raw, verse):
    generated = '{"reference": "A", "verse": "' + verse_raw + '", "explanation": "C"}'
    assert _parse(app_module, generated) == _ok("A", verse, "C")


def test_fallback_lone_surrogate_with_unescaped_quote_is_parse_failure(app_module):
    generated = r'{"reference": "A", "verse": "He said "hi" \ud800", "explanation": "C"}'
    assert _parse(app_module, generated) == _failed(generated)
