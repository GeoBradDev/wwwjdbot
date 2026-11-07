"""Issue #3: verse_api maps match_verse results to status codes.

handle_verse_request holds the endpoint logic without importing FastAPI, so it
runs in CI; the verse_api wrapper is tested with fastapi.responses stubbed.
match_verse is never really called: no GPU, torch, transformers or Modal
credentials are needed."""

import asyncio
import importlib
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOOD = {
    "reference": "Matthew 5:44",
    "verse": "Love your enemies",
    "explanation": "Hate is not the way.",
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "generation_time": "1.23 seconds",
}
GOOD_RAW = {
    "raw": {
        "reference": "Matthew 5:44",
        "verse": "Love your enemies",
        "explanation": "Hate is not the way.",
    }
}


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        yield importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


class _Match:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        if self.exc is not None:
            raise self.exc
        return self.result


def _assert_error(app_module, status, body, expected_status, message):
    assert status == expected_status
    assert set(body) == {"error", "id"}
    assert body["error"] == message
    assert re.fullmatch(r"[0-9a-f]{32}", body["id"])


def test_success_returns_exactly_the_raw_shape(app_module):
    match = _Match(GOOD)
    status, body = app_module.handle_verse_request("Some statement", match)
    assert status == 200
    assert body == GOOD_RAW
    assert match.calls == ["Some statement"]


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_or_whitespace_text_is_422_without_calling_match(app_module, text):
    match = _Match(GOOD)
    status, body = app_module.handle_verse_request(text, match)
    _assert_error(app_module, status, body, 422, app_module.ERROR_EMPTY_TEXT)
    assert match.calls == []


def test_non_string_text_is_422(app_module):
    match = _Match(GOOD)
    status, body = app_module.handle_verse_request(None, match)
    _assert_error(app_module, status, body, 422, app_module.ERROR_EMPTY_TEXT)
    assert match.calls == []


def test_parse_failure_is_502_without_raw_output(app_module):
    raw_output = "the model rambled: SECRET-RAW-OUTPUT"
    match = _Match({"error": "parse_failed", "raw_output": raw_output})
    status, body = app_module.handle_verse_request("Some statement", match)
    _assert_error(app_module, status, body, 502, app_module.ERROR_UNPARSEABLE)
    assert "SECRET-RAW-OUTPUT" not in repr(body)


@pytest.mark.parametrize(
    "result",
    [
        {**GOOD, "verse": ""},
        {**GOOD, "explanation": "   "},
        {k: v for k, v in GOOD.items() if k != "reference"},
        {**GOOD, "reference": 5},
        {**GOOD, "verse": None},
        {**GOOD, "explanation": ["x"]},
        "not a dict",
        None,
    ],
)
def test_partial_non_string_or_non_dict_result_is_502(app_module, result):
    status, body = app_module.handle_verse_request("Some statement", _Match(result))
    _assert_error(app_module, status, body, 502, app_module.ERROR_UNPARSEABLE)


def test_exception_from_match_is_generic_500(app_module):
    match = _Match(exc=RuntimeError("secret detail"))
    status, body = app_module.handle_verse_request("Some statement", match)
    _assert_error(app_module, status, body, 500, app_module.ERROR_INTERNAL)
    assert "secret detail" not in repr(body)


def test_error_ids_differ_between_calls(app_module):
    _, first = app_module.handle_verse_request("", _Match(GOOD))
    _, second = app_module.handle_verse_request("", _Match(GOOD))
    assert first["id"] != second["id"]


class _JSONResponse:
    def __init__(self, status_code=200, content=None):
        self.status_code = status_code
        self.content = content


def test_verse_api_wrapper_returns_json_response(app_module, monkeypatch):
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _JSONResponse
    fastapi = types.ModuleType("fastapi")
    fastapi.responses = responses
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)

    remote = _Match(GOOD)
    monkeypatch.setattr(app_module, "match_verse", types.SimpleNamespace(remote=remote))

    raw = inspect.unwrap(app_module.verse_api._raw_f_)
    assert inspect.iscoroutinefunction(raw)

    response = asyncio.run(raw(app_module.VerseRequest(text="Some statement")))
    assert isinstance(response, _JSONResponse)
    assert response.status_code == 200
    assert response.content == GOOD_RAW
    assert remote.calls == ["Some statement"]

    response = asyncio.run(raw(app_module.VerseRequest(text="  ")))
    assert isinstance(response, _JSONResponse)
    assert response.status_code == 422
    assert response.content["error"] == app_module.ERROR_EMPTY_TEXT
    assert remote.calls == ["Some statement"]
