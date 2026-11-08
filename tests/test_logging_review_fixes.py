"""Issue #5 review round 1: the 500 path logs its status outcome, the two
containers' exception lines can be told apart, and an import with the debug
flag set warns that full text logging is on."""

import importlib
import logging
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FLAG = "WWJD_LOG_FULL_TEXT"
RID = "0123456789abcdef0123456789abcdef"


def _import(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    return importlib.import_module("wwjd_bot_modal")


@pytest.fixture
def app_module(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    try:
        yield _import(monkeypatch, None)
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def _raising(text, request_id=None):
    raise RuntimeError()


def _error_where(caplog):
    wheres = []
    for record in caplog.records:
        if record.levelno >= logging.ERROR:
            found = re.match(r"\[[0-9a-f]{32}\] (.+?) failed", record.getMessage())
            assert found, record.getMessage()
            wheres.append(found.group(1))
    return wheres


def test_500_path_logs_status_outcome(app_module, caplog):
    status, body = app_module.handle_verse_request("hello", _raising)
    assert status == 500
    assert f"[{body['id']}] done: status=500" in caplog.text


def test_verse_api_and_match_verse_exception_lines_differ(app_module, caplog):
    app_module.handle_verse_request("hello", _raising)
    handler_side = _error_where(caplog)
    caplog.clear()

    def failing(text):
        raise RuntimeError()

    with pytest.raises(app_module.MatchVerseError):
        app_module._run_guarded(RID, failing, "hello")
    match_side = _error_where(caplog)

    assert len(handler_side) == 1 and len(match_side) == 1
    assert handler_side[0] != match_side[0]


def _flag_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.name == "wwjd_bot"
        and record.levelno == logging.WARNING
        and FLAG in record.getMessage()
    ]


def test_import_with_flag_on_warns(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    try:
        _import(monkeypatch, "1")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)
    warnings = _flag_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage().lower()
    assert "user text" in message and "model output" in message


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_import_with_flag_off_does_not_warn(monkeypatch, caplog, flag):
    caplog.set_level(logging.INFO)
    try:
        _import(monkeypatch, flag)
    finally:
        sys.modules.pop("wwjd_bot_modal", None)
    assert _flag_warnings(caplog) == []
