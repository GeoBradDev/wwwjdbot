"""Issue #2 review: log labels and the MatchVerseError message name
MatchVerse.match, not the removed match_verse function."""

import asyncio
import importlib
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FLAG = "WWJD_LOG_FULL_TEXT"
RID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def app_module(monkeypatch, caplog):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    caplog.set_level(logging.INFO)
    try:
        yield importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def test_run_guarded_failure_is_labelled_matchverse_match(app_module, caplog):
    def failing(text):
        raise RuntimeError()

    with pytest.raises(app_module.MatchVerseError) as info:
        app_module._run_guarded(RID, failing, "hello")
    assert str(info.value) == f"MatchVerse.match failed for request {RID}"
    assert f"[{RID}] MatchVerse.match failed: builtins.RuntimeError" in caplog.text
    assert "match_verse" not in caplog.text


def test_handler_500_is_labelled_matchverse_match_call(app_module, caplog):
    async def raising(text, request_id=None):
        raise RuntimeError()

    status, body = asyncio.run(app_module.handle_verse_request("hello", raising))
    assert status == 500
    assert f"[{body['id']}] MatchVerse.match call failed: builtins.RuntimeError" in caplog.text
    assert "match_verse" not in caplog.text
