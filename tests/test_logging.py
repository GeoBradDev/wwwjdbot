"""Issue #5: user text and model output must not reach the logs by default,
and exception text must never reach a response body.

Fakes raise prebuilt exceptions, so no traceback source line contains a
sentinel; a sentinel in the logs can only come from a logged message."""

import ast
import importlib
import inspect
import logging
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = ROOT / "wwjd_bot_modal.py"
FLAG = "WWJD_LOG_FULL_TEXT"

TEXT = "SENTINEL-USER-TEXT jane@example.com"
OUTPUT = "SENTINEL-MODEL-OUTPUT the model rambled"
SENTINELS = ("SENTINEL-USER-TEXT", "jane@example.com", "SENTINEL-MODEL-OUTPUT")
GOOD = {
    "reference": "Matthew 5:44",
    "verse": "Love your enemies",
    "explanation": "Hate is not the way.",
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "generation_time": "1.23 seconds",
}
ID_RE = re.compile(r"[0-9a-f]{32}")


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


class _Match:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.request_ids = []

    def __call__(self, text, request_id=None):
        self.request_ids.append(request_id)
        if self.exc is not None:
            raise self.exc
        return self.result


def _chained_exception():
    exc = RuntimeError(f"tokenizer choked on {TEXT}")
    exc.__cause__ = ValueError(f"bad output {OUTPUT}")
    return exc


def _logged(caplog):
    parts = [caplog.text]
    for record in caplog.records:
        parts.append(record.getMessage())
        if record.exc_info:
            parts.append(logging.Formatter().formatException(record.exc_info))
        if record.exc_text:
            parts.append(record.exc_text)
    return "\n".join(parts)


def _assert_no_sentinel(text):
    for sentinel in SENTINELS:
        assert sentinel not in text, f"{sentinel!r} leaked: {text!r}"


# Criterion 3


def test_module_has_no_print_calls():
    tree = ast.parse(APP_FILE.read_text())
    prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert prints == [], f"print() calls on lines {prints}"


def test_module_logger_is_wwjd_bot_at_info(app_module):
    assert app_module.logger is logging.getLogger("wwjd_bot")
    assert app_module.logger.isEnabledFor(logging.INFO)
    assert not app_module.logger.isEnabledFor(logging.DEBUG)


def test_reimport_does_not_stack_handlers(app_module, monkeypatch):
    count = len(logging.getLogger("wwjd_bot").handlers)
    sys.modules.pop("wwjd_bot_modal", None)
    importlib.import_module("wwjd_bot_modal")
    assert len(logging.getLogger("wwjd_bot").handlers) == count


# Criteria 1 and 4: default logs carry metadata only


def test_success_logs_metadata_only(app_module, caplog):
    match = _Match(GOOD)
    status, _ = app_module.handle_verse_request(TEXT, match)
    assert status == 200
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert match.request_ids[0] in logs
    assert f"input_length={len(TEXT)}" in logs
    assert "status=200" in logs


def test_422_logs_metadata_only(app_module, caplog):
    status, body = app_module.handle_verse_request("   ", _Match(GOOD))
    assert status == 422
    logs = _logged(caplog)
    assert body["id"] in logs
    assert "status=422" in logs
    assert "reason=empty_text" in logs


@pytest.mark.parametrize(
    "result,reason",
    [
        ({"error": "parse_failed", "raw_output": OUTPUT}, "match_error"),
        (f"not a dict {OUTPUT}", "not_dict"),
        ({**GOOD, "verse": "", "raw_output": OUTPUT}, "invalid_fields"),
        ({**GOOD, "reference": OUTPUT, "explanation": 5}, "invalid_fields"),
    ],
)
def test_502_logs_reason_category_only(app_module, caplog, result, reason):
    status, body = app_module.handle_verse_request(TEXT, _Match(result))
    assert status == 502
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert body["id"] in logs
    assert "status=502" in logs
    assert f"reason={reason}" in logs


def test_500_logs_type_names_and_frames_but_no_messages(app_module, caplog):
    status, body = app_module.handle_verse_request(TEXT, _Match(exc=_chained_exception()))
    assert status == 500
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert "choked" not in logs
    assert "bad output" not in logs
    assert body["id"] in logs
    assert "builtins.RuntimeError <- builtins.ValueError" in logs
    assert "Traceback (most recent call last)" in logs
    assert "in __call__" in logs
    assert "test_logging.py" in logs


def test_500_without_cause_logs_context_type(app_module, caplog):
    def failing(text, request_id=None):
        try:
            raise KeyError(TEXT)
        except KeyError:
            raise TypeError(f"wrapped {TEXT}")  # noqa: B904 the context is the point

    status, _ = app_module.handle_verse_request(TEXT, failing)
    assert status == 500
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert "wrapped" not in logs
    assert "builtins.TypeError <- builtins.KeyError" in logs


# Criterion 2: no exception text or raw output in bodies, flag off or on


@pytest.mark.parametrize("flag", [None, "1"])
@pytest.mark.parametrize(
    "match,expected_status",
    [
        (_Match(exc=_chained_exception()), 500),
        (_Match({"error": "parse_failed", "raw_output": OUTPUT}), 502),
        (_Match({**GOOD, "verse": None, "raw_output": OUTPUT}), 502),
    ],
)
def test_error_bodies_hold_no_exception_text_or_output(
    app_module, monkeypatch, flag, match, expected_status
):
    if flag is not None:
        monkeypatch.setenv(FLAG, flag)
    status, body = app_module.handle_verse_request(TEXT, match)
    assert status == expected_status
    assert set(body) == {"error", "id"}
    assert ID_RE.fullmatch(body["id"])
    _assert_no_sentinel(repr(body))
    assert "choked" not in repr(body)


# Criterion 5: the debug flag


def test_flag_on_logs_user_text(app_module, monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    app_module.handle_verse_request(TEXT, _Match(GOOD))
    assert TEXT in _logged(caplog)


def test_flag_on_logs_raw_output_on_502(app_module, monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    app_module.handle_verse_request(TEXT, _Match({"error": "parse_failed", "raw_output": OUTPUT}))
    assert OUTPUT in _logged(caplog)


def test_flag_on_logs_full_exception(app_module, monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    app_module.handle_verse_request(TEXT, _Match(exc=_chained_exception()))
    logs = _logged(caplog)
    assert f"tokenizer choked on {TEXT}" in logs
    assert f"bad output {OUTPUT}" in logs


@pytest.mark.parametrize("value", ["0", "true", "", "yes"])
def test_other_flag_values_are_off(app_module, monkeypatch, caplog, value):
    monkeypatch.setenv(FLAG, value)
    app_module.handle_verse_request(TEXT, _Match(GOOD))
    app_module.handle_verse_request(TEXT, _Match({"error": "parse_failed", "raw_output": OUTPUT}))
    app_module.handle_verse_request(TEXT, _Match(exc=_chained_exception()))
    _assert_no_sentinel(_logged(caplog))


def test_full_text_image_env(app_module, monkeypatch):
    assert app_module._full_text_image_env() == {}
    monkeypatch.setenv(FLAG, "true")
    assert app_module._full_text_image_env() == {}
    monkeypatch.setenv(FLAG, "1")
    assert app_module._full_text_image_env() == {"WWJD_LOG_FULL_TEXT": "1"}


# _log_generation (match_verse's logging helper)

GEN_OK = '{"reference": "Matthew 5:44", "verse": "Love your enemies", "explanation": "' + OUTPUT + '"}'
GEN_BAD = f"no json here {OUTPUT}"
RID = "0123456789abcdef0123456789abcdef"


def test_log_generation_default_logs_metadata_only(app_module, caplog):
    prompt = app_module.build_prompt(TEXT)
    result = app_module._log_generation(RID, TEXT, prompt, GEN_OK, 1.234, "m")
    assert result == {
        "reference": "Matthew 5:44",
        "verse": "Love your enemies",
        "explanation": OUTPUT,
        "model": "m",
        "generation_time": "1.23 seconds",
    }
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert "[INST]" not in logs and "Christian" not in logs
    assert RID in logs
    assert f"input_length={len(TEXT)}" in logs
    assert f"output_length={len(GEN_OK)}" in logs
    assert "generation_time=1.23s" in logs
    assert "parse=ok" in logs


def test_log_generation_reports_parse_failure(app_module, caplog):
    result = app_module._log_generation(RID, TEXT, "prompt", GEN_BAD, 0.5, "m")
    assert result == {"error": "parse_failed", "raw_output": GEN_BAD}
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert "parse=parse_failed" in logs
    assert "generation_time=0.50s" in logs


def test_log_generation_flag_on_logs_prompt_and_output(app_module, monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    prompt = app_module.build_prompt(TEXT)
    app_module._log_generation(RID, TEXT, prompt, GEN_BAD, 0.5, "m")
    logs = _logged(caplog)
    assert prompt in logs
    assert GEN_BAD in logs


# Criterion 6: request id plumbing


@pytest.mark.parametrize(
    "match",
    [_Match(exc=_chained_exception()), _Match({"error": "parse_failed", "raw_output": OUTPUT})],
)
def test_match_receives_the_body_id(app_module, caplog, match):
    _, body = app_module.handle_verse_request(TEXT, match)
    assert match.request_ids == [body["id"]]
    assert body["id"] in _logged(caplog)


def test_match_receives_a_request_id_on_success(app_module):
    match = _Match(GOOD)
    app_module.handle_verse_request(TEXT, match)
    assert len(match.request_ids) == 1
    assert ID_RE.fullmatch(match.request_ids[0])


def test_normalize_request_id_keeps_a_valid_id(app_module):
    assert app_module._normalize_request_id(RID) == RID


@pytest.mark.parametrize(
    "value", [None, "", RID.upper(), RID + "0", RID[:-1], TEXT, 12345, RID + "\n"]
)
def test_normalize_request_id_replaces_anything_else(app_module, value):
    result = app_module._normalize_request_id(value)
    assert ID_RE.fullmatch(result)
    assert result != value


# Criterion 7: match_verse never lets the original exception out


def test_run_guarded_raises_bare_match_verse_error(app_module, caplog):
    exc = _chained_exception()

    def failing(text):
        raise exc

    with pytest.raises(app_module.MatchVerseError) as info:
        app_module._run_guarded(RID, failing, TEXT)
    error = info.value
    assert RID in str(error)
    _assert_no_sentinel(str(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert RID in logs
    assert "builtins.RuntimeError <- builtins.ValueError" in logs
    assert "in failing" in logs


def test_run_guarded_passes_the_return_value_through(app_module):
    assert app_module._run_guarded(RID, lambda a, b: {"sum": a + b}, 1, 2) == {"sum": 3}


def test_match_verse_accepts_request_id_defaulting_to_none(app_module):
    raw = inspect.unwrap(app_module.match_verse._raw_f_)
    params = inspect.signature(raw).parameters
    assert list(params) == ["text", "request_id"]
    assert params["request_id"].default is None


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.mark.filterwarnings("ignore:The match_verse function is executing locally")
def test_match_verse_model_failure_surfaces_as_match_verse_error(app_module, monkeypatch, caplog):
    exc = OSError(f"could not tokenize {TEXT}")

    def _from_pretrained(*args, **kwargs):
        raise exc

    loader = types.SimpleNamespace(from_pretrained=_from_pretrained)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _stub_module(
            "transformers",
            AutoTokenizer=loader,
            AutoModelForCausalLM=loader,
            BitsAndBytesConfig=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _stub_module("torch", float16="float16"))
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_fake")

    with pytest.raises(app_module.MatchVerseError) as info:
        app_module.match_verse.local(TEXT, request_id=RID)
    assert info.value.__cause__ is None and info.value.__context__ is None
    _assert_no_sentinel(str(info.value))
    logs = _logged(caplog)
    _assert_no_sentinel(logs)
    assert RID in logs
    assert "builtins.OSError" in logs
