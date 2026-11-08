"""Issue #12: GPU cost is bounded by a container cap, a prompt token cap and a
request body byte cap.

No GPU, torch, transformers, FastAPI or Modal credentials are needed: the
tokenizer, model, fastapi modules and the request are fakes."""

import ast
import asyncio
import contextlib
import importlib
import inspect
import json
import logging
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = ROOT / "wwjd_bot_modal.py"
FLAG = "WWJD_LOG_FULL_TEXT"
CAP_ENV = "WWJD_MAX_GPU_CONTAINERS"
ID_RE = re.compile(r"[0-9a-f]{32}")


def _import_app(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    return importlib.import_module("wwjd_bot_modal")


@pytest.fixture
def app_module(monkeypatch, caplog):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(CAP_ENV, raising=False)
    caplog.set_level(logging.INFO)
    try:
        yield _import_app(monkeypatch)
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


@pytest.fixture
def clean_import(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(CAP_ENV, raising=False)
    try:
        yield lambda: _import_app(monkeypatch)
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


# Criterion 1: GPU container cap


def _class_def(name):
    tree = ast.parse(APP_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no class {name}")


def test_match_verse_passes_max_containers_from_the_constant():
    cls = _class_def("MatchVerse")
    calls = [
        d for d in cls.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "app.cls"
    ]
    assert len(calls) == 1
    kwargs = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kwargs["max_containers"] == "MAX_GPU_CONTAINERS"


def test_max_gpu_containers_defaults_to_2(clean_import):
    assert clean_import().MAX_GPU_CONTAINERS == 2


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_max_gpu_containers_means_the_default(clean_import, monkeypatch, value):
    monkeypatch.setenv(CAP_ENV, value)
    assert clean_import().MAX_GPU_CONTAINERS == 2


@pytest.mark.parametrize(("value", "expected"), [("3", 3), (" 3 ", 3), ("1", 1)])
def test_max_gpu_containers_reads_the_env_var(clean_import, monkeypatch, value, expected):
    monkeypatch.setenv(CAP_ENV, value)
    assert clean_import().MAX_GPU_CONTAINERS == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", "+2", "2x"])
def test_bad_max_gpu_containers_fails_the_import(clean_import, monkeypatch, value):
    monkeypatch.setenv(CAP_ENV, value)
    with pytest.raises(ValueError) as info:
        clean_import()
    assert CAP_ENV in str(info.value)
    assert value not in str(info.value).replace(CAP_ENV, "")


# Criterion 2: prompt token cap

SENTINEL = "SENTINEL-TOKEN-TEXT"
GENERATED = '{"reference": "Matthew 5:44", "verse": "Love your enemies", "explanation": "Hate is not the way."}'
GOOD = {
    "reference": "Matthew 5:44",
    "verse": "Love your enemies",
    "explanation": "Hate is not the way.",
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "generation_time": "1.23 seconds",
}
GOOD_RAW = {"raw": {k: GOOD[k] for k in ("reference", "verse", "explanation")}}


class _Inputs(dict):
    def __init__(self, length):
        super().__init__(input_ids=None)
        self.input_ids = types.SimpleNamespace(shape=(1, length))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _Tokenizer:
    eos_token = "</s>"
    eos_token_id = 2

    def __init__(self, length):
        self.length = length
        self.calls = []

    def __call__(self, prompt, return_tensors=None):
        self.calls.append((prompt, return_tensors))
        return _Inputs(self.length)

    def decode(self, tokens, skip_special_tokens=False):
        return GENERATED


class _Model:
    device = "cpu"

    def __init__(self):
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        return [[0, 1, 2, 3, 4]]


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(no_grad=contextlib.nullcontext, float16="float16"),
    )


def test_max_prompt_tokens_constant(app_module):
    assert app_module.MAX_PROMPT_TOKENS == 6144
    assert app_module.ERROR_TOO_MANY_TOKENS == "The 'text' field is too long for the model; select less text."


@pytest.mark.parametrize("below_cap", [None, 1, 0])
def test_tokenize_capped_returns_the_inputs_within_the_cap(app_module, below_cap):
    cap = app_module.MAX_PROMPT_TOKENS
    length = 1 if below_cap is None else cap - below_cap
    tokenizer = _Tokenizer(length)
    inputs, count = app_module._tokenize_capped(tokenizer, "a prompt", cap)
    assert inputs is not None
    assert inputs.input_ids.shape == (1, length)
    assert count == length
    assert tokenizer.calls == [("a prompt", "pt")]


def test_tokenize_capped_returns_none_over_the_cap(app_module):
    cap = app_module.MAX_PROMPT_TOKENS
    tokenizer = _Tokenizer(cap + 1)
    inputs, count = app_module._tokenize_capped(tokenizer, "a prompt", cap)
    assert inputs is None
    assert count == cap + 1
    assert tokenizer.calls == [("a prompt", "pt")]


def test_generate_verse_at_the_cap_generates(app_module, fake_torch):
    tokenizer, model = _Tokenizer(app_module.MAX_PROMPT_TOKENS), _Model()
    result = app_module._generate_verse(tokenizer, model, "Some statement", "0" * 32)
    assert model.generate_calls == 1
    assert {k: result[k] for k in ("reference", "verse", "explanation")} == GOOD_RAW["raw"]
    assert tokenizer.calls[0][0] == app_module.build_prompt("Some statement")


def test_generate_verse_over_the_cap_never_generates(app_module, fake_torch, caplog):
    cap = app_module.MAX_PROMPT_TOKENS
    tokenizer, model = _Tokenizer(cap + 1), _Model()
    text = SENTINEL + " more words"
    request_id = "0123456789abcdef0123456789abcdef"
    result = app_module._generate_verse(tokenizer, model, text, request_id)
    assert result == {"error": "input_too_long"}
    assert model.generate_calls == 0
    assert SENTINEL not in caplog.text
    assert (
        f"[{request_id}] rejected: reason=input_too_long input_length={len(text)} "
        f"prompt_tokens={cap + 1} max_prompt_tokens={cap}"
    ) in caplog.text


def test_input_too_long_passes_through_run_guarded(app_module, fake_torch):
    tokenizer, model = _Tokenizer(app_module.MAX_PROMPT_TOKENS + 1), _Model()
    result = app_module._run_guarded(
        "0" * 32, app_module._generate_verse, tokenizer, model, "x", "0" * 32
    )
    assert result == {"error": "input_too_long"}


class _AsyncMatch:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def __call__(self, text, request_id=None):
        self.calls.append(text)
        return self.result


def test_handle_verse_request_maps_input_too_long_to_422(app_module, caplog):
    text = SENTINEL + " statement"
    match = _AsyncMatch({"error": "input_too_long"})
    status, body = asyncio.run(app_module.handle_verse_request(text, match))
    assert status == 422
    assert set(body) == {"error", "id"}
    assert body["error"] == app_module.ERROR_TOO_MANY_TOKENS
    assert ID_RE.fullmatch(body["id"])
    assert SENTINEL not in repr(body)
    assert SENTINEL not in caplog.text
    assert f"[{body['id']}] done: status=422 reason=input_too_long" in caplog.text
    assert "status=502" not in caplog.text


def test_other_match_errors_stay_502(app_module):
    match = _AsyncMatch({"error": "parse_failed", "raw_output": "x"})
    status, body = asyncio.run(app_module.handle_verse_request("Some statement", match))
    assert status == 502
    assert body["error"] == app_module.ERROR_UNPARSEABLE


# Criterion 3: request body byte cap


class _Stream:
    """Async iterable of chunks that records how many were read."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read = 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self.chunks:
            self.read += 1
            yield chunk


class _FakeRequest:
    def __init__(self, body=b"", headers=None, chunks=None):
        self.headers = {"content-length": str(len(body))} if headers is None else headers
        self._stream = _Stream(chunks if chunks is not None else [body, b""])

    def stream(self):
        return self._stream


class _JSONResponse:
    def __init__(self, status_code=200, content=None):
        self.status_code = status_code
        self.content = content


class _RequestValidationError(Exception):
    def __init__(self, errors, *, body=None):
        super().__init__(errors)
        self._errors = list(errors)
        self.body = body

    def errors(self):
        return self._errors


@pytest.fixture
def stub_fastapi(monkeypatch):
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _JSONResponse
    exceptions = types.ModuleType("fastapi.exceptions")
    exceptions.RequestValidationError = _RequestValidationError
    fastapi = types.ModuleType("fastapi")
    fastapi.responses = responses
    fastapi.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)
    monkeypatch.setitem(sys.modules, "fastapi.exceptions", exceptions)


@pytest.fixture
def gpu(app_module, monkeypatch):
    match = _AsyncMatch(GOOD)

    def _blocking_remote(*args, **kwargs):
        raise AssertionError("the blocking .remote was called; use .remote.aio")

    _blocking_remote.aio = match
    method = types.SimpleNamespace(remote=_blocking_remote)
    monkeypatch.setattr(app_module, "MatchVerse", lambda: types.SimpleNamespace(match=method))
    return match


def _raw_verse_api(app_module):
    return inspect.unwrap(app_module.verse_api._raw_f_)


def _read(app_module, request, max_bytes):
    return asyncio.run(app_module.read_limited_body(request.headers, request.stream(), max_bytes))


def test_body_cap_constants(app_module):
    assert app_module.MAX_BODY_BYTES == 65536
    assert app_module.ERROR_BODY_TOO_LARGE == "The request body must be at most 65536 bytes."


def test_read_limited_body_rejects_a_large_content_length_without_reading(app_module):
    request = _FakeRequest(headers={"content-length": "70000"}, chunks=[b"x" * 70000])
    body, measure = _read(app_module, request, 65536)
    assert body is None
    assert measure == "content_length=70000"
    assert request.stream().read == 0


@pytest.mark.parametrize("headers", [{}, {"content-length": "10"}, {"content-length": "junk"}])
def test_read_limited_body_stops_at_the_first_chunk_over_the_cap(app_module, headers):
    chunks = [b"a" * 40000, b"b" * 40000, b"c" * 40000, b""]
    request = _FakeRequest(headers=headers, chunks=chunks)
    body, measure = _read(app_module, request, 65536)
    assert body is None
    assert measure == "bytes_read=80000"
    assert request.stream().read == 2


def test_read_limited_body_accepts_exactly_the_cap(app_module):
    chunks = [b"a" * 30000, b"b" * 35536, b""]
    request = _FakeRequest(headers={"content-length": "65536"}, chunks=chunks)
    body, _ = _read(app_module, request, 65536)
    assert body == b"a" * 30000 + b"b" * 35536


def _verse_api(app_module, request):
    return asyncio.run(_raw_verse_api(app_module)(request))


def test_verse_api_takes_a_raw_request(app_module):
    tree = ast.parse(APP_FILE.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "verse_api")
    assert [(a.arg, ast.unparse(a.annotation)) for a in func.args.args] == [("request", "Request")]
    assert inspect.iscoroutinefunction(_raw_verse_api(app_module))


def test_verse_api_413_on_large_content_length(app_module, stub_fastapi, gpu, caplog):
    request = _FakeRequest(headers={"content-length": "70000"}, chunks=[b"x" * 70000])
    response = _verse_api(app_module, request)
    assert response.status_code == 413
    assert set(response.content) == {"error", "id"}
    assert response.content["error"] == app_module.ERROR_BODY_TOO_LARGE
    assert ID_RE.fullmatch(response.content["id"])
    assert request.stream().read == 0
    assert gpu.calls == []
    assert (
        f"[{response.content['id']}] rejected: status=413 reason=body_too_large content_length=70000"
        in caplog.text
    )


def test_verse_api_413_on_a_streamed_body_over_the_cap(app_module, stub_fastapi, gpu, caplog):
    secret = b'{"text": "' + SENTINEL.encode() + b"x" * 70000 + b'"}'
    chunks = [secret[:65000], secret[65000:66000], secret[66000:], b""]
    request = _FakeRequest(headers={}, chunks=chunks)
    response = _verse_api(app_module, request)
    assert response.status_code == 413
    assert response.content["error"] == app_module.ERROR_BODY_TOO_LARGE
    assert request.stream().read == 2
    assert gpu.calls == []
    assert f"[{response.content['id']}] rejected: status=413 reason=body_too_large bytes_read=66000" in caplog.text
    assert SENTINEL not in caplog.text


def test_verse_api_accepts_a_body_of_exactly_the_cap(app_module, stub_fastapi, gpu):
    raw = b'{"text": "' + b"a" * 100 + b'"}' + b" " * (65536 - 112)
    assert len(raw) == 65536
    response = _verse_api(app_module, _FakeRequest(raw))
    assert response.status_code == 200
    assert response.content == GOOD_RAW
    assert gpu.calls == ["a" * 100]


def test_verse_api_valid_body_awaits_remote_aio(app_module, stub_fastapi, gpu):
    response = _verse_api(app_module, _FakeRequest(json.dumps({"text": "Some statement"}).encode()))
    assert response.status_code == 200
    assert response.content == GOOD_RAW
    assert gpu.calls == ["Some statement"]


def test_verse_api_invalid_json_is_fastapis_validation_error(app_module, stub_fastapi, gpu):
    with pytest.raises(_RequestValidationError) as info:
        _verse_api(app_module, _FakeRequest(b'{"text": "unterminated'))
    errors = info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "json_invalid"
    assert errors[0]["loc"][0] == "body"
    assert errors[0]["input"] == {}
    assert gpu.calls == []


@pytest.mark.parametrize(
    ("raw", "loc", "error_type"),
    [(b"{}", ("body", "text"), "missing"), (b'{"text": 5}', ("body", "text"), "string_type")],
)
def test_verse_api_invalid_body_is_fastapis_validation_error(
    app_module, stub_fastapi, gpu, raw, loc, error_type
):
    with pytest.raises(_RequestValidationError) as info:
        _verse_api(app_module, _FakeRequest(raw))
    errors = info.value.errors()
    assert [(tuple(e["loc"]), e["type"]) for e in errors] == [(loc, error_type)]
    assert "url" not in errors[0]
    assert gpu.calls == []


def test_verse_api_over_long_text_is_still_the_4000_character_422(app_module, stub_fastapi, gpu):
    response = _verse_api(app_module, _FakeRequest(json.dumps({"text": "a" * 4001}).encode()))
    assert response.status_code == 422
    assert response.content["error"] == app_module.ERROR_TEXT_TOO_LONG
    assert gpu.calls == []
