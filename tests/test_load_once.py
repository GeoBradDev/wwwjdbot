"""Issue #2: the model loads once per GPU container, the HTTP endpoint runs on
a CPU-only image and awaits the GPU call without blocking, and over-long text
is rejected before any GPU work.

transformers and torch are replaced by fakes, so no GPU, model download or
Modal credentials are needed."""

import ast
import asyncio
import contextlib
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
ID_RE = re.compile(r"[0-9a-f]{32}")
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
GENERATED = '{"reference": "Matthew 5:44", "verse": "Love your enemies", "explanation": "Hate is not the way."}'
SENTINEL = "SENTINEL-LONG-TEXT"


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


# Fake transformers and torch


class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=None)
        self.input_ids = types.SimpleNamespace(shape=(1, 3))

    def to(self, device):
        return self


class _Tokenizer:
    eos_token = "</s>"
    eos_token_id = 2

    def __call__(self, prompt, return_tensors=None):
        return _Inputs()

    def decode(self, tokens, skip_special_tokens=False):
        return GENERATED


class _Model:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[0, 1, 2, 3, 4]]


class _Loader:
    def __init__(self, factory):
        self.factory = factory
        self.calls = 0

    def from_pretrained(self, *args, **kwargs):
        self.calls += 1
        return self.factory()


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_ml(monkeypatch):
    tokenizer_loader = _Loader(_Tokenizer)
    model_loader = _Loader(_Model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _stub_module(
            "transformers",
            AutoTokenizer=tokenizer_loader,
            AutoModelForCausalLM=model_loader,
            BitsAndBytesConfig=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _stub_module("torch", float16="float16", no_grad=contextlib.nullcontext),
    )
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_fake")
    return types.SimpleNamespace(tokenizer=tokenizer_loader, model=model_loader)


def _load_lines(caplog):
    return [r.getMessage() for r in caplog.records if "model loaded" in r.getMessage()]


# Criterion 1: load once per container


def test_match_local_loads_the_model_once_per_instance(app_module, fake_ml, caplog):
    instance = app_module.MatchVerse()
    first = instance.match.local("Some statement")
    second = instance.match.local("Another statement")
    third = instance.match.local("A third statement")
    for result in (first, second, third):
        assert {k: result[k] for k in ("reference", "verse", "explanation")} == GOOD_RAW["raw"]
    assert fake_ml.tokenizer.calls == 1
    assert fake_ml.model.calls == 1
    lines = _load_lines(caplog)
    assert len(lines) == 1
    assert re.fullmatch(r"model loaded: load_time=\d+\.\d\ds", lines[0])


def test_match_never_loads_after_the_enter_hook(app_module, fake_ml, caplog):
    instance = app_module.MatchVerse._get_user_cls()()
    instance.load()
    assert (fake_ml.tokenizer.calls, fake_ml.model.calls) == (1, 1)
    instance.match("Some statement")
    instance.match("Another statement", request_id="0123456789abcdef0123456789abcdef")
    assert (fake_ml.tokenizer.calls, fake_ml.model.calls) == (1, 1)
    assert len(_load_lines(caplog)) == 1


def _class_def(name):
    tree = ast.parse(APP_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no class {name}")


def _function_def(name):
    tree = ast.parse(APP_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


def _decorator_names(node):
    names = []
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        names.append(ast.unparse(target))
    return names


def _decorator_call(node, name):
    for deco in node.decorator_list:
        if isinstance(deco, ast.Call) and ast.unparse(deco.func) == name:
            return deco
    raise AssertionError(f"{node.name} has no @{name}(...)")


def test_match_verse_is_a_gpu_class_with_enter_and_method(app_module):
    cls = _class_def("MatchVerse")
    call = _decorator_call(cls, "app.cls")
    kwargs = {k.arg: k.value for k in call.keywords}
    assert ast.unparse(kwargs["image"]) == "image"
    assert ast.literal_eval(kwargs["gpu"]) == "T4"
    assert ast.literal_eval(kwargs["scaledown_window"]) == 300
    assert ast.literal_eval(kwargs["timeout"]) == 600
    assert ast.unparse(kwargs["volumes"]) == "{MODEL_DIR: model_volume}"
    assert "huggingface-token" in ast.unparse(kwargs["secrets"])
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert _decorator_names(methods["load"]) == ["modal.enter"]
    assert _decorator_names(methods["match"]) == ["modal.method"]
    assert set(app_module.MatchVerse._get_partial_functions()) == {"load", "match"}
    assert app_module.MatchVerse._get_class_service_function()._spec_.gpus == "T4"


def test_match_verse_function_is_gone(app_module):
    assert not hasattr(app_module, "match_verse")


# Criterion 2: the endpoint holds no GPU and runs on the API image


def test_verse_api_requests_no_gpu_and_uses_the_api_image(app_module):
    assert app_module.verse_api._spec_.gpus is None
    call = _decorator_call(_function_def("verse_api"), "app.function")
    kwargs = {k.arg: k.value for k in call.keywords}
    assert "gpu" not in kwargs
    assert ast.unparse(kwargs["image"]) == "api_image"


def test_api_image_installs_the_api_lock(app_module):
    assert app_module.API_REQUIREMENTS == ROOT / "requirements-api.txt"
    tree = ast.parse(APP_FILE.read_text())
    api_image = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "api_image" for t in node.targets
        ):
            api_image = ast.unparse(node.value)
            break
    assert api_image is not None, "no module-level api_image assignment"
    assert "debian_slim(python_version='3.12')" in api_image
    assert "uv_pip_install(requirements=[str(API_REQUIREMENTS)]" in api_image
    assert "[str(REQUIREMENTS)]" not in api_image


def test_full_text_flag_reaches_both_images():
    tree = ast.parse(APP_FILE.read_text())
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_full_text_image_env()"
    ]
    assert blocks, "no `if _full_text_image_env():` block"
    assigned = set()
    for block in blocks:
        for stmt in block.body:
            if isinstance(stmt, ast.Assign) and ast.unparse(stmt.value).endswith(
                ".env(_full_text_image_env())"
            ):
                assigned.update(ast.unparse(t) for t in stmt.targets)
    assert {"image", "api_image"} <= assigned


# Criterion 4: the endpoint awaits the GPU call without blocking


def test_handle_verse_request_is_async(app_module):
    assert inspect.iscoroutinefunction(app_module.handle_verse_request)


class _JSONResponse:
    def __init__(self, status_code=200, content=None):
        self.status_code = status_code
        self.content = content


def _stub_fastapi(monkeypatch):
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _JSONResponse
    fastapi = types.ModuleType("fastapi")
    fastapi.responses = responses
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)


class _Remote:
    """Stands in for MatchVerse().match.remote: calling it blocks, so it fails."""

    def __init__(self, aio):
        self.aio = aio

    def __call__(self, *args, **kwargs):
        raise AssertionError("the blocking .remote was called; use .remote.aio")


def _patch_match_verse(app_module, monkeypatch, aio):
    remote = _Remote(aio)
    method = types.SimpleNamespace(remote=remote)
    monkeypatch.setattr(
        app_module, "MatchVerse", lambda: types.SimpleNamespace(match=method)
    )


def test_two_concurrent_requests_both_reach_the_gpu_call(app_module, monkeypatch):
    _stub_fastapi(monkeypatch)
    entered = []

    async def scenario():
        both_in = asyncio.Event()

        async def aio(text, request_id=None):
            entered.append(text)
            if len(entered) == 2:
                both_in.set()
            await asyncio.wait_for(both_in.wait(), timeout=5)
            return GOOD

        _patch_match_verse(app_module, monkeypatch, aio)
        raw = inspect.unwrap(app_module.verse_api._raw_f_)
        return await asyncio.gather(
            raw(app_module.VerseRequest(text="first")),
            raw(app_module.VerseRequest(text="second")),
        )

    responses = asyncio.run(scenario())
    assert sorted(entered) == ["first", "second"]
    assert [r.status_code for r in responses] == [200, 200]
    assert [r.content for r in responses] == [GOOD_RAW, GOOD_RAW]


# Criterion 5: maximum text length


class _AsyncMatch:
    def __init__(self, result=GOOD):
        self.result = result
        self.calls = []

    async def __call__(self, text, request_id=None):
        self.calls.append(text)
        return self.result


def test_max_text_length_constants(app_module):
    assert app_module.MAX_TEXT_LENGTH == 4000
    assert app_module.ERROR_TEXT_TOO_LONG == "The 'text' field must be at most 4000 characters."


def test_text_of_exactly_max_length_is_accepted(app_module):
    match = _AsyncMatch()
    text = "a" * 4000
    status, body = asyncio.run(app_module.handle_verse_request(text, match))
    assert status == 200
    assert body == GOOD_RAW
    assert match.calls == [text]


def test_text_over_max_length_is_422_without_calling_match(app_module, caplog):
    match = _AsyncMatch()
    text = SENTINEL + "a" * (4001 - len(SENTINEL))
    assert len(text) == 4001
    status, body = asyncio.run(app_module.handle_verse_request(text, match))
    assert status == 422
    assert set(body) == {"error", "id"}
    assert body["error"] == app_module.ERROR_TEXT_TOO_LONG
    assert ID_RE.fullmatch(body["id"])
    assert match.calls == []
    assert SENTINEL not in repr(body)
    assert SENTINEL not in caplog.text
    assert (
        f"[{body['id']}] rejected: status=422 reason=text_too_long input_length=4001"
        in caplog.text
    )


def test_whitespace_only_text_over_max_length_is_the_empty_text_422(app_module, caplog):
    match = _AsyncMatch()
    status, body = asyncio.run(app_module.handle_verse_request(" \n" * 3000, match))
    assert status == 422
    assert body["error"] == app_module.ERROR_EMPTY_TEXT
    assert match.calls == []
    assert "reason=empty_text" in caplog.text
    assert "text_too_long" not in caplog.text


def test_verse_api_rejects_over_long_text_before_the_gpu_call(app_module, monkeypatch):
    _stub_fastapi(monkeypatch)
    match = _AsyncMatch()
    _patch_match_verse(app_module, monkeypatch, match)
    raw = inspect.unwrap(app_module.verse_api._raw_f_)
    response = asyncio.run(raw(app_module.VerseRequest(text="a" * 4001)))
    assert response.status_code == 422
    assert response.content["error"] == app_module.ERROR_TEXT_TOO_LONG
    assert match.calls == []
