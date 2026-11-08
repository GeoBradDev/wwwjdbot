"""Issue #12: GPU cost is bounded by a container cap, a prompt token cap and a
request body byte cap.

No GPU, torch, transformers, FastAPI or Modal credentials are needed: the
tokenizer, model, fastapi modules and the request are fakes."""

import ast
import importlib
import logging
import re
import sys
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
