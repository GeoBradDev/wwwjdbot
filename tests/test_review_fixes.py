"""Review findings on issue #1: pin the uv binary used by the image build, and
fail loudly when the Hugging Face token Secret is missing."""

import ast
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = ROOT / "wwjd_bot_modal.py"


def test_uv_pip_install_pins_uv_version():
    tree = ast.parse(APP_FILE.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "uv_pip_install"
    ]
    assert calls, "no uv_pip_install() call"
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        value = kwargs.get("uv_version")
        assert isinstance(value, ast.Constant), (
            "uv_pip_install without uv_version pulls ghcr.io/astral-sh/uv:latest"
        )
        assert re.fullmatch(r"\d+\.\d+\.\d+", str(value.value)), (
            f"uv_version must be an exact release, got {value.value!r}"
        )


class _Stop(Exception):
    pass


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.mark.filterwarnings("ignore:The MatchVerse.match function is executing locally")
def test_missing_hf_token_fails_loudly(monkeypatch):
    calls = []

    def _from_pretrained(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop("model load reached")

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
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        module = importlib.import_module("wwjd_bot_modal")
        with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN"):
            module.MatchVerse().match.local("some statement")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)
    assert calls == [], "model load was attempted without a token"
