"""Static checks for issue #1: the Modal image must be fully pinned and the
model load must not rely on removed transformers arguments."""

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = ROOT / "wwjd_bot_modal.py"
REQ_IN = ROOT / "requirements.in"
REQ_TXT = ROOT / "requirements.txt"
REQ_DEV = ROOT / "requirements-dev.txt"

PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.+!_-]+$")


def _requirement_lines(path):
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Drop environment markers; the pin itself is what matters.
        line = line.split(";", 1)[0].strip()
        lines.append(line)
    return lines


def _pins(path):
    pins = {}
    for line in _requirement_lines(path):
        assert PIN_RE.match(line), f"{path.name}: not pinned: {line!r}"
        name, version = line.split("==", 1)
        name = re.sub(r"\[.*\]", "", name).lower().replace("_", "-")
        pins[name] = version
    return pins


def _app_tree():
    return ast.parse(APP_FILE.read_text())


def _calls(tree, attr):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def _source_files():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    rels = [r for r in out.split("\0") if r and not r.startswith("tests/")]
    assert "wwjd_bot_modal.py" in rels, "git ls-files did not list wwjd_bot_modal.py"
    return [ROOT / r for r in rels]


# Criterion 1


def test_requirements_txt_every_line_pinned():
    assert REQ_TXT.exists(), "requirements.txt (compiled lock) is missing"
    lines = _requirement_lines(REQ_TXT)
    assert lines, "requirements.txt has no requirements"
    for line in lines:
        assert not line.startswith("-"), f"option or editable line in lock: {line!r}"
        assert "://" not in line, f"URL requirement in lock: {line!r}"
        assert PIN_RE.match(line), f"not pinned: {line!r}"


def test_requirements_in_pins_core_packages():
    assert REQ_IN.exists(), "requirements.in is missing"
    pins = _pins(REQ_IN)
    for name in ("fastapi", "transformers", "torch", "accelerate", "bitsandbytes"):
        assert name in pins, f"{name} not pinned in requirements.in"
    assert pins["transformers"].startswith("4."), "transformers must stay on 4.x"
    lock = _pins(REQ_TXT)
    for name, version in pins.items():
        assert lock.get(name) == version, f"{name} lock {lock.get(name)} != {version}"


def test_image_pins_python_version():
    calls = _calls(_app_tree(), "debian_slim")
    assert calls, "no debian_slim() call"
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        value = kwargs.get("python_version")
        assert isinstance(value, ast.Constant) and value.value == "3.12", (
            "debian_slim must be called with python_version='3.12'"
        )


def test_image_installs_from_requirements_file():
    tree = _app_tree()
    for call in _calls(tree, "pip_install") + _calls(tree, "uv_pip_install"):
        literal = [a for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        assert not literal, "image installs loose package names instead of the lock"
    installs = _calls(tree, "uv_pip_install") + _calls(tree, "pip_install_from_requirements")
    assert installs, "image is not built from a requirements file"
    text = APP_FILE.read_text()
    assert '"requirements.txt"' in text or "'requirements.txt'" in text, (
        "requirements.txt is not referenced by the image"
    )


# Criterion 2


def test_no_use_auth_token():
    for path in _source_files():
        assert "use_auth_token" not in path.read_text(), f"use_auth_token in {path.name}"


def test_no_transformers_cache():
    for path in _source_files():
        assert "TRANSFORMERS_CACHE" not in path.read_text(), f"TRANSFORMERS_CACHE in {path.name}"


# Criterion 4


def test_dev_requirements_pin_modal():
    assert REQ_DEV.exists(), "requirements-dev.txt is missing"
    pins = _pins(REQ_DEV)
    assert pins.get("modal") == "1.6.3"
    assert "pytest" in pins
    assert pins.get("pydantic") == _pins(REQ_TXT).get("pydantic"), (
        "pydantic in requirements-dev.txt must match the image lock"
    )


# Criterion 5


def test_hf_home_set_on_image():
    tree = _app_tree()
    model_dir = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODEL_DIR" for t in node.targets
        ):
            model_dir = node.value
    assert isinstance(model_dir, ast.Constant)
    assert model_dir.value == "/root/.cache/huggingface"
    found = False
    for call in _calls(tree, "env"):
        for arg in call.args:
            if not isinstance(arg, ast.Dict):
                continue
            for key, value in zip(arg.keys, arg.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "HF_HOME"
                    and isinstance(value, ast.Name)
                    and value.id == "MODEL_DIR"
                ):
                    found = True
    assert found, 'image does not call .env({"HF_HOME": MODEL_DIR})'


# Defaults unchanged, test deploys isolated


def _import_app(monkeypatch, env):
    for key in ("WWJD_APP_NAME", "WWJD_HF_CACHE_VOLUME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        return importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def test_default_app_and_volume_names(monkeypatch):
    module = _import_app(monkeypatch, {})
    assert module.app.name == "wwjd-bot"
    assert module.model_volume.name == "huggingface-cache"


@pytest.mark.parametrize(
    "app_name,volume_name", [("wwjd-bot-test", "wwjd-hf-cache-test")]
)
def test_app_and_volume_names_overridable(monkeypatch, app_name, volume_name):
    module = _import_app(
        monkeypatch, {"WWJD_APP_NAME": app_name, "WWJD_HF_CACHE_VOLUME": volume_name}
    )
    assert module.app.name == app_name
    assert module.model_volume.name == volume_name
