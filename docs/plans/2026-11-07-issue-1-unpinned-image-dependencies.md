# Plan: Unpinned image dependencies: any rebuild installs Transformers 5 and model load crashes on use_auth_token (issue #1)

Risk tier: High. Not a forced-High category, but raised twice: the real behavior (a Modal image build plus a gated model load on a T4) cannot be exercised by a unit test, and the change alters existing runtime behavior (dependency versions, auth, cache path) rather than adding to it. Approved by the orchestrator on the user's behalf.

Domain skills:
- superpowers:test-driven-development: invoke.
- context7-plugin:docs (or context7 MCP): invoke, to confirm the Modal 1.6 `Image.uv_pip_install` / `pip_install_from_requirements` and `Image.env` signatures and the transformers 4.57 `token=` kwarg.
- python-fastapi: not applicable because the FastAPI endpoint is not changed.
- django, django-ninja-api, expo, react skills: not applicable because the repo is a single Modal Python app.

## Acceptance criteria

1. Every package in the Modal image is pinned (`name==version`, including transitive dependencies, through a compiled requirements file), and so is the Python version (`debian_slim(python_version="3.12")`).
2. `use_auth_token` and `TRANSFORMERS_CACHE` no longer appear in the code.
3. A fresh image build (`modal deploy` with the image cache cleared, or a forced rebuild) loads the model and serves a request successfully. Driven only against a separate test app (`wwjd-bot-test`), never production. If no Modal credentials exist on this machine, this criterion is reported as blocked and handed to the user.
4. The Modal client that deploys the app is pinned in a local dev requirements file.
5. The Hugging Face cache location is set on the image through `HF_HOME` before `transformers` is imported, and resolves to the same path as today (`/root/.cache/huggingface`), so the existing `huggingface-cache` Volume contents keep being found.

## Files that change

- `requirements.in` (new): the top-level image dependencies, each pinned: `fastapi[standard]==0.129.2`, `transformers==4.57.6`, `torch==2.9.1`, `accelerate==1.12.0`, `bitsandbytes==0.48.2`. If the resolver rejects a combination, pick the nearest compatible release in the same series and record it here. transformers stays on 4.x because that is the line the deployed app was built against; the code change below also works on 5.x, but upgrading majors is not this issue.
- `requirements.txt` (new): fully pinned lock compiled from `requirements.in` with `uv pip compile requirements.in -o requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28` (keep the header uv writes so the command is reproducible).
- `requirements-dev.txt` (new): local tooling pins: `modal==1.6.3`, `pytest==9.2.0`, `pydantic` pinned to the version in `requirements.txt` (the module imports pydantic at top level when deploying).
- `wwjd_bot_modal.py`:
  - `image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(requirements=[<abs path to requirements.txt>]).env({"HF_HOME": MODEL_DIR})` (or `pip_install_from_requirements` if `uv_pip_install` does not take `requirements=` in modal 1.6.3). The requirements path is built from `Path(__file__).parent` so deploys work from any cwd. Move `MODEL_DIR` above the image definition.
  - App name from env: `modal.App(os.environ.get("WWJD_APP_NAME", "wwjd-bot"))`, and Volume name from env: `os.environ.get("WWJD_HF_CACHE_VOLUME", "huggingface-cache")`, so a test deploy can use its own app and an empty cache Volume (forced fresh download) without touching production. Defaults keep production identical.
  - In `match_verse`: delete `os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR`, delete `from huggingface_hub import login` and `login(token)`, and pass `token=token` instead of `use_auth_token=token` to both `from_pretrained` calls. The token is still read from `HUGGINGFACE_TOKEN` (the key the existing `huggingface-token` Secret provides).
  - Nothing else in the file changes (prompt, parsing, endpoint, GPU settings are other issues).
- `tests/test_image_pins.py` (new): static tests (see Proof).
- `CLAUDE.md`: already added (git exception). No further change unless Phase 8 finds one.

## Order of work

1. Write `tests/test_image_pins.py` with the defect tests below; run them against the current code; confirm they fail for the expected reasons; commit the tests alone (red). Note their paths in `.claude/fix-lock` (untracked, removed once green; do not commit it).
2. Add `requirements.in`, compile `requirements.txt`, add `requirements-dev.txt`.
3. Edit `wwjd_bot_modal.py` as above.
4. Gate (below), green.
5. Local drive (below). Then the Modal live drive if credentials exist.

## Risks

- Riskiest step: the version set. torch, bitsandbytes and the T4 (sm_75) must agree on a CUDA build; torch 2.9.1 default wheels (CUDA 12.8) still ship sm_75 kernels and bitsandbytes 0.48.x supports them. Only a real Modal build proves it; that is criterion 3.
- `HF_HOME` vs the old effective path: the default HF_HOME is already `/root/.cache/huggingface`, so setting it explicitly to `MODEL_DIR` changes nothing on disk. Setting `HF_HUB_CACHE` instead would move the hub cache and force a re-download; do not do that.
- Dropping `login()`: it also stopped writing the token file into `HF_HOME/token`, which sits on the shared Volume. Passing `token=` explicitly is enough for gated downloads.
- Env-var app and volume names: the defaults must equal today's names exactly, or a deploy would create a new app/volume.
- Not doing: upgrading to transformers 5, model reload per request (#2), logging (#5), prompt (#4), parsing (#3), README/CI (#6).

## Gate (derived; the repo has no gate yet)

From the worktree root, in a venv at `.venv` (gitignored) created with `uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements-dev.txt`:
1. `.venv/bin/python -m py_compile wwjd_bot_modal.py`
2. `.venv/bin/python -c "import wwjd_bot_modal"` (module imports with the pinned modal client and no credentials)
3. `.venv/bin/python -m pytest -q`

## Proof

- Criterion 1: `test_requirements_txt_every_line_pinned` (every requirement line is `name[extras]==version`, no URLs or `-e`), `test_requirements_in_pins_core_packages` (the five packages are pinned and transformers is 4.x), `test_image_pins_python_version` (AST: `debian_slim` called with `python_version="3.12"`), `test_image_installs_from_requirements_file` (AST: image built from `requirements.txt`, no string-literal `pip_install(...)` packages).
- Criterion 2: `test_no_use_auth_token` and `test_no_transformers_cache` (source text of every `*.py` outside `tests/` and `.venv/`).
- Criterion 4: `test_dev_requirements_pin_modal`.
- Criterion 5: `test_hf_home_set_on_image` (AST: `.env({"HF_HOME": MODEL_DIR})` on the image) and MODEL_DIR is `/root/.cache/huggingface`.
- Defaults unchanged: `test_default_app_and_volume_names` (import the module with `WWJD_APP_NAME` and `WWJD_HF_CACHE_VOLUME` unset; `app.name == "wwjd-bot"` and `model_volume.name == "huggingface-cache"`; this one is a regression guard and passes before the fix by design). `test_app_and_volume_names_overridable` (with both env vars set, the app and volume take the test names; added during implementation so the env override itself is covered).
- Local drive (no GPU, no Modal): in a scratch venv with the pinned `transformers`, `torch` (CPU wheel of the same version) and `accelerate`, run the same `AutoTokenizer.from_pretrained(m, token=None)` / `AutoModelForCausalLM.from_pretrained(m, token=None, device_map="auto")` calls on `hf-internal-testing/tiny-random-MistralForCausalLM` and generate a few tokens; also show `use_auth_token=` on transformers 5.22.2 raises `TypeError` (the original defect) while `token=` works there too.
- Criterion 3 (live): `WWJD_APP_NAME=wwjd-bot-test WWJD_HF_CACHE_VOLUME=wwjd-hf-cache-test MODAL_FORCE_BUILD=1 modal deploy wwjd_bot_modal.py`, POST `{"text": "..."}` to the test endpoint, expect a 200 with `raw.reference` / `raw.verse` / `raw.explanation`, then `modal app stop wwjd-bot-test` and delete the test volume. Requires Modal credentials and the `huggingface-token` Secret in the same workspace.
