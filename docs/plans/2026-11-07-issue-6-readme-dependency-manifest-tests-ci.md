# Plan: No README, dependency manifest, tests, or CI (issue #6)

Risk tier: Medium. Not a forced-High category. Raised once for restructuring existing runtime code (prompt building and output parsing move out of the GPU function), and once more for touching more than a few files, but lowered back because the moved logic is pure and fully covered by unit tests that pin current behavior byte for byte. Plan approved by the orchestrator on the user's behalf (the user asked for autonomous work).

Domain skills:
- superpowers:test-driven-development: invoke (characterization tests first, then extraction).
- context7-plugin:docs: invoke, to confirm the `modal serve` / `modal deploy` CLI usage cited in the README.
- python-fastapi: not applicable because the FastAPI endpoint and its contract are not changed.
- django, django-ninja-api, expo, react skills: not applicable because the repo is a single Modal Python app.

## Acceptance criteria

1. `README.md` lets a new contributor deploy following only the README. It covers: purpose; architecture (Chrome extension `GeoBradDev/WWJC_ChromeExtension`, then the Modal web endpoint `verse_api`, then the GPU function `match_verse`); prerequisites (Modal account and CLI, Hugging Face gated access to `mistralai/Mistral-7B-Instruct-v0.2`, Modal Secret `huggingface-token` holding env var `HUGGINGFACE_TOKEN`, Volume `huggingface-cache`); the `WWJD_APP_NAME` and `WWJD_HF_CACHE_VOLUME` overrides for a test app; `modal serve` and `modal deploy`; the current API contract as the code implements it today; how to run lint and tests locally.
2. Prompt building and output parsing are module-level pure functions in `wwjd_bot_modal.py` (`build_prompt(text)` and `parse_model_output(generated_text, model_name, elapsed_time)`), called by `match_verse`. Behavior is identical: the prompt string is byte-for-byte the same, and every parse path returns the same dict as before, including the current `{"error": "Parsing failed: ...", "raw_output": ...}` path when the matched JSON is invalid (issue #3 changes that later; not here).
3. pytest unit tests for both functions run with no GPU, no torch, no transformers and no Modal credentials.
4. `.github/workflows/ci.yml` runs on push and pull_request: `ruff check .`, `pytest`, and an import check of the Modal app (`python -c "import wwjd_bot_modal"`). Actions are pinned to full commit SHAs, Python to 3.12, and tool versions through `requirements-dev.txt`. The workflow passes on the PR.
5. `ruff check .` passes, with the existing findings fixed only by safe mechanical changes; the two blind `except Exception` handlers keep their behavior and carry a targeted `# noqa: BLE001` with a reason.
6. `.gitignore` ignores `.idea/` and `.claude/worktrees/`.

## Files that change

- `wwjd_bot_modal.py`:
  - Sort the top-level imports (ruff I001): stdlib `json`, `os`, `re`, `time`, `pathlib.Path`, then `modal`, `pydantic.BaseModel`.
  - Add `MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"` at module level; `match_verse` sets `model_name = MODEL_NAME`.
  - Add `build_prompt(text: str) -> str` returning exactly the current f-string. Copy the literal lines verbatim, keeping the same 4-space continuation indentation and the existing em dash and en dash characters inside the prompt (changing them would change model input, so they stay; this is the only place they exist).
  - Add `parse_model_output(generated_text: str, model_name: str, elapsed_time: float) -> dict` containing the current regex/json/try/except body verbatim, with `except Exception as e:  # noqa: BLE001 <reason>`. It must stay a blind catch: narrowing it could change what is caught.
  - `match_verse` keeps the local `transformers` and `torch` imports (sorted), drops the now-unused local `os`, `re`, `json` imports, and calls `build_prompt(text)` and `return parse_model_output(generated_text, model_name, elapsed_time)`. The prints, timing and generation call stay as they are.
  - `verse_api`: only add `# noqa: BLE001 <reason>` to its `except Exception`. No other change.
  - The helpers live in this file, not a new module, because Modal 1.x only ships the entrypoint file to the container automatically; a separate module would need `add_local_python_source` and an unverifiable deploy.
- `tests/test_prompt_and_parsing.py` (new): characterization tests (see Proof). Non-ASCII prompt characters are written as `\u2014` / `\u2013` escapes so the test file has no em dash.
- `tests/test_image_pins.py`: `zip(arg.keys, arg.values, strict=True)` (ruff B905; `ast.Dict` keys and values are always the same length, so behavior is unchanged).
- `pyproject.toml` (new): tool config only, no packaging. `[tool.ruff]` `target-version = "py312"`, `extend-exclude = [".venv", ".claude"]`; `[tool.ruff.lint]` `select = ["E", "F", "W", "I", "B", "BLE", "UP"]`, `ignore = ["E501"]` with a comment that the prompt is one long string literal whose lines cannot be wrapped without changing model input. `[tool.pytest.ini_options]` `testpaths = ["tests"]`.
- `requirements-dev.txt`: add `ruff==0.17.4`.
- `.github/workflows/ci.yml` (new): name `CI`; `on: push` and `pull_request`; `permissions: contents: read`; one job on `ubuntu-24.04`: `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1`, `actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0  # v7.1.0` with `python-version: "3.12"` and `cache: pip` keyed on `requirements-dev.txt`; `pip install -r requirements-dev.txt`; `ruff check .`; `pytest`; `python -c "import wwjd_bot_modal; print(wwjd_bot_modal.app.name)"`. No secrets, no Modal token, no deploy.
- `.gitignore`: append `.idea/` and `.claude/worktrees/`.
- `README.md` (new): criterion 1. API contract as implemented today: `POST` JSON `{"text": "..."}`; success `200` with `{"raw": {"reference", "verse", "explanation"}}` (normally strings, possibly empty; on the JSON path they are whatever JSON values the model emitted, with no type coercion today); empty `text` returns `200` with `{"error": "Missing 'text' field in request"}`; a missing `text` key is rejected by FastAPI validation with `422`; failures inside processing return `200` with `{"error": "Processing failed: ..."}`. Note that issue #3 will revise error handling. Production URL `https://<workspace>--wwjd-bot-verse-api.modal.run` pattern, with the URL the extension uses. Deploy steps: install dev requirements, `modal setup`, accept the model license on Hugging Face, create a read token, `modal secret create huggingface-token HUGGINGFACE_TOKEN=<token>` (placeholder only), the Volume is created on first deploy (`create_if_missing=True`) or with `modal volume create huggingface-cache`, `modal serve wwjd_bot_modal.py` for a dev endpoint, `modal deploy wwjd_bot_modal.py`, a test deploy with `WWJD_APP_NAME=wwjd-bot-test WWJD_HF_CACHE_VOLUME=wwjd-hf-cache-test modal deploy wwjd_bot_modal.py`, a `curl` example. Updating the lock with the `uv pip compile` command from `requirements.in`, noting that `uv` must be installed for that step only. Local checks: `ruff check .`, `pytest`.
- `docs/plans/2026-11-07-issue-6-readme-dependency-manifest-tests-ci.md`: this plan.

## Order of work

1. Write `tests/test_prompt_and_parsing.py` against the not-yet-existing functions; run; confirm it fails because `build_prompt` / `parse_model_output` do not exist. Commit the test alone (red) and list it in `.claude/fix-lock` (untracked).
2. Extract the two functions and wire `match_verse` to them; sort imports; add the noqa comments. Tests green.
3. Add `pyproject.toml`, the ruff pin, the `zip(strict=True)` fix; `ruff check .` clean.
4. Add `.gitignore` entries and the CI workflow.
5. Write `README.md`.
6. Full gate.

## Risks

- Prompt drift: one changed space or character changes model input silently. Mitigated by the golden test built from the original source (see Proof).
- Parse drift: mitigated by tests over every branch.
- Modal container import: the helpers are module-level in the entrypoint file, which Modal imports in the container, so they resolve remotely. Top-level imports stay stdlib plus `modal` and `pydantic`, both in the image lock.
- CI import check: `modal.Secret.from_name` and `modal.Volume.from_name` are lazy, so importing needs no credentials (the existing tests already import the module).
- Riskiest step: the prompt extraction. Not done: fixing #3 (fallback reachability, status codes), #5 handlers, pyright in CI, any deploy.

## Proof

- Criterion 2 (prompt): `test_build_prompt_matches_original`. Reading the original from `git show 6b16340:wwjd_bot_modal.py` would not work in CI's shallow checkout, so the test holds the expected prompt as an explicit string literal reconstructed from the original source (with escapes for the dash characters) and asserts `build_prompt("X") == expected`; plus a check that the user text is inserted verbatim (including quotes and braces) exactly once. The implementer verifies the literal against the original once locally by evaluating the original f-string from `git show 6b16340:wwjd_bot_modal.py` and records the command output.
- Criterion 2 (parse), one test each: valid JSON with all keys (extra keys dropped, `model` and `generation_time` formatted `"%.2f seconds"`); JSON embedded in prose; parseable JSON missing a key falls back to the regex extraction; invalid JSON inside braces returns the `Parsing failed` error dict with `raw_output` (pins current #3 behavior); no braces and no fields returns empty reference/verse with the whole text as explanation; greedy brace match across two objects.
- Criterion 3: the test module imports only `wwjd_bot_modal` (no torch/transformers stubs needed for these functions).
- Criteria 4 and 5: `ruff check .`, `pytest`, the import check, locally and in the PR's CI run.
- Criterion 6: `git check-ignore .idea/ .claude/worktrees/x`.
- Criterion 1: read-through of the README by the verifier against the code. Live drive: no deploy (no credentials, out of scope); the import check plus `modal` CLI help confirm commands named.
