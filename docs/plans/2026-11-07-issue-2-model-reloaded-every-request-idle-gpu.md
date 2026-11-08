# Plan: Model is reloaded on every request and the HTTP endpoint holds an idle second GPU (issue #2)

Risk tier: High. Not a forced-High area, but every raise applies: the Modal container lifecycle cannot be fully exercised by a unit test (no GPU or Modal credentials here), the API contract gains a new `422`, existing behavior changes (load path, images, call shape), the endpoint becomes truly async, and more than a few files change. Plan approved by the orchestrator on the user's behalf.

Domain skills:
- superpowers:test-driven-development: invoke.
- python-fastapi: invoke, for the async endpoint and the request validation.
- context7-plugin:docs: invoke, to confirm the Modal 1.6.3 API for `@app.cls`, `@modal.enter()`, `@modal.method()`, `scaledown_window` and `.remote.aio`. A local probe with modal 1.6.3 (no credentials) already showed: `@app.cls(..., gpu="T4", scaledown_window=300)` builds; `Cls._get_user_cls()` returns the plain class; `MatchVerse().match.local(...)` runs the enter hook once per instance and then the method; `.remote` has an `.aio` attribute; `verse_api._spec_.gpus` is `None` without a `gpu=` argument and the class service function's `_spec_.gpus` is `"T4"`.
- django, django-ninja-api, expo, react, frontend, payments, email skills: not applicable because the repo is a single Modal Python app.

## Acceptance criteria

1. The model is loaded once per container, not once per request. `match_verse` becomes a class `MatchVerse` declared with `@app.cls(image=image, gpu="T4", volumes={MODEL_DIR: model_volume}, timeout=600, scaledown_window=300, secrets=[huggingface-token])`. A `@modal.enter()` method checks the token, loads the tokenizer and the 4-bit model once, and logs one metadata line (`model loaded: load_time=<s>s`). A `@modal.method()` `match(self, text, request_id=None)` only builds the prompt, generates and parses. Proven by unit tests with fakes: across two or more `match` calls on one instance, `from_pretrained` runs once per loader and the load line is logged once; the method never calls `from_pretrained`. The live check (a second warm request shows no model-load log lines) is deferred: no Modal credentials (see Proof).
2. The HTTP endpoint function `verse_api` requests no GPU, and its image is a lightweight image built from a second pinned lock (`requirements-api.in` compiled to `requirements-api.txt`) that holds FastAPI and pydantic and no torch, transformers, accelerate or bitsandbytes. Versions in the API lock equal the same packages in `requirements.txt`.
3. Warm-request latency: deferred, no Modal credentials on this machine. The PR records "deferred: no Modal credentials" with the exact test-app commands to measure it.
4. The endpoint awaits the GPU call asynchronously and does not block the event loop: `handle_verse_request` is `async` and awaits `match(...)`, and `verse_api` passes `MatchVerse().match.remote.aio`. Proven by a test where two concurrent `verse_api` calls both reach the fake `aio` call before either returns, and the blocking `.remote` is never called.
5. Requests whose `text` is longer than `MAX_TEXT_LENGTH = 4000` characters (`len(text)`) are rejected with `422` before any GPU work. The body follows the #3 contract: exactly `{"error": ERROR_TEXT_TOO_LONG, "id": <32 hex>}` with `ERROR_TEXT_TOO_LONG = "The 'text' field must be at most 4000 characters."` (the message is built from the constant). The text is never echoed back and not logged by default; the log line is `rejected: status=422 reason=text_too_long input_length=<n>`. Exactly 4000 characters is accepted. The empty-text check runs first, so a whitespace-only text of any length stays the empty-text `422`.
6. All #3 and #5 behavior is kept: the `200`, `422`, `502` and `500` bodies; metadata-only logging; `WWJD_LOG_FULL_TEXT`; the request id passed as `request_id` and normalized in the GPU method; `_run_guarded` and `MatchVerseError` around generation, so exception text from generation never leaves the GPU container. Existing tests keep every assertion; only call shapes change (see Files that change).
7. The `WWJD_APP_NAME` and `WWJD_HF_CACHE_VOLUME` overrides keep working, and `WWJD_LOG_FULL_TEXT=1` is copied into both images' env.
8. `README.md` describes the new architecture (CPU endpoint, GPU class, load once per container, `scaledown_window`), the deploy notes (cold start versus warm, the API lock and how to compile it), and the API contract row for the new `422`.
9. The CI gate (`ruff check .`, `pytest`, the import check) passes.

## Design

All app code stays in `wwjd_bot_modal.py` (one file by design; see README).

### Images

```python
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
API_REQUIREMENTS = Path(__file__).parent / "requirements-api.txt"

image = (... unchanged GPU image from requirements.txt, HF_HOME ...)
api_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    requirements=[str(API_REQUIREMENTS)], uv_version="0.10.3"
)
if _full_text_image_env():
    image = image.env(_full_text_image_env())
    api_image = api_image.env(_full_text_image_env())
```

`requirements-api.in`:

```
# Top-level dependencies of the verse_api (CPU) image. Compile the lock with:
#   uv pip compile requirements-api.in -o requirements-api.txt -c requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
fastapi[standard]==0.129.2
pydantic==2.14.3
```

`-c requirements.txt` constrains the API lock to the versions in the GPU lock, so both containers run the same FastAPI and pydantic. Compile with `uv` (available on this machine). Do not edit either lock by hand.

### GPU class

Split `_generate_verse` into:

- `_load_model(token) -> (tokenizer, model)`: the current tokenizer and `from_pretrained` calls with `BitsAndBytesConfig`, `tokenizer.pad_token = tokenizer.eos_token`, `model.eval()`. Imports torch and transformers inside the function, as now.
- `_generate_verse(tokenizer, model, text, request_id) -> dict`: build the prompt, tokenize, generate with the same arguments, decode, then `_log_generation(...)` as now. Imports torch inside for `torch.no_grad()`.

```python
@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    gpu="T4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
class MatchVerse:
    @modal.enter()
    def load(self):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_TOKEN is not set; attach the huggingface-token Modal Secret")
        start = time.time()
        self.tokenizer, self.model = _load_model(token)
        logger.info("model loaded: load_time=%.2fs", time.time() - start)

    @modal.method()
    def match(self, text: str, request_id: str | None = None):
        request_id = _normalize_request_id(request_id)
        return _run_guarded(request_id, _generate_verse, self.tokenizer, self.model, text, request_id)
```

The enter hook runs before any request text exists, so a load failure cannot carry user text; it is not wrapped in `_run_guarded`. Modal fails the call, and `handle_verse_request` turns any exception into the generic `500`. The module-level `match_verse` function is removed; all references become `MatchVerse`.

`scaledown_window=300`: Modal's default is 60 seconds. A browser-extension user tends to send several selections within a few minutes, and a cold start (container start plus loading 7B in 4 bit) costs tens of seconds or more, while 5 idle minutes of a T4 cost a few cents per burst. 300 seconds keeps a reading session warm without holding the GPU for long. Memory snapshots are not added (GPU snapshots are not trivial here); listed as a follow-up.

### Endpoint

```python
MAX_TEXT_LENGTH = 4000
ERROR_TEXT_TOO_LONG = f"The 'text' field must be at most {MAX_TEXT_LENGTH} characters."

async def handle_verse_request(text, match) -> tuple[int, dict]:
    ... empty check as now ...
    if len(text) > MAX_TEXT_LENGTH:
        logger.info("[%s] rejected: status=422 reason=text_too_long input_length=%d", request_id, len(text))
        return 422, {"error": ERROR_TEXT_TOO_LONG, "id": request_id}
    ... as now, but: result = await match(text, request_id=request_id) ...

@app.function(image=api_image, timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
async def verse_api(body: VerseRequest):
    from fastapi.responses import JSONResponse
    status, payload = await handle_verse_request(body.text, MatchVerse().match.remote.aio)
    return JSONResponse(status_code=status, content=payload)
```

Why the length check is in `handle_verse_request` and not `Field(max_length=...)` on `VerseRequest`: FastAPI's default validation `422` is `{"detail": [...]}` and includes the submitted `input`, which would echo the text back, and `@modal.fastapi_endpoint` gives no FastAPI app to register a custom handler on. The handler check is simple, keeps the `{"error","id"}` contract and never echoes the text. `VerseRequest` stays `text: str`.

Why 4000: a browser selection the extension sends is a statement, usually one to a few paragraphs (a tweet is 280 characters, a long paragraph about 1000 to 1500). 4000 characters (about 1000 Mistral tokens) covers several paragraphs while bounding prompt size and generation cost on a T4.

## Files that change

- `wwjd_bot_modal.py`: `api_image`, `API_REQUIREMENTS`, `_load_model`, `_generate_verse` signature, `MatchVerse` class replacing `match_verse`, async `handle_verse_request` with the length check, `MAX_TEXT_LENGTH`, `ERROR_TEXT_TOO_LONG`, `verse_api` with no GPU on `api_image` awaiting `.remote.aio`.
- `requirements-api.in`, `requirements-api.txt`: new.
- `tests/test_load_once.py` (new): load once per container, no GPU on the endpoint, API image and lock, async non-blocking call, max length.
- Existing tests, call shapes only, no assertion weakened, no test removed:
  - every `handle_verse_request(...)` call becomes `asyncio.run(app_module.handle_verse_request(...))` (or a small local helper doing that), and the fakes become awaitable (`async def __call__` on `_Match`; lambdas become small `async def` functions) in `tests/test_verse_api.py`, `tests/test_logging.py`, `tests/test_logging_review_fixes.py`, `tests/test_parse_review_fixes.py`, `tests/test_parse_review_fixes_4.py`, `tests/test_parse_review_fixes_6.py`;
  - `tests/test_verse_api.py::test_verse_api_wrapper_returns_json_response`: monkeypatch `MatchVerse` so `MatchVerse().match.remote.aio` is the async fake, and `.remote` itself fails the test if called;
  - `tests/test_review_fixes.py::test_missing_hf_token_fails_loudly`: call `module.MatchVerse().match.local("some statement")`; same assertions (`RuntimeError` matching `HUGGINGFACE_TOKEN`, no `from_pretrained` call);
  - `tests/test_logging.py::test_match_verse_accepts_request_id_defaulting_to_none`: read the signature of the user class method `match` (via `MatchVerse._get_user_cls()`), expecting `["self", "text", "request_id"]` with `request_id` defaulting to `None`;
  - `tests/test_logging.py::test_match_verse_model_failure_surfaces_as_match_verse_error`: the loaders succeed with fakes, and the failure that holds `TEXT` is raised during generation (the fake tokenizer call or `generate`), since that is what now runs per request; same assertions (`MatchVerseError`, no cause or context, no sentinel in the error or logs, `RID` and the exception type in logs). The warning filter text changes to match the new local-execution warning if Modal emits one.
- `tests/test_image_pins.py`: add checks that `requirements-api.txt` is fully pinned, holds no torch, transformers, accelerate or bitsandbytes, and matches `requirements.txt` for every shared package; `requirements-api.in` pins fastapi and pydantic equal to `requirements.in`/`requirements.txt`. Existing checks must still pass (the GPU image still installs from `"requirements.txt"`).
- `README.md`: Architecture, Prerequisites (token check is in the enter hook), Deploy notes (cold versus warm, `scaledown_window`), API contract (new `422` row; the `timeout=600` row now mentions the `MatchVerse` container), Logging (the new reason `text_too_long` and the `model loaded` line), Dependencies (the API lock and its compile command).

## Order of work

1. Red: write `tests/test_load_once.py` and the new `test_image_pins.py` checks; run them failing for the expected reasons (no `MatchVerse`, no API lock, no length check, sync handler). Commit the red tests alone.
2. Add `requirements-api.in` and compile `requirements-api.txt` with the command above.
3. Refactor `wwjd_bot_modal.py` per the design, adapting the existing tests' call shapes in the same commit.
4. Run the gate; then update `README.md`.
5. Live drive (below).

## Risks

- Riskiest: the async conversion of `handle_verse_request`. Every existing test call site must be adapted; a missed `await` leaves a coroutine that is never awaited and a `TypeError` on unpacking, which the tests would catch. The exception handler must still catch exceptions raised by the awaited call.
- Modal `.local()` on a class method runs the enter hook; if Modal changes that, the load-once tests break loudly, not silently.
- The API image must still import `wwjd_bot_modal` at container start: the module top level imports only stdlib, `modal` and `pydantic`, and `REQUIREMENTS` path computation is harmless. torch and transformers stay imported inside functions only.
- Dropping `match_verse` breaks anyone calling it by name (`modal run`, other apps); none known in this repo.
- An enter-hook failure now fails the container rather than the single call; Modal retries container start for later calls, and each failing call gets the generic `500`.
- Chosen not to do: memory snapshots, `max_containers` or concurrency limits, serving FastAPI from the GPU class, a request body size limit (the body is parsed before the length check; the API container is CPU only, so this is not GPU work).

## Proof

- Criterion 1: `test_load_once.py` uses fake `transformers` and `torch` modules; calls `MatchVerse().match.local(...)` twice on one instance (and directly via `_get_user_cls()` with `load()` then `match()` twice) and asserts each loader's `from_pretrained` ran once and `model loaded` was logged once; asserts calling `match` twice after `load` makes zero further `from_pretrained` calls; asserts `@modal.enter` and `@modal.method` are on `load` and `match` (via AST or Modal metadata) and that the class service function spec has `gpus == "T4"`; asserts `scaledown_window=300` in the `@app.cls` call (AST).
- Criterion 2: `verse_api._spec_.gpus is None`; the `verse_api` decorator call uses `image=api_image` (AST); lock checks in `test_image_pins.py`.
- Criterion 4: a test with an `aio` fake that awaits a shared `asyncio.Event`; two `verse_api` calls gathered on one loop both enter the fake before the event is set, then both return `200`. `.remote` raises if called.
- Criterion 5: 4000 characters gives `200` and calls `match`; 4001 gives `422` with exactly `{"error": ERROR_TEXT_TOO_LONG, "id": hex}`, `match` not called, the text not in the body or the logs (flag off), the log has `reason=text_too_long`; a whitespace-only text over the limit gives the empty-text `422`.
- Criteria 6 and 7: the existing suite, adapted only in call shape, plus the existing override tests.
- Live drive (no Modal credentials, nothing deployed): a scratch script with fake `transformers` and `torch` modules imports the app, runs `MatchVerse().match.local` three times and prints stderr to show a single `model loaded` line; runs `verse_api`'s raw function with `fastapi.responses` stubbed and a fake `aio` for success, too long, empty, exception and parse failure, showing status, body and the log lines. Failure and edge paths: 4000 and 4001 characters, whitespace-only over the limit, missing token in the enter hook.
- Deferred, no Modal credentials: the live "second warm request shows no model-load log lines" check and the warm latency measurement. Commands to run with credentials:

  ```sh
  WWJD_APP_NAME=wwjd-bot-test WWJD_HF_CACHE_VOLUME=wwjd-hf-cache-test modal deploy wwjd_bot_modal.py
  URL=https://<workspace>--wwjd-bot-test-verse-api.modal.run
  for i in 1 2 3; do
    curl -s -o /dev/null -L -w '%{http_code} %{time_total}s\n' \
      -H 'Content-Type: application/json' -d '{"text": "Charity only encourages laziness."}' "$URL"
  done
  modal app logs wwjd-bot-test   # expect one "model loaded" line for the three requests
  modal app stop wwjd-bot-test
  ```

  The first request is the cold one; requests 2 and 3 within 300 seconds are warm.
