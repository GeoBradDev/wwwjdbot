# WWJD Bot

WWJD Bot takes a statement a user highlights on a web page and answers it with a Bible
verse that challenges it, plus a short explanation. It runs as a [Modal](https://modal.com)
app serving `mistralai/Mistral-7B-Instruct-v0.2` on a GPU.

## Architecture

1. **Chrome extension** ([GeoBradDev/WWJC_ChromeExtension](https://github.com/GeoBradDev/WWJC_ChromeExtension)).
   What the backend relies on, as the extension currently behaves: it sends `POST` with
   `{"text": "<selection>"}` to the production URL and reads `raw.reference`, `raw.verse` and
   `raw.explanation` from the response. It treats any non-2xx status as a failure, shows the
   text of an `error` key when the body has one, and expects each `raw` field to be a
   non-empty string (anything else counts as an incomplete response). The URL is its `API_URL`
   constant (currently in `lib/request.js`). See the extension repo for the details of how it
   renders results and errors.
2. **`verse_api`** (in `wwjd_bot_modal.py`). A FastAPI web endpoint
   (`@modal.fastapi_endpoint(method="POST", docs=True)`) on a CPU-only container: it requests
   no GPU and runs on a light image (`api_image`) built from `requirements-api.txt`, which has
   `fastapi[standard]` (with its extras, such as uvicorn) and pydantic but no torch,
   transformers, accelerate or bitsandbytes. It takes the raw request and reads the body
   itself, stopping once it has read more than `MAX_BODY_BYTES` (64 KiB) and answering `413`,
   so it never parses more than the cap; then it decodes and validates the body as
   `{"text": str}` the way FastAPI would, awaits `MatchVerse().match.remote.aio(...)` without blocking its event loop, and
   maps the result to a status code (see API contract). `@modal.concurrent(max_inputs=20)`
   lets one CPU container serve up to 20 requests at once while they wait on the GPU;
   each `MatchVerse` container still runs one input at a time.
3. **`MatchVerse`** (in `wwjd_bot_modal.py`). A GPU class (`@app.cls`, T4). Its
   `@modal.enter()` hook `load` runs once when a container starts: it checks the token and
   loads the tokenizer and the model in 4-bit from the Hugging Face cache Volume. Its
   `@modal.method()` `match(text, request_id=None)` runs per request: it builds the prompt
   with `build_prompt(text)`, generates, and turns the output into a dict with
   `parse_model_output(...)`. A warm container serves later requests without reloading the
   model, and `scaledown_window=300` keeps it warm for 5 idle minutes (Modal's default is 60
   seconds), so a reading session with several selections stays fast while an idle T4 is
   released soon after. `max_containers` caps the number of `MatchVerse` containers running
   at once (default 2, see Cost limits). Before generating, `match` counts the tokens of the
   full prompt and refuses one over `MAX_PROMPT_TOKENS` (6144) without calling the model.

Everything lives in one file, `wwjd_bot_modal.py`. Modal ships only the entrypoint file to
the container automatically, so the helpers stay in it rather than a separate module.

## Prerequisites

- A Modal account and the Modal CLI. The CLI is pinned in `requirements-dev.txt`.
- Python 3.12.
- A Hugging Face account with access to the gated model
  [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  (open the model page and accept the license), and a Hugging Face access token with read
  permission.
- A Modal Secret named `huggingface-token` that holds the env var `HUGGINGFACE_TOKEN`.
  The `MatchVerse` enter hook raises an error if the token is missing, so the container
  fails to start. Modal reports an enter-hook failure as a task failure and shuts the
  container down; the waiting call fails only once Modal escalates the crash loop to input
  failures, which Modal's server decides. The call eventually fails and the client gets the
  generic `500`, but a `303` can come first, and if the escalation takes longer than
  `verse_api`'s `timeout=600` the client gets Modal's timeout response instead.
- A Modal Volume named `huggingface-cache` for the model weights. It is created on the first
  deploy (`create_if_missing=True`), so creating it by hand is optional.

## Deploy

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt

# Log in to Modal (opens a browser to create a token)
modal setup

# Store the Hugging Face token as a Modal Secret (use your own token, never commit it)
modal secret create huggingface-token HUGGINGFACE_TOKEN=<your-hf-token>

# Optional: create the model cache Volume ahead of time
modal volume create huggingface-cache
```

Run a temporary dev endpoint that reloads on code changes and stops when you press Ctrl+C:

```sh
modal serve wwjd_bot_modal.py
```

Modal adds a `-dev` suffix to URLs from `modal serve`, so the endpoint is
`https://<workspace>--wwjd-bot-verse-api-dev.modal.run`. The exact URL is printed in the
terminal.

Deploy the persistent endpoint:

```sh
modal deploy wwjd_bot_modal.py
```

The production URL follows the pattern `https://<workspace>--wwjd-bot-verse-api.modal.run`.
The extension calls `https://brad-stricherz--wwjd-bot-verse-api.modal.run`, so a deploy to a
different workspace or app name needs the extension's `API_URL` constant (currently in
`lib/request.js`) updated to match.

The first request after a deploy downloads the model into the Volume and loads it in 4-bit
on a T4, which can take longer than Modal's 150 second web request limit (see the API
contract below); later cold starts read the weights from the cache.

Cold versus warm: a request that finds no running `MatchVerse` container pays a cold start
(starting a T4 container and loading the model, tens of seconds or more), and the log shows
one `model loaded: load_time=<s>s` line for it. A request that reaches a warm container (one
that served a request in the last 300 seconds, the `scaledown_window`) only generates, and
logs no model load. The `verse_api` container is CPU only and starts quickly.

### Test deploy alongside production

Two env vars, read when the file is imported, let you deploy a separate copy without touching
the production app or its cache:

- `WWJD_APP_NAME` (default `wwjd-bot`): the Modal app name, which is also part of the URL.
- `WWJD_HF_CACHE_VOLUME` (default `huggingface-cache`): the Volume used for model weights.

```sh
WWJD_APP_NAME=wwjd-bot-test WWJD_HF_CACHE_VOLUME=wwjd-hf-cache-test modal deploy wwjd_bot_modal.py
```

That app is served at `https://<workspace>--wwjd-bot-test-verse-api.modal.run`. It uses the
same `huggingface-token` Secret.

### Cost limits

The endpoint has no authentication, so anyone with the URL can make it spend GPU time. Three
caps bound that cost:

- `max_containers` on `MatchVerse`: at most `MAX_GPU_CONTAINERS` T4 containers run at once.
  The default is 2, and an empty or unset `WWJD_MAX_GPU_CONTAINERS` means the default; set it
  (a whole number from 1 to 4294967295, the largest value Modal accepts; leading zeros are
  ignored) in the shell that runs `modal deploy` to change it. Like `WWJD_APP_NAME` it is
  read when the file is imported, and any other value fails the import with a message naming
  the env var, so a typo fails the deploy instead of removing the cap. A large value within
  that range effectively removes the cap. The cap is per app: a test deploy with its own
  `WWJD_APP_NAME` runs its own containers on top of production's. Requests beyond the cap queue in Modal: under load they wait longer and
  may get the `303` or Modal's timeout response (see API contract).
- `MAX_PROMPT_TOKENS` (6144): the full prompt, template included, is tokenized and refused
  over the cap before generation. Measured with the cached Mistral-7B-Instruct-v0.2
  tokenizer on 4000-character samples, the full prompt is: template alone 241 tokens,
  English 1113, Russian 2110, Arabic 3775, Thai 4242, common emoji 4242, Hindi 4398,
  Japanese 4550, Chinese 4606, Korean 5031. Common CJK characters and emoji are single
  tokens, so all of these fit. Byte fallback is refused: rare CJK characters take about 3
  tokens each (12242) and characters outside the vocabulary, such as U+20000, about 4
  (16242). Prompt plus the 500 generated tokens stays under 6,700 of the model's 32,768.
- `MAX_BODY_BYTES` (65536): `verse_api` answers `413` for a declared `Content-Length` over
  the cap without reading the body, and otherwise stops reading at the first chunk that takes
  the total past the cap, so it never parses more than 64 KiB. Modal delivers the body in
  chunks it sizes itself, so the bytes received can pass the cap by up to one chunk. The 4000
  character text, even at the worst JSON escape of 12 bytes per character, fits in 48,000
  bytes.

These caps do not bound the total spend over time. Set a spend limit in the Modal workspace
settings as well; this repo does not change Modal settings.

### Try it

```sh
curl -L -H 'Content-Type: application/json' \
  -d '{"text": "Charity only encourages laziness."}' \
  https://<workspace>--wwjd-bot-verse-api.modal.run
```

`-d` makes the first request a `POST`. `-L` follows the `303` that Modal returns when a
request runs past 150 seconds; leave out `-X POST` so curl fetches the result URL with `GET`
as the redirect intends.

Interactive API docs are at `/docs` on the same host. They do not show the request body
schema, because `verse_api` reads the body itself (see API contract).

## API contract

Request: `POST` with a JSON body `{"text": "<statement>"}` of at most 65536 bytes. The body is
parsed as JSON whatever the `Content-Type` header says.

| Case | Status | Body |
| --- | --- | --- |
| Success | `200` | `{"raw": {"reference": "...", "verse": "...", "explanation": "..."}}` |
| `text` is empty or only whitespace | `422` | `{"error": "The 'text' field must be a non-empty string.", "id": "<id>"}` |
| `text` is longer than 4000 characters (`MAX_TEXT_LENGTH`, counted with `len(text)`; whitespace-only text of any length is the empty-text case above) | `422` | `{"error": "The 'text' field must be at most 4000 characters.", "id": "<id>"}`, returned before any GPU work; the text is not echoed back |
| Body larger than 65536 bytes (`MAX_BODY_BYTES`), by its `Content-Length` or by the bytes actually read | `413` | `{"error": "The request body must be at most 65536 bytes.", "id": "<id>"}`, returned before any JSON parsing or GPU work |
| Body is empty, is not valid JSON (including whitespace only), has no `text` key, or is not valid for `{"text": str}` | `422` | FastAPI's default validation error (`{"detail": [...]}`), the same errors FastAPI itself gives: `missing` for an empty body or `null`, `model_attributes_type` for any other value that is not an object, `json_invalid` (with `input` `{}`) for invalid JSON, pydantic's errors otherwise |
| Body cannot be decoded at all: invalid UTF-8, nesting too deep for the JSON decoder, or an integer over Python's digit limit | `400` | `{"detail": "There was an error parsing the body"}`, FastAPI's own response for this case |
| Body decodes but fails validation with input nested too deeply to encode in the `422` (for example `{"text": ` followed by 2000 nested arrays) | `400` | the same `{"detail": "There was an error parsing the body"}`; here this code differs from plain FastAPI on purpose, which answered with a bare `500` because its `422` handler could not encode the input |
| The full prompt is longer than 6144 tokens (`MAX_PROMPT_TOKENS`) | `422` | `{"error": "The 'text' field is too long for the model; select less text.", "id": "<id>"}`, returned before generation; the text is not echoed back |
| The model output cannot be parsed into three non-empty string fields, even by the fallback | `502` | `{"error": "The model returned a response that could not be parsed.", "id": "<id>"}` |
| Any exception while calling `MatchVerse.match`, including a container whose enter hook fails (that call fails only once Modal escalates the crash loop to input failures, so a `303` can come first, or Modal's timeout response if escalation outlasts `timeout=600`) | `500` | `{"error": "Internal error while processing the request.", "id": "<id>"}` |
| Request still running after 150 seconds (Modal web endpoint limit, likely on a cold start) | `303` | Redirect to a result URL; each fetch of it waits up to another 150 seconds and returns one of the other responses here, or another `303` if the request is still running |
| `verse_api` itself runs past its `timeout=600` (possible on a cold start: the wait includes starting the T4 container for `MatchVerse` and downloading and loading the model; `MatchVerse` has its own `timeout=600`) | Modal's timeout error | Modal's own error response, not a body from this code |

The error messages are fixed strings and never contain exception text or model output.
`id` is a random 32 character hex correlation id. Every log record this code writes for the
request starts with `[<id>]`, in both containers, so a reported `id` finds the matching Modal
log lines (see Logging and data handling).

The `303` comes from Modal, not from this code: every Modal web endpoint has a 150 second
HTTP request limit, and a longer request is redirected to a result URL that can be reloaded
without starting a new request and blocks until the request completes (see
[Modal request timeouts](https://modal.com/docs/guide/webhook-timeouts)). A slow request can
produce a chain of `303`s, so clients must follow redirects repeatedly to get the result;
`curl -L` does.

On success all three `raw` fields are non-empty strings, stripped of surrounding
whitespace; there is no empty success. The rules, from `parse_model_output`:

- The first JSON object in the output (tried at each `{`, left to right) whose `reference`,
  `verse` and `explanation` are all non-empty strings is used. Other keys are dropped. A
  valid JSON object that has all three keys but does not qualify (empty, non-string or
  otherwise unacceptable values) is ignored: it is cut out, the fallback below reads only the rest
  of the output, and a value that would span the cut is not used. A
  value whose closing quote is followed on the same line by such an object also fails
  (`502`), as does a value containing a raw NUL character.
- Otherwise, including when the JSON is invalid (for example a trailing comma or an
  unescaped `"` inside a value), a regex fallback reads each `"key": "value"` pair. A value
  ends at the first of these that fits, so inner quotes are kept: a `"` followed by the
  next quoted `"key":` (with or without a comma) or a `}`; otherwise a `"` before a line
  end; otherwise a `"` at the end of the output (each optionally after a trailing comma).
  A value containing another field's `"key":` tries the next terminator in that order; an
  empty or whitespace value moves straight to the next occurrence of the key. JSON string
  escapes are decoded; a lone surrogate escape such as `\ud800` makes the field
  unacceptable (`502`). Known limits: a later quote in the output can end the last value,
  giving a `200` with the wrong text. This happens when same-line prose after the value
  ends in a quote (`"explanation": "C" as Jesus said "love"`), and when same-line prose is
  followed by a later line ending in a quote, and when a brace-less empty last value has a
  space before its comma (`"explanation": "" ,` then a later line ending in a quote).
  Without such a quote, same-line prose gives
  a `502` (`"explanation": "C" I hope this helps`).
- If the fallback does not find all three as non-empty strings, `MatchVerse.match` returns
  `{"error": "parse_failed", "raw_output": ...}` and `verse_api` responds `502`.

## Logging and data handling

The selected text a user sends can be personal or sensitive, so by default it is not logged.
All logging goes through the `wwjd_bot` logger at `INFO` and above, written to stderr, where
Modal keeps it with the app's logs.

By default each request logs metadata only:

- the request id (the `id` returned in the `413`, `422`, `500` and `502` bodies this code builds);
- the input length, or for an empty-text `422` only the reason `empty_text`; a too-long
  `422` logs the reason `text_too_long` and the input length;
- for a `413`: the reason `body_too_large` and either the declared `content_length` or the
  `bytes_read` when reading stopped (never the body);
- for a prompt over the token cap: from `MatchVerse.match`, the reason `input_too_long`,
  the input length, the prompt token count and the cap; from `verse_api`, `status=422
  reason=input_too_long`;
- from `MatchVerse.match`: the input and output lengths, the generation time and whether the output parsed;
- the outcome status, and for a `502` one fixed reason: `not_dict`, `match_error` or
  `invalid_fields`;
- once per `MatchVerse` container, when the enter hook finishes: `model loaded:
  load_time=<s>s` (no request id, since no request text exists yet);
- for an exception: the exception type names (including chained causes) and the traceback
  frames as file, line and function name only. Exception messages, source lines and local
  variables are not logged, because a message can contain the input.

The user text, the prompt and the model output are never logged by default. The bodies this
code builds (`413`, `422` for empty, too-long or too-many-token text, `500`, `502`) never hold exception text or model output,
whatever the setting. The `{"detail": [...]}` `422` and the `400` are raised by this code
(`_parse_verse_request`) as FastAPI's own exception types and rendered by FastAPI's handlers.
Their errors come from pydantic and the JSON decoder: the messages are fixed parser phrases,
and a pydantic error's `input` holds the submitted value (for example a non-string `text`),
which goes back only to the same client and is never logged.

For debugging, set `WWJD_LOG_FULL_TEXT=1` in the shell that runs `modal serve` or
`modal deploy`. It is copied into the env of both images (`image` and `api_image`), so both containers log the full user text,
the full prompt, the model output (including the raw output behind a `502`) and full
exception tracebacks with messages:

```sh
WWJD_LOG_FULL_TEXT=1 modal serve wwjd_bot_modal.py
```

Only the exact value `1` turns it on; any other value, or unset, means off. It is off by
default. While it is on, importing the app logs a `WARNING` saying so, so a flag left
exported in the shell is visible at the next `modal deploy`. Redeploy without it as soon as you
are done, since those logs then hold user text.

## Development

Lint and test locally (no GPU, torch, transformers or Modal credentials needed):

```sh
pip install -r requirements-dev.txt
ruff check .
pytest
python -c "import wwjd_bot_modal; print(wwjd_bot_modal.app.name)"
```

CI (`.github/workflows/ci.yml`) runs the same three checks on every push and pull request.

### Dependencies

- `requirements.in`: top-level dependencies of the GPU image (`MatchVerse`).
- `requirements.txt`: the compiled, fully pinned lock that the GPU image installs. Do not edit it
  by hand. Regenerate it after changing `requirements.in` with the command below. The two lock
  compile steps alone need [`uv`](https://docs.astral.sh/uv/) installed; nothing else in this README does.

  ```sh
  uv pip compile requirements.in -o requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
  ```

- `requirements-api.in`: top-level dependencies of the CPU image for `verse_api` (`fastapi[standard]`
  and pydantic, at the same versions as the GPU lock).
- `requirements-api.txt`: its compiled lock. Do not edit it by hand. It is compiled with
  `-c requirements.txt` so every shared package has the same version in both containers;
  recompile it after changing `requirements-api.in` or `requirements.txt`:

  ```sh
  uv pip compile requirements-api.in -o requirements-api.txt -c requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
  ```

- `requirements-dev.txt`: local tooling (Modal CLI, pytest, ruff, and pydantic, which must
  match the version in `requirements.txt` because the app imports it at top level).
