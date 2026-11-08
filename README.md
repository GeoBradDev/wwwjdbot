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
   (`@modal.fastapi_endpoint(method="POST", docs=True)`). It validates the body,
   calls `match_verse` remotely and maps the result to a status code (see API contract).
3. **`match_verse`** (in `wwjd_bot_modal.py`). A GPU function (T4) that loads the model in
   4-bit from the Hugging Face cache Volume, builds the prompt with `build_prompt(text)`,
   generates, and turns the output into a dict with `parse_model_output(...)`.

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
  `match_verse` raises an error if the token is missing.
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
contract below); later requests read the weights from the cache.

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

### Try it

```sh
curl -L -H 'Content-Type: application/json' \
  -d '{"text": "Charity only encourages laziness."}' \
  https://<workspace>--wwjd-bot-verse-api.modal.run
```

`-d` makes the first request a `POST`. `-L` follows the `303` that Modal returns when a
request runs past 150 seconds; leave out `-X POST` so curl fetches the result URL with `GET`
as the redirect intends.

Interactive API docs are at `/docs` on the same host.

## API contract

Request: `POST` with a JSON body `{"text": "<statement>"}`.

| Case | Status | Body |
| --- | --- | --- |
| Success | `200` | `{"raw": {"reference": "...", "verse": "...", "explanation": "..."}}` |
| `text` is empty or only whitespace | `422` | `{"error": "The 'text' field must be a non-empty string.", "id": "<id>"}` |
| `text` key missing, or the body is not valid for `{"text": str}` | `422` | FastAPI's default validation error (`{"detail": [...]}`) |
| The model output cannot be parsed into three non-empty string fields, even by the fallback | `502` | `{"error": "The model returned a response that could not be parsed.", "id": "<id>"}` |
| Any exception while calling `match_verse` | `500` | `{"error": "Internal error while processing the request.", "id": "<id>"}` |
| Request still running after 150 seconds (Modal web endpoint limit, likely on a cold start) | `303` | Redirect to a result URL; each fetch of it waits up to another 150 seconds and returns one of the other responses here, or another `303` if the request is still running |
| `verse_api` itself runs past its `timeout=600` (possible on a cold start: the wait includes starting the T4 container for `match_verse` and downloading and loading the model) | Modal's timeout error | Modal's own error response, not a body from this code |

The error messages are fixed strings and never contain exception text or model output.
`id` is a random 32 character hex correlation id. The same id appears in every log line for
the request, from both `verse_api` and `match_verse`, so a reported `id` finds the matching
Modal log lines (see Logging and data handling).

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
- If the fallback does not find all three as non-empty strings, `match_verse` returns
  `{"error": "parse_failed", "raw_output": ...}` and `verse_api` responds `502`.

## Logging and data handling

The selected text a user sends can be personal or sensitive, so by default it is not logged.
All logging goes through the `wwjd_bot` logger at `INFO` and above, written to stderr, where
Modal keeps it with the app's logs.

By default each request logs metadata only:

- the request id (the `id` returned in `422`, `500` and `502` bodies);
- the input length, and for `422` the reason `empty_text`;
- from `match_verse`: the output length, the generation time and whether the output parsed;
- the outcome status, and for a `502` one fixed reason: `not_dict`, `match_error` or
  `invalid_fields`;
- for an exception: the exception type names (including chained causes) and the traceback
  frames as file, line and function name only. Exception messages, source lines and local
  variables are not logged, because a message can contain the input.

The user text, the prompt and the model output are never logged by default, and no exception
text or model output is ever returned in a response body, whatever the setting.

For debugging, set `WWJD_LOG_FULL_TEXT=1` in the shell that runs `modal serve` or
`modal deploy`. It is copied into the image env, so both containers log the full user text,
the full prompt, the model output (including the raw output behind a `502`) and full
exception tracebacks with messages:

```sh
WWJD_LOG_FULL_TEXT=1 modal serve wwjd_bot_modal.py
```

Only the exact value `1` turns it on; any other value, or unset, means off. It is off by
default. Redeploy without it as soon as you are done, since those logs then hold user text.

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

- `requirements.in`: top-level dependencies of the Modal image.
- `requirements.txt`: the compiled, fully pinned lock that the image installs. Do not edit it
  by hand. Regenerate it after changing `requirements.in` with the command below. This step
  alone needs [`uv`](https://docs.astral.sh/uv/) installed; nothing else in this README does.

  ```sh
  uv pip compile requirements.in -o requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
  ```

- `requirements-dev.txt`: local tooling (Modal CLI, pytest, ruff, and pydantic, which must
  match the version in `requirements.txt` because the app imports it at top level).
