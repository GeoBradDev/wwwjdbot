# WWJD Bot

WWJD Bot takes a statement a user highlights on a web page and answers it with a Bible
verse that challenges it, plus a short explanation. It runs as a [Modal](https://modal.com)
app serving `mistralai/Mistral-7B-Instruct-v0.2` on a GPU.

## Architecture

1. **Chrome extension** ([GeoBradDev/WWJC_ChromeExtension](https://github.com/GeoBradDev/WWJC_ChromeExtension)).
   The user selects text and picks the context menu entry. `background.js` sends
   `POST https://brad-stricherz--wwjd-bot-verse-api.modal.run` with `{"text": "<selection>"}`,
   and `popup.js` shows `raw.reference`, `raw.verse` and `raw.explanation` from the response.
2. **`verse_api`** (in `wwjd_bot_modal.py`). A FastAPI web endpoint
   (`@modal.fastapi_endpoint(method="POST", docs=True)`). It validates the body and
   calls `match_verse` remotely.
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
different workspace or app name needs the extension's `background.js` updated to match.

The first request after a deploy downloads the model into the Volume and is slow; later
requests read it from the cache.

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
curl -X POST https://<workspace>--wwjd-bot-verse-api.modal.run \
  -H 'Content-Type: application/json' \
  -d '{"text": "Charity only encourages laziness."}'
```

Interactive API docs are at `/docs` on the same host.

## API contract

This is what the code does today. Issue #3 will revise the error handling, including status
codes.

Request: `POST` with a JSON body `{"text": "<statement>"}`.

| Case | Status | Body |
| --- | --- | --- |
| Success | `200` | `{"raw": {"reference": "...", "verse": "...", "explanation": "..."}}` |
| `text` is an empty string | `200` | `{"error": "Missing 'text' field in request"}` |
| `text` key missing, or the body is not valid for `{"text": str}` | `422` | FastAPI validation error |
| Any exception while calling `match_verse` | `200` | `{"error": "Processing failed: ..."}` |

On success the three `raw` fields are always strings but may be empty, for example when the
model output could not be split into fields. The rules, from `parse_model_output`:

- If the output contains a `{...}` block (first `{` to last `}`) that parses as JSON and has
  `reference`, `verse` and `explanation`, those three values are used.
- Otherwise the fields are pulled out of the text with regexes; missing ones are empty.
- If none are found, `reference` and `verse` are empty and `explanation` is the whole output.
- If the `{...}` block is not valid JSON, `match_verse` returns
  `{"error": "Parsing failed: ...", "raw_output": ...}`, and `verse_api` then responds `200`
  with all three `raw` fields empty.

The extension treats a response without `raw.reference` as "no comment received", which
covers the error bodies and the empty-field cases above.

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
  by hand. Regenerate it after changing `requirements.in` with:

  ```sh
  uv pip compile requirements.in -o requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
  ```

- `requirements-dev.txt`: local tooling (Modal CLI, pytest, ruff, and pydantic, which must
  match the version in `requirements.txt` because the app imports it at top level).
