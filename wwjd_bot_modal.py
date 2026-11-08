import json
import logging
import os
import re
import time
import traceback
import uuid
from pathlib import Path

import modal
from pydantic import BaseModel, ValidationError

try:
    # FastAPI resolves verse_api's annotation from this module's globals and
    # passes the raw request for starlette's Request. The API container has
    # starlette (via fastapi); local tooling and CI do not, and never serve.
    from starlette.requests import Request
except ImportError:
    Request = None

app = modal.App(os.environ.get("WWJD_APP_NAME", "wwjd-bot"))

LOG_FULL_TEXT_ENV = "WWJD_LOG_FULL_TEXT"
logger = logging.getLogger("wwjd_bot")


def _configure_logger() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _log_full_text() -> bool:
    # Read at every call. Off unless the value is exactly "1".
    return os.environ.get(LOG_FULL_TEXT_ENV) == "1"


_configure_logger()
if _log_full_text():
    # Visible at a later modal deploy from a shell where the flag was exported.
    logger.warning(
        "%s=1: full text logging is on; user text, prompts, model output and exception messages will be logged",
        LOG_FULL_TEXT_ENV,
    )


def _full_text_image_env() -> dict:
    return {LOG_FULL_TEXT_ENV: "1"} if _log_full_text() else {}

MAX_GPU_CONTAINERS_ENV = "WWJD_MAX_GPU_CONTAINERS"
DEFAULT_MAX_GPU_CONTAINERS = 2


def _max_gpu_containers() -> int:
    # Read at import, like WWJD_APP_NAME: the value in the shell that runs
    # modal deploy is what Modal registers. A bad value fails the deploy
    # instead of silently removing the cap.
    raw = os.environ.get(MAX_GPU_CONTAINERS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_GPU_CONTAINERS
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) < 1:
        raise ValueError(f"{MAX_GPU_CONTAINERS_ENV} must be a whole number of at least 1")
    return int(raw)


# Upper bound on concurrently running MatchVerse (GPU) containers; requests
# beyond it queue in Modal.
MAX_GPU_CONTAINERS = _max_gpu_containers()

MODEL_DIR = "/root/.cache/huggingface"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
API_REQUIREMENTS = Path(__file__).parent / "requirements-api.txt"

# GPU image for MatchVerse, with pinned dependencies from the compiled lock file
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(REQUIREMENTS)], uv_version="0.10.3")
    .env({"HF_HOME": MODEL_DIR})
)
# CPU image for verse_api: fastapi[standard] and pydantic, no torch, from its own lock
api_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    requirements=[str(API_REQUIREMENTS)], uv_version="0.10.3"
)
# WWJD_LOG_FULL_TEXT=1 in the shell that runs modal serve/deploy turns full
# text logging on in both containers; by default the images are unchanged.
if _full_text_image_env():
    image = image.env(_full_text_image_env())
    api_image = api_image.env(_full_text_image_env())

# Use a persistent volume to cache Hugging Face models
model_volume = modal.Volume.from_name(
    os.environ.get("WWJD_HF_CACHE_VOLUME", "huggingface-cache"),
    create_if_missing=True,
)


class VerseRequest(BaseModel):
    text: str


_REQUEST_ID = re.compile(r"[0-9a-f]{32}")


def _normalize_request_id(request_id) -> str:
    if isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id):
        return request_id
    return uuid.uuid4().hex


def _exception_types(exc: BaseException) -> str:
    # Type names of exc and its chained causes/contexts, outermost first, cycle safe.
    names, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        names.append(f"{type(exc).__module__}.{type(exc).__qualname__}")
        exc = exc.__cause__ or exc.__context__
    return " <- ".join(names)


def _traceback_frames(exc: BaseException) -> str:
    # File, line and function only: no source lines, messages or locals.
    return "\n".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
        for frame in traceback.extract_tb(exc.__traceback__)
    )


def _log_exception(request_id: str, where: str, exc: BaseException) -> None:
    # Exception messages can embed the user text, so by default only type
    # names and code locations are logged.
    if _log_full_text():
        logger.error("[%s] %s failed", request_id, where, exc_info=exc)
        return
    logger.error(
        "[%s] %s failed: %s (message withheld; set %s=1 to log it)\nTraceback (most recent call last):\n%s",
        request_id,
        where,
        _exception_types(exc),
        LOG_FULL_TEXT_ENV,
        _traceback_frames(exc),
    )


def build_prompt(text: str) -> str:
    return f"""<s>[INST]
    Evaluate the following statement from a Christian biblical perspective:

    "{text}"

    Determine whether this statement reflects or contradicts the values taught by Jesus and core Christian doctrine.

    If the statement conflicts with biblical teachings, generate a clear and direct response that:
    1. Identifies a Bible verse that challenges the statement
    2. Presents the full text of that verse
    3. Provides a brief but pointed explanation that highlights the contradiction or hypocrisy in light of Christian values

    The tone should be respectful but unflinching—especially when the statement misrepresents Christian teaching or uses it to justify harm, exclusion, or hate.

    Return your response in JSON format with these fields:
    {{
      "reference": "The Bible verse reference (e.g., Matthew 5:44)",
      "verse": "The full text of the verse",
      "explanation": "A short explanation (1–2 sentences) that clearly shows how the verse challenges the statement"
    }}
    [/INST]"""


FIELDS = ("reference", "verse", "explanation")

# A fallback value ends, in this order of preference for each start, at a
# quote followed by: (1) the next quoted "key": (with or without a comma) or
# an optional comma and a closing brace; (2) an optional comma, spaces or
# tabs, and a line end; (3) an optional comma and whitespace to the end of
# the text.
_VALUE_TIERS = tuple(
    re.compile(r"(.*?)(?=" + end + ")", re.DOTALL)
    for end in (
        r'"\s*(?:,?\s*"[A-Za-z_][A-Za-z0-9_-]*"\s*:|,?\s*\})',
        r'",?[ \t]*\r?\n',
        r'"\s*,?\s*\Z',
    )
)
# A captured value that contains another field's key ran past its own end.
_FIELD_KEY = re.compile(r'"(?:reference|verse|explanation)"\s*:')


def _encodable(value: str) -> bool:
    # A lone surrogate (for example from \ud800) makes Starlette's JSONResponse
    # raise UnicodeEncodeError outside the handler.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _non_empty_strings(values: dict) -> dict | None:
    fields = {}
    for key in FIELDS:
        value = values.get(key)
        if not isinstance(value, str) or not value.strip() or not _encodable(value):
            return None
        fields[key] = value.strip()
    return fields


def _parse_json_fields(generated_text: str) -> tuple[dict | None, str]:
    """Return the first qualifying fields, and the text for the fallback: the
    output with every valid but non-qualifying object that has all three
    keys replaced by _CUT."""
    decoder = json.JSONDecoder()
    spans = []
    for match in re.finditer(r"\{", generated_text):
        try:
            value, end = decoder.raw_decode(generated_text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            fields = _non_empty_strings(value)
            if fields:
                return fields, generated_text
            if all(key in value for key in FIELDS):
                spans.append((match.start(), end))
    return None, _remove_spans(generated_text, spans)


# Marks where an ignored object was cut out. It is not whitespace, so a
# "key": right before the cut does not start a value, and a fallback value
# containing it would span the cut and is not used.
_CUT = "\x00"


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts = []
    position = 0
    for start, end in merged:
        parts.append(text[position:start])
        parts.append(_CUT)
        position = end
    parts.append(text[position:])
    return "".join(parts)


_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_ESCAPE = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
    r'|\\(["\\/bfnrt])'
    r"|\\u([0-9a-fA-F]{4})"
)


def _decode_escape(match: re.Match) -> str:
    high, low, simple, code = match.groups()
    if high:
        return chr(0x10000 + ((int(high, 16) - 0xD800) << 10) + (int(low, 16) - 0xDC00))
    if simple:
        return _SIMPLE_ESCAPES[simple]
    return chr(int(code, 16))


def _decode_fallback_value(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except json.JSONDecodeError:
        # One left-to-right pass, so "\\n" is a backslash then "n". Unknown
        # escapes stay as written; a lone surrogate is rejected later.
        return _ESCAPE.sub(_decode_escape, value)


def _find_fallback_value(generated_text: str, key: str) -> str | None:
    for start in re.finditer(rf'"{key}"\s*:\s*"', generated_text):
        for value_re in _VALUE_TIERS:
            match = value_re.match(generated_text, start.end())
            if not match:
                continue
            if not match.group(1).strip():
                # An empty value: a later tier would only reach a later quote.
                break
            if _CUT not in match.group(1) and not _FIELD_KEY.search(match.group(1)):
                return match.group(1)
    return None


def _parse_regex_fields(generated_text: str) -> dict | None:
    values = {}
    for key in FIELDS:
        value = _find_fallback_value(generated_text, key)
        if value is not None:
            values[key] = _decode_fallback_value(value)
    return _non_empty_strings(values)


def parse_model_output(generated_text: str, model_name: str, elapsed_time: float) -> dict:
    fields, fallback_text = _parse_json_fields(generated_text)
    if not fields:
        fields = _parse_regex_fields(fallback_text)
    if not fields:
        return {"error": "parse_failed", "raw_output": generated_text}
    return {
        **fields,
        "model": model_name,
        "generation_time": f"{elapsed_time:.2f} seconds",
    }


class MatchVerseError(RuntimeError):
    """Raised by MatchVerse.match in place of any generation failure; carries only the request id."""


def _run_guarded(request_id: str, func, *args):
    try:
        return func(*args)
    except Exception as e:  # noqa: BLE001 log safely, then re-raise without the message
        _log_exception(request_id, "MatchVerse.match", e)
    # Raised outside the except block, so it has no __cause__ or __context__.
    raise MatchVerseError(f"MatchVerse.match failed for request {request_id}")


def _log_generation(request_id, text, prompt, generated_text, elapsed_time, model_name) -> dict:
    result = parse_model_output(generated_text, model_name, elapsed_time)
    logger.info(
        "[%s] generated: input_length=%d output_length=%d generation_time=%.2fs parse=%s",
        request_id,
        len(text),
        len(generated_text),
        elapsed_time,
        "parse_failed" if "error" in result else "ok",
    )
    if _log_full_text():
        logger.info("[%s] prompt sent:\n%s", request_id, prompt)
        logger.info("[%s] model output:\n%s", request_id, generated_text)
    return result


def _load_model(token: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


# A cost bound, not a fit bound: Mistral-7B-Instruct-v0.2 takes 32,768 tokens.
# The template is roughly 250 tokens and 4000 characters of text is roughly
# 1000 to 1500 tokens (about 4000 for dense scripts), so 4096 accepts every
# realistic selection but rejects the byte-fallback worst case (about 4
# tokens per character). These counts are estimates; see the issue #12 plan.
MAX_PROMPT_TOKENS = 4096
RESULT_INPUT_TOO_LONG = "input_too_long"


def _tokenize_capped(tokenizer, prompt: str, max_tokens: int):
    """Tokenize prompt once. Return (inputs, token_count) when the count is at
    most max_tokens, else (None, token_count)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    count = inputs.input_ids.shape[1]
    if count > max_tokens:
        return None, count
    return inputs, count


def _generate_verse(tokenizer, model, text: str, request_id: str) -> dict:
    import torch

    prompt = build_prompt(text)

    inputs, prompt_tokens = _tokenize_capped(tokenizer, prompt, MAX_PROMPT_TOKENS)
    if inputs is None:
        logger.info(
            "[%s] rejected: reason=input_too_long input_length=%d prompt_tokens=%d max_prompt_tokens=%d",
            request_id,
            len(text),
            prompt_tokens,
            MAX_PROMPT_TOKENS,
        )
        return {"error": RESULT_INPUT_TOO_LONG}
    inputs = inputs.to(model.device)
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            min_length=inputs.input_ids.shape[1] + 20,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    elapsed_time = time.time() - start_time

    return _log_generation(request_id, text, prompt, generated_text, elapsed_time, MODEL_NAME)


# One GPU container serves many requests: the enter hook loads the model once
# per container, and scaledown_window keeps a warm container for 5 idle minutes.
@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    gpu="T4",
    scaledown_window=300,
    max_containers=MAX_GPU_CONTAINERS,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
class MatchVerse:
    @modal.enter()
    def load(self):
        # Runs before any request text exists, so a failure here cannot carry
        # user text; it is not wrapped in _run_guarded.
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


MAX_TEXT_LENGTH = 4000

ERROR_EMPTY_TEXT = "The 'text' field must be a non-empty string."
ERROR_UNPARSEABLE = "The model returned a response that could not be parsed."
ERROR_INTERNAL = "Internal error while processing the request."
ERROR_TEXT_TOO_LONG = f"The 'text' field must be at most {MAX_TEXT_LENGTH} characters."
ERROR_TOO_MANY_TOKENS = "The 'text' field is too long for the model; select less text."


async def handle_verse_request(text, match) -> tuple[int, dict]:
    request_id = uuid.uuid4().hex

    if not isinstance(text, str) or not text.strip():
        logger.info("[%s] rejected: status=422 reason=empty_text", request_id)
        return 422, {"error": ERROR_EMPTY_TEXT, "id": request_id}

    # Checked here rather than with Field(max_length=...): FastAPI's own 422
    # body would echo the submitted text back.
    if len(text) > MAX_TEXT_LENGTH:
        logger.info("[%s] rejected: status=422 reason=text_too_long input_length=%d", request_id, len(text))
        return 422, {"error": ERROR_TEXT_TOO_LONG, "id": request_id}

    logger.info("[%s] request: input_length=%d", request_id, len(text))
    if _log_full_text():
        logger.info("[%s] input text: %r", request_id, text)

    try:
        result = await match(text, request_id=request_id)
    except Exception as e:  # noqa: BLE001 any failure becomes a generic 500
        _log_exception(request_id, "MatchVerse.match call", e)
        logger.info("[%s] done: status=500", request_id)
        return 500, {"error": ERROR_INTERNAL, "id": request_id}

    if isinstance(result, dict) and result.get("error") == RESULT_INPUT_TOO_LONG:
        logger.info("[%s] done: status=422 reason=input_too_long", request_id)
        return 422, {"error": ERROR_TOO_MANY_TOKENS, "id": request_id}

    if not isinstance(result, dict):
        reason = "not_dict"
    elif "error" in result:
        reason = "match_error"
    else:
        fields = _non_empty_strings(result)
        if fields:
            logger.info("[%s] done: status=200", request_id)
            return 200, {"raw": fields}
        reason = "invalid_fields"
    logger.warning("[%s] unparseable model output: status=502 reason=%s", request_id, reason)
    if _log_full_text() and isinstance(result, dict) and isinstance(result.get("raw_output"), str):
        logger.info("[%s] raw model output: %r", request_id, result["raw_output"])
    return 502, {"error": ERROR_UNPARSEABLE, "id": request_id}


# A valid body is {"text": ...} with at most 4000 characters; the worst JSON
# encoding of one character is a 12-byte surrogate pair escape, so 48,000
# bytes plus framing. 64 KiB leaves room for whitespace.
MAX_BODY_BYTES = 65536
ERROR_BODY_TOO_LARGE = f"The request body must be at most {MAX_BODY_BYTES} bytes."
_CONTENT_LENGTH = re.compile(r"[0-9]+")


async def read_limited_body(headers, chunks, max_bytes: int) -> tuple[bytes | None, str]:
    """Return (body, "") when the body is at most max_bytes, else (None, measure).

    A Content-Length above the cap rejects without reading (measure
    "content_length=<n>"). Otherwise chunks are read until the running total
    exceeds the cap, and nothing after that chunk is read (measure
    "bytes_read=<n>"), so a missing or lying Content-Length is still bounded."""
    declared = (headers.get("content-length") or "").strip()
    if _CONTENT_LENGTH.fullmatch(declared) and int(declared) > max_bytes:
        return None, f"content_length={int(declared)}"
    parts, total = [], 0
    async for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            return None, f"bytes_read={total}"
        parts.append(chunk)
    return b"".join(parts), ""


def _parse_verse_request(raw: bytes) -> VerseRequest:
    """Validate raw as FastAPI does for a `body: VerseRequest` parameter, and
    raise FastAPI's RequestValidationError with the same errors, so the client
    gets the default 422 {"detail": [...]}. Invalid JSON gets FastAPI's own
    json_invalid error, whose input is {} rather than the body."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        errors = [_json_invalid(e.pos, e.msg)]
    except UnicodeDecodeError as e:
        errors = [_json_invalid(e.start, "invalid UTF-8")]
    else:
        try:
            return VerseRequest.model_validate(data)
        except ValidationError as e:
            errors = [{**error, "loc": ("body", *error["loc"])} for error in e.errors(include_url=False)]
    from fastapi.exceptions import RequestValidationError

    raise RequestValidationError(errors)


def _json_invalid(position: int, message: str) -> dict:
    return {
        "type": "json_invalid",
        "loc": ("body", position),
        "msg": "JSON decode error",
        "input": {},
        "ctx": {"error": message},
    }


# No GPU: this container only validates, awaits MatchVerse and maps the result.
# Up to 20 requests share one container while they wait on GPU calls; each
# MatchVerse container still runs one input at a time. app.function must stay
# outermost: Modal raises if concurrent is stacked above it.
@app.function(image=api_image, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="POST", docs=True)
async def verse_api(request: Request):
    from fastapi.responses import JSONResponse

    # The body is read by hand, with a byte cap, before any JSON parsing.
    raw, measure = await read_limited_body(request.headers, request.stream(), MAX_BODY_BYTES)
    if raw is None:
        request_id = uuid.uuid4().hex
        logger.info("[%s] rejected: status=413 reason=body_too_large %s", request_id, measure)
        return JSONResponse(status_code=413, content={"error": ERROR_BODY_TOO_LARGE, "id": request_id})
    body = _parse_verse_request(raw)
    status, payload = await handle_verse_request(body.text, MatchVerse().match.remote.aio)
    return JSONResponse(status_code=status, content=payload)
