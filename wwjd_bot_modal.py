import json
import os
import re
import time
import uuid
from pathlib import Path

import modal
from pydantic import BaseModel

app = modal.App(os.environ.get("WWJD_APP_NAME", "wwjd-bot"))

MODEL_DIR = "/root/.cache/huggingface"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
REQUIREMENTS = Path(__file__).parent / "requirements.txt"

# Create image with pinned dependencies from the compiled lock file
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(REQUIREMENTS)], uv_version="0.10.3")
    .env({"HF_HOME": MODEL_DIR})
)

# Use a persistent volume to cache Hugging Face models
model_volume = modal.Volume.from_name(
    os.environ.get("WWJD_HF_CACHE_VOLUME", "huggingface-cache"),
    create_if_missing=True,
)


class VerseRequest(BaseModel):
    text: str


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


def _parse_json_fields(generated_text: str) -> tuple[dict | None, bool]:
    """Return the first qualifying fields, and whether any valid JSON object
    had all three keys (so the model did answer in JSON)."""
    decoder = json.JSONDecoder()
    saw_all_keys = False
    for match in re.finditer(r"\{", generated_text):
        try:
            value, _ = decoder.raw_decode(generated_text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            fields = _non_empty_strings(value)
            if fields:
                return fields, True
            saw_all_keys = saw_all_keys or all(key in value for key in FIELDS)
    return None, saw_all_keys


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
            if not _FIELD_KEY.search(match.group(1)):
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
    fields, saw_json_answer = _parse_json_fields(generated_text)
    if not fields and not saw_json_answer:
        fields = _parse_regex_fields(generated_text)
    if not fields:
        return {"error": "parse_failed", "raw_output": generated_text}
    return {
        **fields,
        "model": model_name,
        "generation_time": f"{elapsed_time:.2f} seconds",
    }


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def match_verse(text: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = MODEL_NAME
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN is not set; attach the huggingface-token Modal Secret")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()

    prompt = build_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

    print("=== Prompt Sent ===")
    print(prompt)
    print("=== Model Output ===")
    print(generated_text)
    print(f"Response generated in {elapsed_time:.2f} seconds")

    return parse_model_output(generated_text, model_name, elapsed_time)


ERROR_EMPTY_TEXT = "The 'text' field must be a non-empty string."
ERROR_UNPARSEABLE = "The model returned a response that could not be parsed."
ERROR_INTERNAL = "Internal error while processing the request."


def handle_verse_request(text, match) -> tuple[int, dict]:
    request_id = uuid.uuid4().hex

    if not isinstance(text, str) or not text.strip():
        return 422, {"error": ERROR_EMPTY_TEXT, "id": request_id}

    try:
        result = match(text)
    except Exception as e:  # noqa: BLE001 any failure becomes a generic 500
        print(f"[{request_id}] match_verse failed: {type(e).__name__}: {e}")
        return 500, {"error": ERROR_INTERNAL, "id": request_id}

    if not isinstance(result, dict):
        reason = f"result is {type(result).__name__}, not dict"
    elif "error" in result:
        reason = f"match_verse returned error {result['error']!r}"
    else:
        fields = _non_empty_strings(result)
        if fields:
            return 200, {"raw": fields}
        reason = "a field is missing, empty or not a string"
    print(f"[{request_id}] unparseable model output: {reason}")
    return 502, {"error": ERROR_UNPARSEABLE, "id": request_id}


@app.function(image=image, gpu="any", timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
async def verse_api(body: VerseRequest):
    from fastapi.responses import JSONResponse

    status, payload = handle_verse_request(body.text, match_verse.remote)
    return JSONResponse(status_code=status, content=payload)
