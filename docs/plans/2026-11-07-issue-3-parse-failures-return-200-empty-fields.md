# Plan: Parse failures return HTTP 200 with empty fields; regex fallback is unreachable (issue #3)

Risk tier: High. Not a forced-High category. Starts Low and rises one tier for changing the public API contract (status codes and error body), one for altering existing behavior rather than adding, and stays High because the Modal/FastAPI wrapper itself cannot be exercised by the CI unit tests (FastAPI is not installed in CI and there is no GPU or Modal credential for a live deploy). Plan approved by the orchestrator on the user's behalf.

Domain skills:
- superpowers:test-driven-development: invoke.
- python-fastapi: invoke, for the `JSONResponse` and status code handling in `verse_api`.
- context7-plugin:docs: invoke if needed to confirm that a `@modal.fastapi_endpoint` function may return a `fastapi.responses.JSONResponse` with a custom status code.
- django, django-ninja-api, expo, react, frontend skills: not applicable because the repo is a single Modal Python app.

## Acceptance criteria

1. A JSON parse failure in `parse_model_output` runs the regex fallback instead of returning an error. When the fallback also cannot produce all three fields, the client gets a non-2xx status with an error message.
2. Empty or whitespace-only `text` returns `422`.
3. Unit tests feed recorded model outputs into `parse_model_output()` and cover: valid JSON, a verse containing unescaped `"`, a trailing comma, two JSON objects, and no JSON at all.
4. The fallback regex no longer truncates a value at an escaped quote (`\"`); escapes in fallback values are decoded.
5. API contract (decided by the user, "Additive"):
   - Success is unchanged: `200` with `{"raw": {"reference", "verse", "explanation"}}`.
   - Every error is a non-2xx `JSONResponse` with the stable body `{"error": "<message>", "id": "<correlation id>"}`: `422` for empty or whitespace-only `text`, `502` for model output that cannot be parsed even after the fallback, `500` for any unexpected failure (for example an exception from `match_verse.remote`). Not FastAPI's `HTTPException` `{"detail": ...}` shape.
   - Error messages are fixed generic strings. They never contain exception text or raw model output.
   - No empty success: if `reference`, `verse` or `explanation` is not a non-empty string (after stripping whitespace), the response is a `502`, never a `200`.
6. Endpoint-level tests prove the status codes and bodies for success, empty text, whitespace text, parse failure, partial fields, non-string fields, and an exception from `match_verse`, with `match_verse` mocked, and no GPU, torch, transformers, FastAPI or Modal credentials in CI.
7. `README.md`'s API contract section describes the new behavior; the "Issue #3 will revise" note is removed.
8. The CI gate (`ruff check .`, `pytest`, the import check) passes on the PR.

## Design

### `parse_model_output(generated_text, model_name, elapsed_time) -> dict`

Signature unchanged. Two outcomes only:

- Success: `{"reference", "verse", "explanation", "model", "generation_time"}` where the three fields are non-empty strings, stripped of surrounding whitespace, and `generation_time` keeps the existing `f"{elapsed_time:.2f} seconds"` format.
- Failure: `{"error": "parse_failed", "raw_output": generated_text}`. The `raw_output` key stays for server-side debugging; `verse_api` never sends it to the client.

Steps:

1. JSON path. For each `{` in the text, left to right, try `json.JSONDecoder().raw_decode(generated_text, idx)`. Catch only `json.JSONDecodeError` (no blind `except`). Accept the first decoded value that is a `dict` whose `reference`, `verse` and `explanation` are all strings that are non-empty after `.strip()`. This handles JSON embedded in prose and multiple objects (the first complete one wins) and replaces the greedy `{[\s\S]*}` match. Bound the work: stop after the first accepted object; `raw_decode` at each `{` is fine for 500-token outputs. If no object qualifies but at least one syntactically valid object had all three keys (with empty, whitespace, non-string or non-UTF-8 values), the model did answer in JSON: return the failure dict without running step 2 (round 4 fix).
2. Regex fallback, run whenever step 1 accepts nothing and saw no valid object with all three keys. For each key, search for `"<key>"\s*:\s*"` and capture the value lazily (`(.*?)`, with `re.DOTALL`) up to a closing `"`. For each start, the end is chosen in this order, taking the first tier that gives an acceptable value: (1) a `"` followed by the next quoted key with an optional comma (`,?\s*"[A-Za-z_][A-Za-z0-9_-]*"\s*:`, so keys separated only by a newline or a space, and keys with digits, also end a value) or by an optional `,` then `}`; (2) otherwise a `"` followed by an optional `,`, optional spaces or tabs, and a line end (brace-less fields followed by prose on a new line, review fix B); (3) otherwise a `"` followed by an optional `,` and whitespace to the end of the output. A value is acceptable if it is non-empty after `.strip()` and does not contain `"reference"\s*:`, `"verse"\s*:` or `"explanation"\s*:` (such a value ran past its own end, for example a key named in prose before the real object; review fix A). So an unescaped inner quote before a raw newline (`"He said "Peace"\nand left"`) does not cut a value short when a key follows (delta review fix 1), and the last value stops at a line end before quoted prose on a later line instead of running to the last quote in the output (round 3 fix). A tier that matches with an empty or whitespace value ends the tiers for that start (a later tier would only reach a later quote; round 4 fix), while a value containing another field's key falls through to the next tier. If no tier fits, the search for that key continues at the next occurrence of `"<key>"\s*:\s*"`, and the first acceptable candidate wins. This keeps an unescaped inner quote (`"He said "love" them"`) and an escaped one (`\"`) inside the value, and stops correctly before a trailing comma. Then decode the captured text: try `json.loads('"' + value + '"')`; on `json.JSONDecodeError` (for example because of unescaped inner quotes), decode in a single left-to-right pass over `\\(["\\/bfnrt]|u[0-9a-fA-F]{4})`, so `\\` is consumed before a following `n`, joining a valid `\uD8xx\uDCxx` surrogate pair into one character; unknown escapes stay as written, and a lone surrogate is then rejected by the UTF-8 check in step 3 (review fix D). Strip the result.
3. If all three fallback values are non-empty, return success. In both paths, and in `handle_verse_request` step 4, a value is acceptable only if it is a string, non-empty after `.strip()`, and `value.encode("utf-8")` succeeds: a lone surrogate (from `\ud800` in valid JSON or in the fallback) would make Starlette's `JSONResponse.render` raise `UnicodeEncodeError` outside the handler and return a plain-text `500` with no `id` (review fix C). Otherwise return the failure dict. The old "whole text becomes `explanation`" path and the "partial fields succeed" path are removed: both produced a `200` with empty fields.

Remove the `try`/`except Exception` with its `# noqa: BLE001` from `parse_model_output`; no blind catch remains there.

### `handle_verse_request(text, match) -> tuple[int, dict]`

New module-level pure function in `wwjd_bot_modal.py` (no FastAPI import), so the endpoint logic is testable in CI. `match` is a callable that takes the text and returns the `match_verse` result (in production, `match_verse.remote`).

1. `request_id = uuid.uuid4().hex`.
2. If `text` is not a string, or `text.strip()` is empty: return `422, {"error": "The 'text' field must be a non-empty string.", "id": request_id}`. Do not call `match`.
3. Call `match(text)` (the original text, not stripped) inside `try`/`except Exception` (keep `# noqa: BLE001` with the reason "any failure becomes a generic 500"). On exception, `print` one line with the id and the exception type and message for the Modal logs (issue #5 owns logging; do not add more), and return `500, {"error": "Internal error while processing the request.", "id": request_id}`.
4. If the result is not a `dict`, has an `error` key, or any of `reference`, `verse`, `explanation` is not a string that is non-empty after `.strip()`: `print` one line with the id and a short reason (not the raw output, which `match_verse` already prints), and return `502, {"error": "The model returned a response that could not be parsed.", "id": request_id}`.
5. Otherwise return `200, {"raw": {"reference": ..., "verse": ..., "explanation": ...}}` with exactly those three keys (no `model`, `generation_time` or `id`, so the success shape is unchanged).

Export the three messages as module constants (`ERROR_EMPTY_TEXT`, `ERROR_UNPARSEABLE`, `ERROR_INTERNAL`) so tests compare against them.

### `verse_api`

```python
async def verse_api(body: VerseRequest):
    from fastapi.responses import JSONResponse

    status, payload = handle_verse_request(body.text, match_verse.remote)
    return JSONResponse(status_code=status, content=payload)
```

`fastapi` is imported inside the function because it is in the Modal image (`requirements.txt`) but not in `requirements-dev.txt`, and the CI import check must keep working without it. The Pydantic `422` for a missing `text` key is left as FastAPI's default shape (documented in the README).

## Files that change

- `wwjd_bot_modal.py`: add `import uuid`; rewrite `parse_model_output` per Design; add the three error constants and `handle_verse_request`; reduce `verse_api` to the wrapper above. `build_prompt`, `match_verse`, the image and the prompt are untouched.
- `tests/test_prompt_and_parsing.py`: the parse tests from issue #6 pinned the old buggy behavior on purpose; update them to the corrected behavior and add the new cases (see Proof). The module docstring changes to say the parse tests pin the issue #3 behavior. The `build_prompt` tests are not touched.
- `tests/test_verse_api.py` (new): endpoint tests for `handle_verse_request`, plus one test of the `verse_api` wrapper that stubs `fastapi.responses` in `sys.modules` with a minimal `JSONResponse` recorder, replaces the module attribute `match_verse` (monkeypatch) with an object whose `.remote` is a mock, and awaits the raw coroutine behind `verse_api` (modal 1.6.3 has no `get_raw_f()`; it exposes the `_raw_f_` property, a wrapper whose `inspect.unwrap` is the original `async def`, so the test and the live drive use `inspect.unwrap(verse_api._raw_f_)`). If neither works cleanly, test the wrapper by AST instead (it calls `handle_verse_request` with `match_verse.remote` and returns `JSONResponse(status_code=..., content=...)`) and say so in the plan.
- `README.md`: rewrite the `## API contract` section (table rows and the parse rules) per criterion 5; remove "This is what the code does today. Issue #3 will revise..."; keep the `303` and timeout rows; replace the paragraph about `raw` fields carrying any JSON values with: on success all three are non-empty strings. Update Architecture item 2 to say `verse_api` maps the result to a status code. No other README edits.
- `docs/plans/2026-11-07-issue-3-parse-failures-return-200-empty-fields.md`: this plan.

## Order of work

1. Commit this plan alone (`docs: plan for issue 3`).
2. Red test (defect 1, fallback unreachable plus the five recorded outputs): update and add parse tests in `tests/test_prompt_and_parsing.py`; confirm they fail for the expected reason; commit alone; add to `.claude/fix-lock`.
3. Implement `parse_model_output`; green; commit.
4. Red test (defect 2, status codes): `tests/test_verse_api.py`; confirm red (the function does not exist yet, or returns 200); commit alone; add to `.claude/fix-lock`.
5. Implement `handle_verse_request` and the `verse_api` wrapper; green; commit; `rm .claude/fix-lock`.
6. README contract update; commit.
7. Full gate, then the live drive.

## Risks

- Stricter success rule: outputs that used to return a partly filled `200` now return `502`. That is the user's decision ("never return an empty success").
- The lookahead regex is heuristic. A verse whose text contains `", "word":` or `" "word":` would end early, and one that quotes another field's `"key":` is skipped. Terminators are tried in the order key or `}`, then line end, then end of output, so a value with an inner quote before a newline ends at that line end unless a key or `}` follows it (for example the last value of brace-less output spanning lines). Brace-less output with prose on the same line after the last value is a `502` when that prose has no quote (`"explanation": "C" I hope this helps`), but if the same-line prose ends in a quote (`"explanation": "C" as Jesus said "love"`) the value runs to that last quote and the response is a `200` with the wrong explanation. The same wrong `200` happens when same-line prose is followed by a later line ending in a quote. An empty or whitespace value moves to the key's next occurrence instead of running to a later quote, and a valid JSON object with all three keys that does not qualify is a `502` without the fallback (round 4 fix). Acceptable for a fallback; JSON parsing still wins when the output contains a valid JSON object whose three fields are non-empty strings.
- `JSONResponse` from a Modal `fastapi_endpoint` cannot be deployed here. FastAPI documents returning a `Response` subclass directly from a path operation; the live drive below runs the real FastAPI app locally to cover it.
- Lazy `fastapi` import: an import error would surface only at request time in Modal. The image installs `fastapi[standard]==0.129.2`, so it is present.
- Pickling: `match_verse` still returns a plain dict across `.remote()`, so no custom exception type crosses the Modal boundary.

Not doing: guided or structured decoding, prompt changes, logging rework (issue #5), unifying the Pydantic `422` body, moving `match_verse.remote` off the event loop, the `gpu="any"` on `verse_api`.

## Proof

- Criterion 1, 3, 4: `tests/test_prompt_and_parsing.py`:
  - valid JSON with all keys (existing, kept) and JSON in prose (kept);
  - unescaped quote: `{"reference": "Matthew 22:39", "verse": "He said "love your neighbor" to them", "explanation": "x"}` gives verse `He said "love your neighbor" to them`;
  - escaped quote: a fallback-only input whose verse contains `\"` returns the full verse with a literal `"`;
  - trailing comma: `{"reference": "Matthew 5:44", "verse": "Love your enemies", "explanation": "Hate is not the way.",}` gives all three fields (replaces `test_parse_invalid_json_returns_parsing_failed`);
  - two objects: returns the first object's fields (replaces `test_parse_greedy_brace_match_spans_two_objects`);
  - no JSON at all: plain prose returns the failure dict (replaces `test_parse_no_braces_no_fields_returns_text_as_explanation`);
  - missing key / partial fields (updated `test_parse_json_missing_key_falls_back_to_regex` and `test_parse_fields_without_braces_use_regex`): failure dict; a brace-less input with all three quoted fields succeeds through the regex;
  - JSON with non-string or empty values (`5`, `null`, `""`, `"   "`): not accepted from JSON; failure dict when the fallback finds nothing either.
- Review fixes: `tests/test_parse_review_fixes.py` (frozen alone): A, fields separated by newlines or spaces without commas, an extra key with digits after the last field, and a key named in prose before the object all give the real three fields. B, brace-less fields followed by prose on a new line give the three fields. C, a lone surrogate gives the failure dict from the parser and a `502` from the handler, and every `200` payload in that test renders with Starlette's exact `json.dumps(...).encode("utf-8")` call. D, with an unescaped quote in the value, `\u2019`, `\t`, `\"`, `\/`, `\n` and surrogate pairs are decoded, `\\n` stays a backslash plus `n`, unknown escapes stay as written, and a lone surrogate is a parse failure.
- Delta review: `tests/test_parse_review_fixes_2.py` (frozen alone): an unescaped inner quote before a raw newline keeps the full value, with and without braces; a strict value that contains another field's key falls back to the line end for the same start; prose on a new line after the last value still parses.
- Round 3: `tests/test_parse_review_fixes_3.py` (frozen alone): the last value stops before quoted prose on a later line (three inputs, with and without an unclosed brace); the "Peace" case keeps the full verse with and without braces; prose on a new line after the last value and brace-less output ending at the last value still parse.
- Round 4: `tests/test_parse_review_fixes_4.py` (frozen alone): an empty or whitespace last value followed by a later quote (one-line JSON, multi-line JSON, brace-less) is a failure dict and a `502` through `handle_verse_request`; an empty value moves to the key's next occurrence; a valid JSON object with all three keys that does not qualify skips the fallback.
- Criterion 2, 5, 6: `tests/test_verse_api.py`: `""` and `"   \n"` give `422` and `match` is not called; a good dict gives `200` with exactly the `raw` shape; the `parse_failed` dict, a dict with an empty or non-string field, and a non-dict give `502`; `match` raising `RuntimeError("secret detail")` gives `500` whose body is exactly `{"error": ERROR_INTERNAL, "id": <32 hex>}` and does not contain `secret detail`; a `502` body does not contain the raw output; ids differ between calls; plus the wrapper test.
- Criterion 7: README diff.
- Criterion 8: the gate.
- Live drive (High: changed path plus failure and edge paths): in a throwaway venv outside the repo (`uv venv` in the scratch directory, `uv pip install fastapi==0.129.2 httpx`, plus the dev requirements), build a `fastapi.FastAPI()` app that registers the raw `verse_api` coroutine as `POST /` with `match_verse` replaced by a stub whose `.remote` returns canned outputs (by passing each canned model output through `parse_model_output`), and drive it with `fastapi.testclient.TestClient`. Show status code and body for: valid JSON, trailing comma, unescaped quote, two objects, no JSON (502), `{"text": ""}` and `{"text": "  "}` (422), `{}` (FastAPI 422 `detail`), and a stub that raises (500 with no exception text). No Modal deploy.
