"""Prompt building and model output parsing as pure functions.

The build_prompt tests are characterization tests from issue #6 and pin the
prompt as of commit 6b16340. The parse_model_output tests pin the issue #3
behavior: the regex fallback runs whenever JSON parsing fails, and any output
without three non-empty string fields is a parse failure."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# The prompt from the original f-string in match_verse (commit 6b16340),
# formatted with text="X". Each line keeps its 4-space indentation; the dash
# characters are written as escapes so this file has no literal em dash.
EXPECTED_PROMPT_X = "\n".join(
    [
        "<s>[INST]",
        "    Evaluate the following statement from a Christian biblical perspective:",
        "",
        '    "X"',
        "",
        "    Determine whether this statement reflects or contradicts the values"
        " taught by Jesus and core Christian doctrine.",
        "",
        "    If the statement conflicts with biblical teachings, generate a clear"
        " and direct response that:",
        "    1. Identifies a Bible verse that challenges the statement",
        "    2. Presents the full text of that verse",
        "    3. Provides a brief but pointed explanation that highlights the"
        " contradiction or hypocrisy in light of Christian values",
        "",
        "    The tone should be respectful but unflinching\u2014especially when the"
        " statement misrepresents Christian teaching or uses it to justify harm,"
        " exclusion, or hate.",
        "",
        "    Return your response in JSON format with these fields:",
        "    {",
        '      "reference": "The Bible verse reference (e.g., Matthew 5:44)",',
        '      "verse": "The full text of the verse",',
        '      "explanation": "A short explanation (1\u20132 sentences) that clearly'
        ' shows how the verse challenges the statement"',
        "    }",
        "    [/INST]",
    ]
)


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("wwjd_bot_modal", None)
    try:
        yield importlib.import_module("wwjd_bot_modal")
    finally:
        sys.modules.pop("wwjd_bot_modal", None)


def _ok(reference, verse, explanation, generation_time="1.23 seconds"):
    return {
        "reference": reference,
        "verse": verse,
        "explanation": explanation,
        "model": MODEL,
        "generation_time": generation_time,
    }


# build_prompt


def test_build_prompt_matches_original(app_module):
    assert app_module.build_prompt("X") == EXPECTED_PROMPT_X


def test_build_prompt_inserts_text_verbatim_once(app_module):
    text = 'He said "love {everyone}" {{ok}} \\n'
    prompt = app_module.build_prompt(text)
    assert prompt.count(text) == 1
    assert prompt == EXPECTED_PROMPT_X.replace('"X"', f'"{text}"')


def test_model_name_constant(app_module):
    assert app_module.MODEL_NAME == MODEL


# parse_model_output


def test_parse_valid_json_with_all_keys(app_module):
    generated = (
        '{"reference": "Matthew 5:44", "verse": "Love your enemies",'
        ' "explanation": "Hate is not the way.", "extra": "dropped"}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Matthew 5:44", "Love your enemies", "Hate is not the way.")


def test_parse_generation_time_two_decimals(app_module):
    generated = '{"reference": "a", "verse": "b", "explanation": "c"}'
    result = app_module.parse_model_output(generated, MODEL, 12.0)
    assert result["generation_time"] == "12.00 seconds"
    result = app_module.parse_model_output(generated, MODEL, 0.005)
    assert result["generation_time"] == "0.01 seconds"


def test_parse_json_embedded_in_prose(app_module):
    generated = (
        "Here is my answer:\n"
        '{\n  "reference": "John 13:34",\n  "verse": "Love one another",\n'
        '  "explanation": "Jesus commands love."\n}\n'
        "I hope this helps."
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("John 13:34", "Love one another", "Jesus commands love.")


def test_parse_strips_whitespace_around_json_values(app_module):
    generated = '{"reference": " Luke 6:31 ", "verse": "\\nDo to others\\n", "explanation": " x "}'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Luke 6:31", "Do to others", "x")


def _failed(generated):
    return {"error": "parse_failed", "raw_output": generated}


def test_parse_json_missing_key_is_parse_failure(app_module):
    generated = '{"reference": "John 3:16", "verse": "For God so loved the world"}'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _failed(generated)


def test_parse_partial_fields_without_braces_is_parse_failure(app_module):
    generated = 'reference: x "reference": "Luke 6:31" and "explanation": "Golden rule."'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _failed(generated)


def test_parse_all_fields_without_braces_use_regex(app_module):
    generated = (
        'Answer: "reference": "Luke 6:31", "verse": "Do to others as you would'
        ' have them do to you", "explanation": "Golden rule."'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok(
        "Luke 6:31", "Do to others as you would have them do to you", "Golden rule."
    )


def test_parse_trailing_comma_falls_back_to_regex(app_module):
    generated = (
        '{"reference": "Matthew 5:44", "verse": "Love your enemies",'
        ' "explanation": "Hate is not the way.",}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Matthew 5:44", "Love your enemies", "Hate is not the way.")


def test_parse_unescaped_quote_in_verse(app_module):
    generated = (
        '{"reference": "Matthew 22:39", "verse": "He said "love your neighbor"'
        ' to them", "explanation": "x"}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Matthew 22:39", 'He said "love your neighbor" to them', "x")


def test_parse_escaped_quote_in_fallback_value_is_decoded(app_module):
    # The trailing comma makes the JSON invalid, so only the fallback can
    # read this; the old [^"]+ regex stopped at the backslash.
    generated = (
        '{"reference": "Matthew 22:39", "verse": "He said \\"love\\" them",'
        ' "explanation": "Line one\\nline two",}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Matthew 22:39", 'He said "love" them', "Line one\nline two")


def test_parse_two_objects_returns_first(app_module):
    generated = (
        '{"reference": "A", "verse": "B", "explanation": "C"}\n'
        '{"reference": "D", "verse": "E", "explanation": "F"}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("A", "B", "C")


def test_parse_no_json_is_parse_failure(app_module):
    generated = "Love your neighbor as yourself."
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _failed(generated)


@pytest.mark.parametrize(
    "generated",
    [
        '{"reference": 5, "verse": "Love your enemies", "explanation": "x"}',
        '{"reference": "Luke 6:31", "verse": null, "explanation": "x"}',
        '{"reference": "Luke 6:31", "verse": "Do to others", "explanation": ""}',
        '{"reference": "Luke 6:31", "verse": "   ", "explanation": "x"}',
        '{"reference": "Luke 6:31", "verse": "Do to others", "explanation": ["x"]}',
    ],
)
def test_parse_non_string_or_empty_values_are_parse_failure(app_module, generated):
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _failed(generated)
