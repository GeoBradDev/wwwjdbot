"""Characterization tests for issue #6: prompt building and model output
parsing move out of match_verse into pure functions, with identical behavior.

These pin the behavior as of commit 6b16340, including the invalid-JSON
"Parsing failed" path that issue #3 will change later."""

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


def test_parse_json_missing_key_falls_back_to_regex(app_module):
    generated = '{"reference": "John 3:16", "verse": "For God so loved the world"}'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("John 3:16", "For God so loved the world", "")


def test_parse_fields_without_braces_use_regex(app_module):
    generated = 'reference: x "reference": "Luke 6:31" and "explanation": "Golden rule."'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("Luke 6:31", "", "Golden rule.")


def test_parse_invalid_json_returns_parsing_failed(app_module):
    # Current behavior (issue #3 will change it): invalid JSON inside braces
    # short-circuits to an error dict even though the regex could match.
    generated = '{"reference": "Matthew 5:44", "verse": "Love your enemies",}'
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == {
        "error": "Parsing failed: Expecting property name enclosed in double"
        " quotes: line 1 column 60 (char 59)",
        "raw_output": generated,
    }


def test_parse_no_braces_no_fields_returns_text_as_explanation(app_module):
    generated = "Love your neighbor as yourself."
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == _ok("", "", generated)


def test_parse_greedy_brace_match_spans_two_objects(app_module):
    generated = (
        '{"reference": "A", "verse": "B", "explanation": "C"}\n'
        '{"reference": "D", "verse": "E", "explanation": "F"}'
    )
    result = app_module.parse_model_output(generated, MODEL, 1.234)
    assert result == {
        "error": "Parsing failed: Extra data: line 2 column 1 (char 53)",
        "raw_output": generated,
    }
