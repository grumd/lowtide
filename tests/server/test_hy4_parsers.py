"""Hy4-flash reasoning + tool-call parsers (``hy_v4``) on raw model output.

The RAW_* strings are verbatim ``/v1/completions`` output of the NVFP4 FTW (TP=2, temperature 0)
for prompts rendered with the checkpoint's chat template; the think / tool tags are added
tokens that are not special, so they survive detokenization. Each case is fed whole, one tag
or text run per chunk (how the detokenizer delivers single-token tags), and one character
per chunk.
"""

from __future__ import annotations

import json
import re

import pytest

from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.reasoning_parser import ReasoningParser

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

CASES = [
    {
        "id": 'no_think_plain',
        "thinking": False,
        "raw": (
            'Red, blue, yellow'
        ),
        "reasoning": '',
        "content": 'Red, blue, yellow',
        "calls": [],
    },
    {
        "id": 'think',
        "thinking": True,
        "raw": (
            'The user asks: "What is 17 * 23? Answer with just the number." So we need to compute 17*'
            '23. 17*20 = 340, 17*3 = 51, sum = 391. So answer is 391. They want just the number. So o'
            'utput "391". No extra text. Ensure no spaces or punctuation. Just the number.</think:ope'
            'nsource>391'
        ),
        "reasoning": 'The user asks: "What is 17 * 23? Answer with just the number." So we need to compute 17*23. 17*20 = 340, 17*3 = 51, sum = 391. So answer is 391. They want just the number. So output "391". No extra text. Ensure no spaces or punctuation. Just the number.',
        "content": '391',
        "calls": [],
    },
    {
        "id": 'think_tool',
        "thinking": True,
        "raw": (
            "</think:opensource>I'll check the current weather in Paris for you.<tool_calls:opensourc"
            'e><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_val'
            'ue:opensource>Paris</arg_value:opensource><arg_key:opensource>unit</arg_key:opensource><'
            'arg_value:opensource>celsius</arg_value:opensource></tool_call:opensource></tool_calls:o'
            'pensource>'
        ),
        "reasoning": '',
        "content": "I'll check the current weather in Paris for you.",
        "calls": [('get_weather', {'city': 'Paris', 'unit': 'celsius'})],
    },
    {
        "id": 'no_think_tool',
        "thinking": False,
        "raw": (
            '<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_ke'
            'y:opensource><arg_value:opensource>Paris</arg_value:opensource><arg_key:opensource>unit<'
            '/arg_key:opensource><arg_value:opensource>celsius</arg_value:opensource></tool_call:open'
            'source></tool_calls:opensource>'
        ),
        "reasoning": '',
        "content": '',
        "calls": [('get_weather', {'city': 'Paris', 'unit': 'celsius'})],
    },
]


def _chunkings(raw: str):
    yield "whole", [raw]
    yield "tags", [p for p in re.split(r"(<[^<>]*?>)", raw) if p]
    yield "chars", list(raw)


def _stream(raw_chunks, force_reasoning: bool):
    reasoning = ReasoningParser("hy_v4", force_reasoning=force_reasoning)
    tools = FunctionCallParser(TOOLS, "hy_v4")
    text, content, calls = "", "", []
    for chunk in raw_chunks + [None]:
        r, c = reasoning.parse_stream_chunk(chunk) if chunk is not None else reasoning.flush()
        text += r
        normal, items = tools.parse_stream_chunk(c)
        content += normal
        calls += items
    content += tools.finish_stream()
    names = [c.name for c in calls if c.name]
    args = "".join(c.parameters for c in calls)
    return text, content, names, args


def _one_shot(raw: str, force_reasoning: bool):
    text, rest = ReasoningParser("hy_v4", force_reasoning=force_reasoning).parse_non_stream(raw)
    parsed = FunctionCallParser(TOOLS, "hy_v4").parse_non_stream(rest)
    return text, parsed.normal_text, parsed.calls


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_one_shot(case):
    reasoning, content, calls = _one_shot(case["raw"], case["thinking"])
    assert reasoning.strip() == case["reasoning"].strip()
    assert content.strip() == case["content"].strip()
    assert [(c.name, json.loads(c.parameters)) for c in calls] == case["calls"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_streaming_matches_one_shot(case):
    for how, chunks in _chunkings(case["raw"]):
        reasoning, content, names, args = _stream(chunks, case["thinking"])
        assert reasoning.strip() == case["reasoning"].strip(), how
        assert content.strip() == case["content"].strip(), how
        assert names == [name for name, _ in case["calls"]], how
        if case["calls"]:
            assert json.loads(args) == case["calls"][0][1], how
        for tag in (":opensource>",):
            assert tag not in reasoning + content, how


def test_block_markers_never_reach_content():
    raw = (
        "Checking.</think:opensource>\n\n<tool_calls:opensource><tool_call:opensource>get_weather"
        "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Oslo</arg_value:opensource>"
        "</tool_call:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Rome</arg_value:opensource></tool_call:opensource></tool_calls:opensource>"
    )
    reasoning, content, calls = _one_shot(raw, True)
    assert reasoning == "Checking." and content.strip() == ""
    assert [json.loads(c.parameters)["city"] for c in calls] == ["Oslo", "Rome"]
    for how, chunks in _chunkings(raw):
        _r, content, names, _args = _stream(chunks, True)
        assert content.strip() == "" and names == ["get_weather", "get_weather"], how


def test_text_after_the_block_is_kept():
    raw = (
        "<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city"
        "</arg_key:opensource><arg_value:opensource>Oslo</arg_value:opensource></tool_call:opensource>"
        "</tool_calls:opensource>done"
    )
    parsed = FunctionCallParser(TOOLS, "hy_v4").parse_non_stream(raw)
    assert parsed.normal_text == "done" and len(parsed.calls) == 1


def test_a_lone_angle_bracket_is_released_at_the_end():
    _r, content, names, _a = _stream(["a <", " b <"], False)
    assert content == "a < b <" and names == []


@pytest.mark.parametrize("kwargs, thinking", [
    ({}, True),
    ({"reasoning_effort": "max"}, True),
    ({"reasoning_effort": "no_think"}, False),
    ({"enable_thinking": False, "thinking_mode": "disabled"}, False),  # reasoning_effort "none"
])
def test_reasoning_starts_open_unless_thinking_is_off(kwargs, thinking):
    """The template opens <think:opensource> at the generation prompt in every mode but
    no_think, so the first output token is already reasoning."""
    from types import SimpleNamespace

    from freetoken.server.generation import GenSpec, _make_reasoning_parser

    state = SimpleNamespace(config=SimpleNamespace(reasoning_parser="hy_v4"))
    spec = GenSpec(messages=[], sampling_params=None, chat_template_kwargs=kwargs)
    reasoning, content = _make_reasoning_parser(spec, state).parse_non_stream("391")
    assert (reasoning, content) == (("391", "") if thinking else ("", "391"))
