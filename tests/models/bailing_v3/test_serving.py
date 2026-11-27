"""Ling-3.2 serving glue: the parsers `auto` picks read what the checkpoint's own chat template writes.

The template renders an assistant tool call as GLM-4.7's ``<tool_call>name`` + ``<arg_key>`` / ``<arg_value>``
pairs with a newline before every part, and opens ``<think>`` at the generation prompt when thinking is on.
"""

from __future__ import annotations

import json
import os

import pytest

FTW = "/run/media/grumd/2TB/models/ftw/ling32-air-nvfp4a16"
needs_checkpoint = pytest.mark.skipif(not os.path.isdir(FTW), reason="needs the Ling-3.2 checkpoint's tokenizer and template")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "changes": {"type": "array"}},
            },
        },
    }
]
ARGS = {"path": "/tmp/a b.py", "changes": [{"line": 3, "text": "x = 1\ny = 2"}]}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(FTW, trust_remote_code=True)


def test_auto_selects_the_glm47_tool_parser_and_the_qwen3_reasoning_parser():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class Config:
        def to_dict(self):
            return {"architectures": ["BailingMoeV3VLForConditionalGeneration"], "model_type": "bailing_moe_v3_vl"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: Config()):
        args, _ = parse_args(["--model", "/models/anon"])
    assert (args.tool_call_parser, args.reasoning_parser) == ("glm47", "qwen3")


def _rendered_call(tokenizer) -> str:
    """The assistant turn the template renders for one tool call, after its empty think block."""
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "edit", "arguments": ARGS}}]},
    ]
    text = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    return text.rsplit("<think></think>", 1)[1].removesuffix("<|role_end|>")


@needs_checkpoint
@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
def test_the_template_s_tool_call_parses_back_streamed_and_whole(tokenizer, chunk):
    from freetoken.server.function_call_parser import FunctionCallParser

    text = _rendered_call(tokenizer)
    assert text.startswith("<tool_call>edit\n<arg_key>path</arg_key>\n<arg_value>"), text
    parser = FunctionCallParser(TOOLS, tool_call_parser="glm47")
    if chunk == 0:
        result = parser.parse_non_stream(text)
        normal, calls = result.normal_text, result.calls
    else:
        normal, calls = "", []
        for i in range(0, len(text), chunk):
            n, c = parser.parse_stream_chunk(text[i : i + chunk])
            normal += n
            calls += c
        normal += parser.finish_stream()
    assert normal.strip() == ""
    names = [c.name for c in calls if c.name]
    args = "".join(c.parameters or "" for c in calls)
    assert names == ["edit"]
    assert json.loads(args) == ARGS


@needs_checkpoint
def test_thinking_output_splits_at_the_closing_tag(tokenizer):
    from freetoken.server.reasoning_parser import ReasoningParser

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert prompt.endswith("<role>ASSISTANT</role>\n<think>")  # the server then parses with force_reasoning
    reasoning, content = ReasoningParser("qwen3", force_reasoning=True).parse_non_stream("Greet back.\n</think>\n\nHello!")
    assert reasoning.strip() == "Greet back." and content.strip() == "Hello!"
