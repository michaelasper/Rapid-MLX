import pytest

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import _run_chat_with_tool_schema_fallback


class DummyRawRequest:
    async def is_disconnected(self):
        return False


class DummyEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


@pytest.mark.asyncio
async def test_tool_schema_fallback_retries_invalid_compact_tool_args():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "search"}],
        tools=TOOLS,
    )
    engine = DummyEngine(
        [
            GenerationOutput(
                text='<tool_call>{"name":"search_docs","arguments":{}}</tool_call>',
                prompt_tokens=100,
                completion_tokens=5,
            ),
            GenerationOutput(
                text='<tool_call>{"name":"search_docs","arguments":{"query":"mlx"}}</tool_call>',
                prompt_tokens=180,
                completion_tokens=5,
            ),
        ]
    )
    full_tools = TOOLS
    compact_tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search docs.",
                "parameters": TOOLS[0]["function"]["parameters"],
            },
        }
    ]

    output, cleaned, tool_calls, validation = await _run_chat_with_tool_schema_fallback(
        engine=engine,
        messages=request.messages,
        request=request,
        chat_kwargs={"tools": compact_tools, "max_tokens": 8},
        raw_request=DummyRawRequest(),
        timeout=30,
        full_tools=full_tools,
        compact_tools_used=True,
    )

    assert '"query":"mlx"' in output.text
    assert len(engine.calls) == 2
    assert engine.calls[0]["tools"] == compact_tools
    assert engine.calls[1]["tools"] == full_tools
    assert validation.ok is True
    assert tool_calls[0].function.name == "search_docs"
    assert '"query"' in tool_calls[0].function.arguments


@pytest.mark.asyncio
async def test_tool_schema_fallback_does_not_retry_plain_no_tool_answer():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        tools=TOOLS,
    )
    engine = DummyEngine(
        [GenerationOutput(text="No tool needed.", prompt_tokens=20, completion_tokens=4)]
    )

    output, cleaned, tool_calls, validation = await _run_chat_with_tool_schema_fallback(
        engine=engine,
        messages=request.messages,
        request=request,
        chat_kwargs={"tools": TOOLS, "max_tokens": 8},
        raw_request=DummyRawRequest(),
        timeout=30,
        full_tools=TOOLS,
        compact_tools_used=True,
    )

    assert output.text == "No tool needed."
    assert len(engine.calls) == 1
    assert cleaned == "No tool needed."
    assert tool_calls is None
    assert validation.ok is True
