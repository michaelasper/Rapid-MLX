import types

import pytest

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


def _engine_with_template() -> BatchedEngine:
    engine = BatchedEngine("mlx-community/Qwen3-0.6B-4bit")
    engine._loaded = True
    engine._is_mllm = False
    engine._apply_chat_template = lambda *args, **kwargs: "rendered prompt"
    engine._compute_prefix_boundary = lambda *args, **kwargs: 0
    return engine


@pytest.mark.asyncio
async def test_chat_marks_tool_prompts_as_integrity_required():
    engine = _engine_with_template()
    captured = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    assert captured["has_tools"] is True
    assert captured["requires_prompt_integrity"] is True


@pytest.mark.asyncio
async def test_chat_preserves_route_supplied_integrity_flag_for_schema_prompts():
    engine = _engine_with_template()
    captured = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "return json"}],
        requires_prompt_integrity=True,
    )

    assert captured["has_tools"] is False
    assert captured["requires_prompt_integrity"] is True


@pytest.mark.asyncio
async def test_stream_chat_marks_tool_prompts_as_integrity_required():
    engine = _engine_with_template()
    captured = {}

    async def fake_stream_generate(self, **kwargs):
        captured.update(kwargs)
        yield GenerationOutput(text="ok", finished=True)

    engine.stream_generate = types.MethodType(fake_stream_generate, engine)

    outputs = []
    async for output in engine.stream_chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    ):
        outputs.append(output)

    assert outputs
    assert captured["has_tools"] is True
    assert captured["requires_prompt_integrity"] is True
