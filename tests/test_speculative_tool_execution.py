import asyncio
import json

from vllm_mlx.api.models import FunctionCall, ToolCall
from vllm_mlx.api.speculative_tools import (
    SpeculativeToolMetrics,
    SpeculativeToolPolicy,
    run_speculative_tool_turn,
)


def _tool_call(name: str, arguments: dict | str) -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(
            name=name,
            arguments=(
                json.dumps(arguments) if isinstance(arguments, dict) else arguments
            ),
        ),
    )


class _Response:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def test_speculative_tool_helpers_are_exported_from_api_package():
    from vllm_mlx.api import (
        SpeculativeToolMetrics as ExportedMetrics,
    )
    from vllm_mlx.api import (
        SpeculativeToolPolicy as ExportedPolicy,
    )
    from vllm_mlx.api import (
        canonical_tool_call_key as exported_key,
    )
    from vllm_mlx.api import (
        run_speculative_tool_turn as exported_runner,
    )

    assert ExportedMetrics is SpeculativeToolMetrics
    assert ExportedPolicy is SpeculativeToolPolicy
    assert exported_key(_tool_call("read_file", {"path": "README.md"}))
    assert exported_runner is run_speculative_tool_turn


async def test_speculative_tool_turn_reuses_matching_safe_tool_future():
    tool_started = asyncio.Event()
    tool_release = asyncio.Event()
    calls = []

    async def main_chat():
        await tool_started.wait()
        return _Response(
            [_tool_call("read_file", '{"limit":100,"path":"README.md"}')]
        )

    async def speculative_chat():
        return _Response([_tool_call("read_file", {"path": "README.md", "limit": 100})])

    async def execute_tool(tool_call):
        calls.append(tool_call.function.name)
        tool_started.set()
        await tool_release.wait()
        return "contents"

    metrics = SpeculativeToolMetrics()
    task = asyncio.create_task(
        run_speculative_tool_turn(
            main_chat=main_chat,
            speculative_chat=speculative_chat,
            execute_tool=execute_tool,
            stateless_tool_names={"read_file"},
            metrics=metrics,
        )
    )

    await asyncio.wait_for(tool_started.wait(), timeout=1)
    tool_release.set()
    result = await task

    assert result.tool_outputs == ["contents"]
    assert result.cache_hits == 1
    assert result.cache_misses == 0
    assert calls == ["read_file"]
    assert metrics.main_tool_calls == 1
    assert metrics.tool_cache_hits == 1
    assert metrics.tool_hit_rate == 1.0
    assert metrics.acceptance_rate == 1.0
    assert metrics.speculations_wasted == 0


async def test_speculative_tool_turn_falls_back_on_cache_miss_and_counts_waste():
    calls = []

    async def main_chat():
        return _Response([_tool_call("read_file", {"path": "README.md"})])

    async def speculative_chat():
        return _Response([_tool_call("search_docs", {"query": "README.md"})])

    async def execute_tool(tool_call):
        calls.append(tool_call.function.name)
        return f"result:{tool_call.function.name}"

    metrics = SpeculativeToolMetrics()
    result = await run_speculative_tool_turn(
        main_chat=main_chat,
        speculative_chat=speculative_chat,
        execute_tool=execute_tool,
        stateless_tool_names={"read_file", "search_docs"},
        metrics=metrics,
    )

    assert result.tool_outputs == ["result:read_file"]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert calls == ["search_docs", "read_file"]
    assert metrics.main_tool_calls == 1
    assert metrics.tool_cache_misses == 1
    assert metrics.tool_hit_rate == 0.0
    assert metrics.speculations_wasted == 1
    assert metrics.acceptance_rate == 0.0


async def test_speculative_tool_turn_matches_raw_qwen_parser_dict_shape():
    async def main_chat():
        return _Response(
            [
                {
                    "id": "call_main",
                    "name": "read_file",
                    "arguments": '{"limit":100,"path":"README.md"}',
                }
            ]
        )

    async def speculative_chat():
        return _Response(
            [
                {
                    "id": "call_spec",
                    "name": "read_file",
                    "arguments": '{"path":"README.md","limit":100}',
                }
            ]
        )

    async def execute_tool(tool_call):
        return "contents"

    metrics = SpeculativeToolMetrics()
    result = await run_speculative_tool_turn(
        main_chat=main_chat,
        speculative_chat=speculative_chat,
        execute_tool=execute_tool,
        stateless_tool_names={"read_file"},
        metrics=metrics,
    )

    assert result.tool_outputs == ["contents"]
    assert result.cache_hits == 1
    assert metrics.acceptance_rate == 1.0


async def test_speculative_tool_turn_does_not_speculate_unallowlisted_tools():
    calls = []

    async def main_chat():
        return _Response([_tool_call("exec", {"cmd": "rm -rf /tmp/nope"})])

    async def speculative_chat():
        return _Response([_tool_call("exec", {"cmd": "rm -rf /tmp/nope"})])

    async def execute_tool(tool_call):
        calls.append(tool_call.function.name)
        return "ok"

    metrics = SpeculativeToolMetrics()
    result = await run_speculative_tool_turn(
        main_chat=main_chat,
        speculative_chat=speculative_chat,
        execute_tool=execute_tool,
        stateless_tool_names={"read_file"},
        metrics=metrics,
    )

    assert result.tool_outputs == ["ok"]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert calls == ["exec"]
    assert metrics.speculations_skipped_unsafe == 1


async def test_speculative_tool_turn_disables_after_low_acceptance_rate():
    speculative_called = False

    async def main_chat():
        return _Response([_tool_call("read_file", {"path": "README.md"})])

    async def speculative_chat():
        nonlocal speculative_called
        speculative_called = True
        return _Response([_tool_call("read_file", {"path": "README.md"})])

    async def execute_tool(tool_call):
        return "contents"

    metrics = SpeculativeToolMetrics(
        main_tool_calls=4,
        tool_cache_misses=4,
        speculations_started=4,
        speculations_wasted=4,
    )
    policy = SpeculativeToolPolicy(
        min_observations_before_disable=4,
        min_acceptance_rate=0.25,
    )
    result = await run_speculative_tool_turn(
        main_chat=main_chat,
        speculative_chat=speculative_chat,
        execute_tool=execute_tool,
        stateless_tool_names={"read_file"},
        metrics=metrics,
        policy=policy,
    )

    assert result.tool_outputs == ["contents"]
    assert speculative_called is False
    assert metrics.disabled_turns == 1
