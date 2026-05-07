# SPDX-License-Identifier: Apache-2.0
"""Client-side speculative tool execution helpers.

This module intentionally stays outside the model scheduler. It overlaps a
single fast speculative tool prediction with the main model request, then
reuses the speculative tool future only on an exact canonical tool-call match.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .tool_calling import canonical_tool_call_key

ChatCallable = Callable[[], Awaitable[Any]]
ToolExecutor = Callable[[Any], Any | Awaitable[Any]]


@dataclass
class SpeculativeToolPolicy:
    """Conservative policy for opt-in client-side tool speculation."""

    min_observations_before_disable: int = 8
    min_acceptance_rate: float = 0.05
    auto_disable: bool = True

    def should_speculate(self, metrics: SpeculativeToolMetrics) -> bool:
        if not self.auto_disable:
            return True
        if metrics.speculations_started < self.min_observations_before_disable:
            return True
        acceptance_rate = metrics.acceptance_rate
        return acceptance_rate is None or acceptance_rate >= self.min_acceptance_rate


@dataclass
class SpeculativeToolMetrics:
    """Runtime guardrail metrics for speculative tool execution."""

    main_tool_calls: int = 0
    tool_cache_hits: int = 0
    tool_cache_misses: int = 0
    speculations_started: int = 0
    speculations_completed: int = 0
    speculations_reused: int = 0
    speculations_wasted: int = 0
    speculations_skipped_unsafe: int = 0
    speculation_errors: int = 0
    disabled_turns: int = 0
    total_overlap_seconds: float = 0.0

    @property
    def tool_hit_rate(self) -> float | None:
        if self.main_tool_calls == 0:
            return None
        return self.tool_cache_hits / self.main_tool_calls

    @property
    def acceptance_rate(self) -> float | None:
        if self.speculations_started == 0:
            return None
        return self.speculations_reused / self.speculations_started

    @property
    def waste_rate(self) -> float | None:
        if self.speculations_started == 0:
            return None
        return self.speculations_wasted / self.speculations_started

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_tool_calls": self.main_tool_calls,
            "tool_cache_hits": self.tool_cache_hits,
            "tool_cache_misses": self.tool_cache_misses,
            "tool_hit_rate": self.tool_hit_rate,
            "acceptance_rate": self.acceptance_rate,
            "speculations_started": self.speculations_started,
            "speculations_completed": self.speculations_completed,
            "speculations_reused": self.speculations_reused,
            "speculations_wasted": self.speculations_wasted,
            "waste_rate": self.waste_rate,
            "speculations_skipped_unsafe": self.speculations_skipped_unsafe,
            "speculation_errors": self.speculation_errors,
            "disabled_turns": self.disabled_turns,
            "total_overlap_seconds": self.total_overlap_seconds,
        }


@dataclass
class SpeculativeToolTurnResult:
    """Result from one main-model turn plus optional tool execution."""

    main_response: Any
    speculative_response: Any | None
    tool_calls: list[Any]
    tool_outputs: list[Any]
    cache_hits: int = 0
    cache_misses: int = 0
    speculation_started: bool = False


@dataclass
class _SpeculativeEntry:
    tool_call: Any
    started_at: float
    task: asyncio.Task | None = None
    completed_at: float | None = None
    reused: bool = False
    consumed: bool = False


def _get_attr_or_item(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_tool_calls(response: Any) -> list[Any]:
    """Extract OpenAI-style tool calls from dicts, Pydantic models, or shims."""

    direct = _get_attr_or_item(response, "tool_calls")
    if direct:
        return list(direct)

    choices = _get_attr_or_item(response, "choices")
    if not choices:
        return []

    tool_calls: list[Any] = []
    for choice in choices:
        message = _get_attr_or_item(choice, "message")
        if not message:
            delta = _get_attr_or_item(choice, "delta")
            message = delta
        calls = _get_attr_or_item(message, "tool_calls") if message else None
        if calls:
            tool_calls.extend(calls)
    return tool_calls


def _tool_name(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict) and "function" not in tool_call:
        name = tool_call.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    function = _get_attr_or_item(tool_call, "function")
    name = _get_attr_or_item(function, "name") if function else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _execute_speculative_tool(
    entry: _SpeculativeEntry,
    execute_tool: ToolExecutor,
    metrics: SpeculativeToolMetrics,
) -> Any:
    try:
        return await _maybe_await(execute_tool(entry.tool_call))
    finally:
        entry.completed_at = time.perf_counter()
        metrics.speculations_completed += 1


def _consume_task_result(
    task: asyncio.Task,
    metrics: SpeculativeToolMetrics,
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        metrics.speculation_errors += 1


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def run_speculative_tool_turn(
    *,
    main_chat: ChatCallable,
    speculative_chat: ChatCallable | None,
    execute_tool: ToolExecutor,
    stateless_tool_names: Iterable[str],
    metrics: SpeculativeToolMetrics | None = None,
    policy: SpeculativeToolPolicy | None = None,
) -> SpeculativeToolTurnResult:
    """Run one client-side speculative tool turn.

    The speculative path uses one low-temperature/greedy speculative call by
    construction. Only allowlisted stateless tools are executed before the main
    model confirms the exact same canonical tool call.
    """

    metrics = metrics or SpeculativeToolMetrics()
    policy = policy or SpeculativeToolPolicy()
    safe_tools = {name for name in stateless_tool_names}
    spec_cache: dict[str, _SpeculativeEntry] = {}
    speculative_response: Any | None = None

    main_task = asyncio.create_task(main_chat())
    spec_task: asyncio.Task | None = None

    async def _populate_speculation() -> None:
        nonlocal speculative_response
        if speculative_chat is None:
            return
        try:
            speculative_response = await speculative_chat()
        except Exception:
            metrics.speculation_errors += 1
            return

        for tool_call in _extract_tool_calls(speculative_response):
            name = _tool_name(tool_call)
            if name not in safe_tools:
                metrics.speculations_skipped_unsafe += 1
                continue

            key = canonical_tool_call_key(tool_call)
            if key is None or key in spec_cache:
                continue

            entry = _SpeculativeEntry(tool_call=tool_call, started_at=time.perf_counter())
            entry.task = asyncio.create_task(
                _execute_speculative_tool(entry, execute_tool, metrics)
            )
            spec_cache[key] = entry
            metrics.speculations_started += 1

    if speculative_chat is not None and policy.should_speculate(metrics):
        spec_task = asyncio.create_task(_populate_speculation())
    elif speculative_chat is not None:
        metrics.disabled_turns += 1

    # Give immediate speculative calls one event-loop tick to launch safe tools.
    await asyncio.sleep(0)
    if spec_task is not None and spec_task.done():
        # If the speculative response was immediate, it may have just spawned
        # the tool future; let that future start before an immediate main
        # response is processed as a miss. This is still a zero-time yield.
        await asyncio.sleep(0)
    main_response = await main_task

    tool_calls = _extract_tool_calls(main_response)
    tool_outputs: list[Any] = []
    cache_hits = 0
    cache_misses = 0

    for tool_call in tool_calls:
        metrics.main_tool_calls += 1
        key = canonical_tool_call_key(tool_call)
        entry = spec_cache.get(key) if key is not None else None

        if entry is not None and entry.task is not None:
            completed_or_now = entry.completed_at or time.perf_counter()
            metrics.total_overlap_seconds += max(
                0.0,
                min(time.perf_counter(), completed_or_now) - entry.started_at,
            )
            try:
                output = await entry.task
                entry.consumed = True
                entry.reused = True
                cache_hits += 1
                metrics.tool_cache_hits += 1
                metrics.speculations_reused += 1
                tool_outputs.append(output)
                continue
            except Exception:
                entry.consumed = True
                metrics.speculation_errors += 1

        cache_misses += 1
        metrics.tool_cache_misses += 1
        tool_outputs.append(await _maybe_await(execute_tool(tool_call)))

    for entry in spec_cache.values():
        if entry.reused or entry.consumed:
            continue
        metrics.speculations_wasted += 1
        if entry.task is None:
            continue
        if entry.task.done():
            _consume_task_result(entry.task, metrics)
        else:
            await _cancel_task(entry.task)

    if spec_task is not None and not spec_task.done():
        await _cancel_task(spec_task)

    return SpeculativeToolTurnResult(
        main_response=main_response,
        speculative_response=speculative_response,
        tool_calls=tool_calls,
        tool_outputs=tool_outputs,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        speculation_started=bool(spec_cache),
    )
