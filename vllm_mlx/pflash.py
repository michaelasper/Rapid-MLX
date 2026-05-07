# SPDX-License-Identifier: Apache-2.0
"""PFlash-style long-prompt compression for prefill acceleration.

Luce PFlash uses a drafter to identify prompt tokens worth preserving before
the target model runs prefill.  This MLX-native implementation keeps the same
serving-side contract: compress long prompts before BatchGenerator sees them.
The scorer is deterministic and token-statistical so it works without CUDA
kernels and can be tested without a Metal device.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Literal


PFlashMode = Literal["off", "auto", "always"]


@dataclass(frozen=True)
class PFlashConfig:
    """Configuration for PFlash-style prompt compression."""

    mode: PFlashMode = "off"
    threshold: int = 32_768
    keep_ratio: float = 0.10
    min_keep_tokens: int = 2_048
    sink_tokens: int = 256
    tail_tokens: int = 2_048
    block_size: int = 128
    query_window: int = 512
    stride_blocks: int = 8
    skip_when_tools: bool = True


@dataclass(frozen=True)
class PFlashResult:
    """Result of a compression attempt."""

    tokens: list[int]
    compressed: bool
    reason: str
    original_tokens: int
    kept_tokens: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.kept_tokens / self.original_tokens


@dataclass(frozen=True)
class _BlockScore:
    start: int
    end: int
    score: float


def compress_tokens(
    tokens: list[int],
    config: PFlashConfig,
    *,
    has_tools: bool = False,
) -> PFlashResult:
    """Compress a token list according to PFlash settings.

    The output preserves original order, always keeps a leading sink and the
    most recent tail, and fills the remaining budget with middle blocks ranked
    by tail-query overlap plus token rarity.  Repeated filler tends to drop;
    uncommon tokens that reappear near the query are retained.
    """

    n_tokens = len(tokens)
    if config.mode == "off":
        return _unchanged(tokens, "off")
    if config.mode == "auto" and n_tokens < config.threshold:
        return _unchanged(tokens, "threshold")
    if has_tools and config.skip_when_tools:
        return _unchanged(tokens, "tools")
    if n_tokens == 0:
        return _unchanged(tokens, "empty")

    block_size = max(1, config.block_size)
    keep_budget = _keep_budget(n_tokens, config)
    if keep_budget >= n_tokens:
        return _unchanged(tokens, "budget")

    sink_end = min(max(0, config.sink_tokens), n_tokens)
    tail_start = max(sink_end, n_tokens - max(0, config.tail_tokens))

    keep_positions = set(range(sink_end))
    keep_positions.update(range(tail_start, n_tokens))

    remaining_budget = keep_budget - len(keep_positions)
    if remaining_budget > 0:
        scored_blocks = _score_middle_blocks(
            tokens=tokens,
            start=sink_end,
            stop=tail_start,
            block_size=block_size,
            query_window=max(1, config.query_window),
            stride_blocks=max(0, config.stride_blocks),
        )

        selected_tokens = 0
        for block in scored_blocks:
            block_len = block.end - block.start
            if selected_tokens > 0 and selected_tokens + block_len > remaining_budget:
                continue
            keep_positions.update(range(block.start, block.end))
            selected_tokens += block_len
            if selected_tokens >= remaining_budget:
                break

    kept = [tokens[i] for i in sorted(keep_positions)]
    if len(kept) >= n_tokens:
        return _unchanged(tokens, "budget")
    return _changed(tokens, kept, "compressed")


def compress_request_tokens(
    tokens: list[int],
    config: PFlashConfig,
    *,
    has_tools: bool = False,
) -> tuple[list[int], dict[str, int | bool | str]]:
    """Compress request tokens and return compact metadata for logging/state."""

    result = compress_tokens(tokens, config, has_tools=has_tools)
    return result.tokens, {
        "compressed": result.compressed,
        "reason": result.reason,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
    }


def _keep_budget(n_tokens: int, config: PFlashConfig) -> int:
    ratio_budget = ceil(n_tokens * _clamp(config.keep_ratio, 0.0, 1.0))
    return max(1, min(n_tokens, max(config.min_keep_tokens, ratio_budget)))


def _score_middle_blocks(
    *,
    tokens: list[int],
    start: int,
    stop: int,
    block_size: int,
    query_window: int,
    stride_blocks: int,
) -> list[_BlockScore]:
    if start >= stop:
        return []

    counts = Counter(tokens)
    query = tokens[max(0, len(tokens) - query_window) :]
    query_counts = Counter(query)
    span = max(1, stop - start)

    blocks: list[_BlockScore] = []
    block_index = 0
    for block_start in range(start, stop, block_size):
        block_end = min(block_start + block_size, stop)
        block = tokens[block_start:block_end]

        overlap = sum(query_counts.get(token, 0) / counts[token] for token in block)
        rarity = sum(1.0 / counts[token] for token in block) / len(block)
        recency = (block_end - start) / span
        stride_bonus = 0.25 if stride_blocks and block_index % stride_blocks == 0 else 0.0

        score = (4.0 * overlap) + rarity + (0.05 * recency) + stride_bonus
        blocks.append(_BlockScore(block_start, block_end, score))
        block_index += 1

    return sorted(blocks, key=lambda item: (-item.score, item.start))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unchanged(tokens: list[int], reason: str) -> PFlashResult:
    return PFlashResult(
        tokens=tokens,
        compressed=False,
        reason=reason,
        original_tokens=len(tokens),
        kept_tokens=len(tokens),
    )


def _changed(tokens: list[int], kept: list[int], reason: str) -> PFlashResult:
    return PFlashResult(
        tokens=kept,
        compressed=True,
        reason=reason,
        original_tokens=len(tokens),
        kept_tokens=len(kept),
    )
