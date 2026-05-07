from vllm_mlx.pflash import PFlashConfig, compress_request_tokens, compress_tokens


def test_pflash_compresses_long_prompt_preserving_edges_and_order():
    tokens = list(range(20)) + [100] * 64 + list(range(200, 264)) + list(range(900, 920))
    config = PFlashConfig(
        mode="auto",
        threshold=64,
        keep_ratio=0.35,
        min_keep_tokens=32,
        sink_tokens=8,
        tail_tokens=8,
        block_size=8,
        query_window=8,
        stride_blocks=0,
    )

    result = compress_tokens(tokens, config)

    assert result.compressed is True
    assert result.original_tokens == len(tokens)
    assert len(result.tokens) < len(tokens)
    assert result.tokens[:8] == tokens[:8]
    assert result.tokens[-8:] == tokens[-8:]
    assert result.tokens == sorted(result.tokens, key=tokens.index)


def test_pflash_keeps_query_overlap_blocks_over_repetitive_filler():
    prefix = list(range(10))
    filler = [7] * 96
    needle_block = [501, 502, 503, 504, 900, 901, 902, 903]
    more_filler = [8] * 96
    tail = [900, 901, 902, 903, 1000, 1001, 1002, 1003]
    tokens = prefix + filler + needle_block + more_filler + tail
    config = PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.20,
        min_keep_tokens=24,
        sink_tokens=4,
        tail_tokens=8,
        block_size=8,
        query_window=8,
        stride_blocks=0,
    )

    result = compress_tokens(tokens, config)

    assert result.compressed is True
    assert all(token in result.tokens for token in needle_block)


def test_pflash_skips_tool_prompts_by_default():
    tokens = list(range(200))
    config = PFlashConfig(
        mode="auto",
        threshold=10,
        keep_ratio=0.10,
        skip_when_tools=True,
    )

    result = compress_tokens(tokens, config, has_tools=True)

    assert result.compressed is False
    assert result.reason == "tools"
    assert result.tokens is tokens


def test_pflash_request_helper_reports_original_and_compressed_counts():
    tokens = list(range(256))
    config = PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.25,
        min_keep_tokens=32,
        sink_tokens=8,
        tail_tokens=16,
        block_size=8,
    )

    compressed, metadata = compress_request_tokens(tokens, config, has_tools=False)

    assert len(compressed) < len(tokens)
    assert metadata == {
        "compressed": True,
        "reason": "compressed",
        "original_tokens": 256,
        "kept_tokens": len(compressed),
    }
