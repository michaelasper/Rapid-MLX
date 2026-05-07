from types import SimpleNamespace

from vllm_mlx.pflash import (
    PFlashConfig,
    compress_request_tokens,
    compress_tokens,
    config_from_args,
    validate_model_support,
)


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


def test_pflash_skips_prompt_integrity_requests():
    tokens = list(range(200))
    config = PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.10,
        skip_when_tools=True,
    )

    result = compress_tokens(tokens, config, requires_prompt_integrity=True)

    assert result.compressed is False
    assert result.reason == "protected_prompt"
    assert result.tokens is tokens


def test_pflash_does_not_exceed_keep_budget_when_blocks_are_large():
    tokens = list(range(100))
    config = PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.03,
        min_keep_tokens=3,
        sink_tokens=0,
        tail_tokens=0,
        block_size=8,
    )

    result = compress_tokens(tokens, config)

    assert result.compressed is True
    assert len(result.tokens) <= 3


def test_pflash_config_validate_rejects_invalid_values():
    invalid_configs = [
        PFlashConfig(mode="unknown"),  # type: ignore[arg-type]
        PFlashConfig(threshold=-1),
        PFlashConfig(keep_ratio=0),
        PFlashConfig(keep_ratio=1.1),
        PFlashConfig(min_keep_tokens=-1),
        PFlashConfig(sink_tokens=-1),
        PFlashConfig(tail_tokens=-1),
        PFlashConfig(block_size=0),
        PFlashConfig(query_window=0),
        PFlashConfig(stride_blocks=-1),
    ]

    for config in invalid_configs:
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid PFlash config: {config!r}")


def test_pflash_config_from_args_validates_and_maps_tool_policy():
    args = SimpleNamespace(
        pflash="auto",
        pflash_threshold=1024,
        pflash_keep_ratio=0.25,
        pflash_min_keep_tokens=128,
        pflash_sink_tokens=16,
        pflash_tail_tokens=64,
        pflash_block_size=32,
        pflash_query_window=128,
        pflash_stride_blocks=4,
        pflash_include_tools=True,
    )

    config = config_from_args(args)

    assert config.mode == "auto"
    assert config.keep_ratio == 0.25
    assert config.skip_when_tools is False


def test_pflash_rejects_multimodal_models_instead_of_silent_noop():
    config = PFlashConfig(mode="auto")

    try:
        validate_model_support(config, model_name="qwen-vl", is_mllm=True)
    except ValueError as exc:
        assert "multimodal" in str(exc)
    else:
        raise AssertionError("expected multimodal PFlash config to be rejected")


def test_pflash_request_helper_reports_compression_metrics():
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
    assert metadata["compressed"] is True
    assert metadata["reason"] == "compressed"
    assert metadata["original_tokens"] == 256
    assert metadata["kept_tokens"] == len(compressed)
    assert metadata["dropped_tokens"] == 256 - len(compressed)
    assert metadata["compression_ratio"] == len(compressed) / 256
