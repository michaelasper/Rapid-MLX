from vllm_mlx.cli import _build_benchmark_context


def test_pflash_benchmark_context_can_generate_long_prompts():
    context = _build_benchmark_context(256)

    assert "Reference context" in context
    assert len(context.split()) >= 200


def test_pflash_benchmark_context_is_empty_when_disabled():
    assert _build_benchmark_context(0) == ""
