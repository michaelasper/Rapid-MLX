from dataclasses import dataclass

from vllm_mlx.pflash import PFlashConfig
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class DummyTokenizer:
    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(char) for char in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass
class DummyResponse:
    uid: int
    token: int
    logprobs: object = None
    finish_reason: str | None = None


def _scheduler(pflash_config: PFlashConfig) -> Scheduler:
    return Scheduler(
        model=object(),
        tokenizer=DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            pflash_config=pflash_config,
        ),
    )


def _compressing_config() -> PFlashConfig:
    return PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.25,
        min_keep_tokens=16,
        sink_tokens=4,
        tail_tokens=4,
        block_size=4,
    )


def test_scheduler_keeps_logical_prompt_accounting_after_pflash():
    scheduler = _scheduler(_compressing_config())
    request = Request(
        "req-accounting",
        list(range(128)),
        SamplingParams(max_tokens=4),
        prefix_boundary=32,
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == 128
    assert request.model_prompt_tokens == len(request.prompt_token_ids)
    assert request.model_prompt_tokens < request.num_prompt_tokens
    assert request.pflash_metadata["original_tokens"] == 128
    assert request.pflash_metadata["kept_tokens"] == request.model_prompt_tokens
    assert request.pflash_metadata["dropped_tokens"] == 128 - request.model_prompt_tokens
    assert request.pflash_metadata["prefix_boundary_disabled"] is True

    stats = scheduler.get_stats()["pflash"]
    assert stats["requests"] == 1
    assert stats["compressed_requests"] == 1
    assert stats["original_tokens"] == 128
    assert stats["kept_tokens"] == request.model_prompt_tokens
    assert stats["dropped_tokens"] == 128 - request.model_prompt_tokens
    assert stats["prefix_boundary_disabled"] == 1
    assert stats["effective_keep_ratio"] == request.model_prompt_tokens / 128


def test_scheduler_skips_prompt_integrity_requests():
    scheduler = _scheduler(_compressing_config())
    request = Request(
        "req-protected",
        list(range(128)),
        SamplingParams(max_tokens=4),
        requires_prompt_integrity=True,
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == 128
    assert request.model_prompt_tokens == 128
    assert request.prompt_token_ids == list(range(128))
    assert request.pflash_metadata["compressed"] is False
    assert request.pflash_metadata["reason"] == "protected_prompt"

    stats = scheduler.get_stats()["pflash"]
    assert stats["requests"] == 1
    assert stats["skipped_requests"] == 1
    assert stats["skipped_by_reason"]["protected_prompt"] == 1


def test_scheduler_outputs_logical_prompt_tokens_after_pflash():
    scheduler = _scheduler(_compressing_config())
    request = Request(
        "req-output",
        list(range(128)),
        SamplingParams(max_tokens=4),
    )
    scheduler.add_request(request)
    scheduler.waiting.clear()
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[7] = request.request_id

    outputs, _finished = scheduler._process_batch_responses(
        [DummyResponse(uid=7, token=65, finish_reason=None)]
    )

    assert outputs[0].prompt_tokens == 128
    assert outputs[0].usage["prompt_tokens"] == 128
    assert request.pflash_metadata["ttft_s"] >= 0
    stats = scheduler.get_stats()["pflash"]
    assert stats["ttft_observations"] == 1
    assert stats["compressed_ttft_observations"] == 1
    assert stats["acceptance_rate"] is None


def test_running_status_exposes_model_input_and_pflash_metrics():
    scheduler = _scheduler(_compressing_config())
    request = Request(
        "req-status",
        list(range(128)),
        SamplingParams(max_tokens=4),
    )
    scheduler.add_request(request)
    scheduler.waiting.clear()
    scheduler.running[request.request_id] = request

    info = scheduler.get_running_requests_info()[0]

    assert info["prompt_tokens"] == 128
    assert info["model_prompt_tokens"] == request.model_prompt_tokens
    assert info["pflash"]["compressed"] is True
    assert info["pflash"]["dropped_tokens"] == 128 - request.model_prompt_tokens
