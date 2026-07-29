from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from replay_kfac_ewc import (
    ReplayConfig,
    ReplayRecord,
    generate_replay,
    load_replay,
    merge_replay,
)


class TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0
    all_special_ids = [0]

    def __len__(self) -> int:
        return 11

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 + (ord(character) % 10) for character in text]


class TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 6)
        self.lm_head = torch.nn.Linear(6, 11, bias=False)
        self.generation_config = SimpleNamespace(bos_token_id=0, eos_token_id=0)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))

    def generate(self, input_ids, min_new_tokens, max_new_tokens, **kwargs):
        del kwargs
        assert min_new_tokens == max_new_tokens
        sampled = torch.randint(
            1,
            11,
            (input_ids.shape[0], min_new_tokens),
            device=input_ids.device,
        )
        return torch.cat((input_ids, sampled), dim=1)


def _record(seq_id: int, tokens: list[int]) -> ReplayRecord:
    return ReplayRecord(
        seq_id=seq_id,
        sampling_seed=100 + seq_id,
        seeding_mode="bos",
        prefix_ids=tokens[:1],
        token_ids=tokens,
        prefix_length=1,
        length=len(tokens),
        mean_logprob=-1.0,
        has_repeat_loop=False,
        unique_token_count=len(set(tokens)),
    )


def test_generate_is_exactly_resumable(tmp_path):
    model = TinyGenerator()
    tokenizer = TinyTokenizer()
    output = tmp_path / "shard.jsonl"
    config = ReplayConfig(
        num_sequences=4,
        min_new_tokens=3,
        max_new_tokens=3,
        seed=17,
        seeding_mode="mixed",
        fsync_every=1,
    )
    first = generate_replay(model, tokenizer, output, config)
    second = generate_replay(model, tokenizer, output, config)
    assert [record.token_ids for record in first] == [
        record.token_ids for record in second
    ]
    assert len(load_replay(output)) == 4
    assert output.with_suffix(".jsonl.manifest.json").exists()
    with pytest.raises(ValueError, match="sampling settings"):
        generate_replay(
            model,
            tokenizer,
            output,
            ReplayConfig(
                num_sequences=4,
                min_new_tokens=3,
                max_new_tokens=3,
                seed=18,
                seeding_mode="mixed",
            ),
        )


def test_prompt_generation_records_prompt_identity(tmp_path):
    records = generate_replay(
        TinyGenerator(),
        TinyTokenizer(),
        tmp_path / "prompted.jsonl",
        ReplayConfig(
            num_sequences=2,
            min_new_tokens=2,
            max_new_tokens=2,
            seeding_mode="prompt",
        ),
        prompts=[
            {"id": "alpha", "prompt": "ab"},
            {"id": "beta", "prompt_ids": [3, 4]},
        ],
    )
    assert records[0].metadata["prompt_id"] == "alpha"
    assert records[1].prefix_ids == [3, 4]


def test_merge_assigns_stable_splits_and_deduplicates_content(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    from replay_kfac_ewc._io import atomic_jsonl

    atomic_jsonl(left, [_record(0, [0, 1, 2]).to_dict()])
    atomic_jsonl(
        right,
        [
            _record(1, [0, 3, 4]).to_dict(),
            _record(2, [0, 1, 2]).to_dict(),
        ],
    )
    output = tmp_path / "library.jsonl"
    merged = merge_replay(
        [right, left],
        output,
        heldout_fraction=0.5,
        split_seed=9,
    )
    assert [record.seq_id for record in merged] == [0, 1]
    first_splits = [record.split for record in merged]
    repeated = merge_replay(
        [left, right],
        tmp_path / "library2.jsonl",
        heldout_fraction=0.5,
        split_seed=9,
    )
    assert [record.split for record in repeated] == first_splits
