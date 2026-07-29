from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from replay_kfac_ewc import (
    FactorBundle,
    FactorEstimationConfig,
    FactorPair,
    KFACEWC,
    LoRADelta,
    ReplayRecord,
    compress_psd,
    dense_accumulator_bytes,
    estimate_factors,
    resolve_linear_modules,
)


def _psd(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(dimension, dimension + 2, generator=generator)
    return source @ source.T


def _bundle(a: torch.Tensor, g: torch.Tensor) -> FactorBundle:
    return FactorBundle(
        factors={
            "block": FactorPair(
                activation=compress_psd(
                    a, energy_threshold=1.0, rank_cap=a.shape[0]
                ),
                gradient=compress_psd(
                    g, energy_threshold=1.0, rank_cap=g.shape[0]
                ),
            )
        },
        position_count=10,
    )


def test_full_delta_penalty_matches_dense_quadratic_and_backpropagates():
    torch.manual_seed(0)
    a = _psd(5, 1)
    g = _psd(4, 2)
    delta = torch.randn(4, 5, requires_grad=True)
    expected = 0.5 * torch.trace(g @ delta @ a @ delta.T)
    actual = KFACEWC(_bundle(a, g)).penalty_from_deltas({"block": delta})
    assert float(actual.detach()) == pytest.approx(
        float(expected), rel=2e-4, abs=2e-4
    )
    actual.backward()
    assert delta.grad is not None
    assert float(delta.grad.norm()) > 0


def test_lora_rxr_penalty_matches_materialized_delta_and_reaches_a_and_b():
    torch.manual_seed(3)
    a = _psd(6, 4)
    g = _psd(5, 5)
    lora_a = torch.randn(2, 6, requires_grad=True)
    lora_b = torch.randn(5, 2, requires_grad=True)
    scale = 2.0
    anchor = KFACEWC(_bundle(a, g))
    low_rank = anchor.penalty_from_lora(
        {"block": LoRADelta(A=lora_a, B=lora_b, scale=scale)}
    )
    dense = anchor.penalty_from_deltas(
        {"block": scale * lora_b @ lora_a}
    )
    assert float(low_rank.detach()) == pytest.approx(
        float(dense.detach()), rel=2e-4, abs=2e-4
    )
    low_rank.backward()
    assert lora_a.grad is not None and float(lora_a.grad.norm()) > 0
    assert lora_b.grad is not None and float(lora_b.grad.norm()) > 0


def test_lora_penalty_accepts_low_precision_trainable_weights():
    a = _psd(4, 8)
    g = _psd(3, 9)
    lora_a = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    lora_b = torch.randn(3, 2, dtype=torch.bfloat16, requires_grad=True)
    penalty = KFACEWC(_bundle(a, g)).penalty_from_lora(
        {"block": LoRADelta(A=lora_a, B=lora_b, scale=1.0)}
    )
    assert penalty.dtype == torch.float32
    penalty.backward()
    assert lora_a.grad is not None
    assert lora_b.grad is not None


def test_peft_adapter_extraction_uses_suffix_and_adapter_scaling():
    a = _psd(4, 10)
    g = _psd(3, 11)

    class FakePeftLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"named": torch.nn.Linear(4, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"named": torch.nn.Linear(2, 3, bias=False)}
            )
            self.scaling = {"named": 1.5}

    model = torch.nn.Module()
    model.base_model = torch.nn.Module()
    model.base_model.model = torch.nn.Module()
    model.base_model.model.block = FakePeftLinear()
    anchor = KFACEWC(_bundle(a, g))
    updates = anchor.lora_updates_from_peft(model, adapter_name="named")
    assert updates["block"].scale == 1.5
    penalty = anchor.penalty_from_peft(model, adapter_name="named")
    penalty.backward()
    assert model.base_model.model.block.lora_A["named"].weight.grad is not None
    assert model.base_model.model.block.lora_B["named"].weight.grad is not None


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 5)
        self.block = torch.nn.Linear(5, 4, bias=False)
        self.output = torch.nn.Linear(4, 13, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.tanh(self.block(self.embedding(input_ids)))
        return SimpleNamespace(logits=self.output(hidden))


def _replay(seq_id: int, tokens: list[int], split: str = "train") -> ReplayRecord:
    return ReplayRecord(
        seq_id=seq_id,
        sampling_seed=seq_id,
        seeding_mode="bos",
        prefix_ids=tokens[:1],
        token_ids=tokens,
        prefix_length=1,
        length=len(tokens),
        mean_logprob=-1,
        has_repeat_loop=False,
        unique_token_count=len(set(tokens)),
        split=split,
    )


def test_factor_estimator_runs_on_generic_causal_lm_and_roundtrips(tmp_path):
    model = TinyCausalLM()
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    modules = resolve_linear_modules(model, ("block",))
    assert list(modules) == ["block"]
    assert dense_accumulator_bytes(modules) == 4 * (5**2 + 4**2)
    records = [
        _replay(0, [0, 1, 2, 3]),
        _replay(1, [0, 3, 4, 5, 6]),
        _replay(2, [0, 7, 8], split="heldout"),
    ]
    bundle = estimate_factors(
        model,
        records,
        FactorEstimationConfig(
            target_modules=("block",),
            batch_size=2,
            placement="cpu",
            energy_threshold=1.0,
            rank_cap=8,
            eigendecomposition_dtype="float64",
        ),
        pad_token_id=0,
    )
    assert bundle.position_count == 7
    assert bundle.factors["block"].activation.dimension == 5
    assert bundle.factors["block"].gradient.dimension == 4
    assert [parameter.requires_grad for parameter in model.parameters()] == original_flags
    bundle.save(tmp_path / "factors")
    loaded = FactorBundle.load(tmp_path / "factors")
    assert loaded.position_count == 7
    assert torch.allclose(
        loaded.factors["block"].activation.dense(),
        bundle.factors["block"].activation.dense(),
    )


def test_memory_guard_fails_before_allocation():
    model = TinyCausalLM()
    with pytest.raises(MemoryError, match="dense factor accumulation"):
        estimate_factors(
            model,
            [_replay(0, [0, 1, 2])],
            FactorEstimationConfig(
                target_modules=("block",),
                max_dense_bytes=1,
            ),
            pad_token_id=0,
        )


def test_factor_checkpoint_resumes_only_matching_replay(tmp_path):
    model = TinyCausalLM()
    records = [
        _replay(0, [0, 1, 2, 3]),
        _replay(1, [0, 4, 5, 6]),
    ]
    checkpoint = tmp_path / "accumulator.pt"
    common = dict(
        target_modules=("block",),
        batch_size=1,
        placement="cpu",
        energy_threshold=1.0,
        rank_cap=8,
        checkpoint_path=str(checkpoint),
        checkpoint_every_batches=1,
    )
    first = estimate_factors(
        model,
        records,
        FactorEstimationConfig(**common),
        pad_token_id=0,
    )
    resumed = estimate_factors(
        model,
        records,
        FactorEstimationConfig(**common, resume=True),
        pad_token_id=0,
    )
    assert resumed.position_count == first.position_count
    assert torch.allclose(
        resumed.factors["block"].gradient.dense(),
        first.factors["block"].gradient.dense(),
    )
    with pytest.raises(ValueError, match="replay digest"):
        estimate_factors(
            model,
            records + [_replay(2, [0, 7, 8])],
            FactorEstimationConfig(**common, resume=True),
            pad_token_id=0,
        )
