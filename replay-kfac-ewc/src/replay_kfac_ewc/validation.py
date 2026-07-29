from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .factors import _model_device, _teacher_forcing_batch
from .replay import ReplayRecord


@dataclass(frozen=True)
class CalibrationReport:
    """Local comparison between K-FAC predictions and measured replay KL."""

    count: int
    log_log_slope: float
    spearman_correlation: float
    kl_per_penalty_geometric_mean: float
    median_kl_per_penalty: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator > 0 else float("nan")


def fit_calibration(
    predicted_penalties: Sequence[float],
    measured_kls: Sequence[float],
) -> CalibrationReport:
    """Fit the local relationship ``measured_KL ≈ c * predicted_penalty``.

    Use perturbations small enough to remain in the local quadratic regime.
    ``kl_per_penalty_geometric_mean`` is the multiplicative coefficient that
    puts the penalty on the measured-KL scale.
    """

    if len(predicted_penalties) != len(measured_kls):
        raise ValueError("predicted and measured arrays must have the same length")
    if len(predicted_penalties) < 2:
        raise ValueError("at least two perturbations are required")
    if any(value <= 0 or not math.isfinite(value) for value in predicted_penalties):
        raise ValueError("predicted penalties must be finite and positive")
    if any(value <= 0 or not math.isfinite(value) for value in measured_kls):
        raise ValueError("measured KLs must be finite and positive")

    log_predicted = [math.log(value) for value in predicted_penalties]
    log_measured = [math.log(value) for value in measured_kls]
    predicted_mean = sum(log_predicted) / len(log_predicted)
    measured_mean = sum(log_measured) / len(log_measured)
    denominator = sum((value - predicted_mean) ** 2 for value in log_predicted)
    slope = (
        sum(
            (x - predicted_mean) * (y - measured_mean)
            for x, y in zip(log_predicted, log_measured)
        )
        / denominator
        if denominator > 0
        else float("nan")
    )
    ratios = [
        measured / predicted
        for predicted, measured in zip(predicted_penalties, measured_kls)
    ]
    ordered_ratios = sorted(ratios)
    middle = len(ratios) // 2
    median = (
        ordered_ratios[middle]
        if len(ratios) % 2
        else 0.5 * (ordered_ratios[middle - 1] + ordered_ratios[middle])
    )
    return CalibrationReport(
        count=len(ratios),
        log_log_slope=slope,
        spearman_correlation=_correlation(
            _average_ranks(predicted_penalties), _average_ranks(measured_kls)
        ),
        kl_per_penalty_geometric_mean=math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        ),
        median_kl_per_penalty=median,
    )


def _as_record(value: ReplayRecord | Mapping[str, Any]) -> ReplayRecord:
    return value if isinstance(value, ReplayRecord) else ReplayRecord.from_dict(value)


@torch.no_grad()
def mean_forward_kl(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    records: Sequence[ReplayRecord | Mapping[str, Any]],
    *,
    pad_token_id: int,
    split: str | None = "heldout",
    batch_size: int = 1,
    max_positions: int | None = None,
) -> float:
    """Measure ``KL(reference || candidate)`` on replay predictor positions."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = [_as_record(record) for record in records]
    if split is not None:
        selected = [record for record in selected if record.split == split]
    if not selected:
        raise ValueError("no replay records selected for KL measurement")
    selected.sort(key=lambda record: (-record.length, record.seq_id))
    reference_device = _model_device(reference_model)
    candidate_device = _model_device(candidate_model)
    reference_training = reference_model.training
    candidate_training = candidate_model.training
    reference_model.eval()
    candidate_model.eval()
    total = 0.0
    count = 0
    try:
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            ids, attention, mask = _teacher_forcing_batch(batch, pad_token_id)
            reference_output = reference_model(
                input_ids=ids.to(reference_device),
                attention_mask=attention.to(reference_device),
                use_cache=False,
            )
            candidate_output = candidate_model(
                input_ids=ids.to(candidate_device),
                attention_mask=attention.to(candidate_device),
                use_cache=False,
            )
            reference_logprob = F.log_softmax(
                reference_output.logits[:, :-1].float(), dim=-1
            )
            candidate_logprob = F.log_softmax(
                candidate_output.logits[:, :-1].float(), dim=-1
            ).to(reference_device)
            probability = reference_logprob.exp()
            token_kl = (
                probability * (reference_logprob - candidate_logprob)
            ).sum(-1)
            valid = token_kl[mask.to(reference_device)]
            total += float(valid.sum().cpu())
            count += int(valid.numel())
            if max_positions is not None and count >= max_positions:
                break
    finally:
        reference_model.train(reference_training)
        candidate_model.train(candidate_training)
    if count < 1:
        raise ValueError("no predictor positions were measured")
    return total / count
