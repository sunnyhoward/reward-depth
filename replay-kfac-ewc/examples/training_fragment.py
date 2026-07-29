"""Minimal integration fragments for PEFT LoRA and full-rank training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from replay_kfac_ewc import FactorBundle, KFACEWC


def train_peft_lora(
    model: torch.nn.Module,
    batches: Iterable[Any],
    preference_loss: Callable[[torch.nn.Module, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    factor_directory: str,
    *,
    anchor_coefficient: float,
    adapter_name: str = "default",
) -> None:
    """Add this structure to an existing preference-training loop."""

    device = next(model.parameters()).device
    factors = FactorBundle.load(factor_directory, device=device)
    anchor = KFACEWC(factors, coefficient=anchor_coefficient)
    model.train()
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        task = preference_loss(model, batch)
        retention = anchor.penalty_from_peft(model, adapter_name)
        (task + retention).backward()
        optimizer.step()


def train_full_rank(
    model: torch.nn.Module,
    batches: Iterable[Any],
    task_loss: Callable[[torch.nn.Module, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    factor_directory: str,
    *,
    anchor_coefficient: float,
) -> None:
    """Full-rank variant; snapshots only the factor-targeted weights."""

    device = next(model.parameters()).device
    factors = FactorBundle.load(factor_directory, device=device)
    anchor = KFACEWC(factors, coefficient=anchor_coefficient)
    reference: Mapping[str, torch.Tensor] = anchor.capture_reference(model)
    model.train()
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        task = task_loss(model, batch)
        retention = anchor.penalty_against_reference(model, reference)
        (task + retention).backward()
        optimizer.step()
