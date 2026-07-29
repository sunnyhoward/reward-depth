from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .factors import FactorBundle, FactorPair


@dataclass(frozen=True)
class LoRADelta:
    """Low-rank update ``delta_weight = scale * B @ A``."""

    A: torch.Tensor
    B: torch.Tensor
    scale: float | torch.Tensor = 1.0

    def validate(self, pair: FactorPair, module_name: str = "") -> None:
        expected_a = (self.A.shape[0], pair.activation.dimension)
        expected_b = (pair.gradient.dimension, self.A.shape[0])
        if tuple(self.A.shape) != expected_a:
            raise ValueError(
                f"{module_name}: LoRA A has shape {tuple(self.A.shape)}; "
                f"expected [rank, {pair.activation.dimension}]"
            )
        if tuple(self.B.shape) != expected_b:
            raise ValueError(
                f"{module_name}: LoRA B has shape {tuple(self.B.shape)}; "
                f"expected [{pair.gradient.dimension}, {self.A.shape[0]}]"
            )


class KFACEWC:
    """Differentiable fixed-reference EWC penalty backed by K-FAC factors."""

    def __init__(
        self,
        factors: FactorBundle,
        *,
        coefficient: float = 1.0,
        strict: bool = True,
    ):
        factors.validate()
        if coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        self.factors = factors
        self.coefficient = float(coefficient)
        self.strict = strict

    def _check_keys(self, supplied: Mapping[str, Any]) -> list[str]:
        available = set(supplied)
        expected = set(self.factors.factors)
        missing = expected - available
        extra = available - expected
        if self.strict and missing:
            raise KeyError(f"missing updates for factor modules: {sorted(missing)}")
        if self.strict and extra:
            raise KeyError(f"updates have no factors: {sorted(extra)}")
        selected = sorted(expected & available)
        if not selected:
            raise ValueError("no updates match the factor bundle")
        return selected

    def terms_from_deltas(
        self, deltas: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Per-module ``0.5 tr(G delta_W A delta_W^T)`` terms."""

        result: dict[str, torch.Tensor] = {}
        for name in self._check_keys(deltas):
            delta = deltas[name]
            pair = self.factors.factors[name]
            expected = (pair.gradient.dimension, pair.activation.dimension)
            if tuple(delta.shape) != expected:
                raise ValueError(
                    f"{name}: delta has shape {tuple(delta.shape)}; expected {expected}"
                )
            gradient_delta = pair.gradient.apply(delta)
            delta_activation = pair.activation.apply(delta.T).T
            result[name] = 0.5 * (gradient_delta * delta_activation).sum()
        return result

    def penalty_from_deltas(
        self, deltas: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        terms = self.terms_from_deltas(deltas)
        return self.coefficient * torch.stack(list(terms.values())).sum()

    def terms_from_lora(
        self, updates: Mapping[str, LoRADelta]
    ) -> dict[str, torch.Tensor]:
        """Per-module penalty without materializing full LoRA weight deltas."""

        result: dict[str, torch.Tensor] = {}
        for name in self._check_keys(updates):
            update = updates[name]
            pair = self.factors.factors[name]
            update.validate(pair, name)
            gradient_b = pair.gradient.apply(update.B)
            activation_at = pair.activation.apply(update.A.T)
            inner_gradient = update.B.to(gradient_b.dtype).T @ gradient_b
            inner_activation = update.A.to(activation_at.dtype) @ activation_at
            scale = torch.as_tensor(
                update.scale, device=update.A.device, dtype=inner_activation.dtype
            )
            result[name] = (
                0.5
                * scale.square()
                * torch.trace(inner_gradient @ inner_activation)
            )
        return result

    def penalty_from_lora(
        self, updates: Mapping[str, LoRADelta]
    ) -> torch.Tensor:
        terms = self.terms_from_lora(updates)
        return self.coefficient * torch.stack(list(terms.values())).sum()

    @staticmethod
    def _container_item(container: Any, adapter_name: str, label: str) -> Any:
        try:
            return container[adapter_name]
        except (KeyError, TypeError, IndexError) as error:
            raise KeyError(
                f"PEFT module has no {label} entry for adapter {adapter_name!r}"
            ) from error

    def lora_updates_from_peft(
        self, model: torch.nn.Module, adapter_name: str = "default"
    ) -> dict[str, LoRADelta]:
        """Extract differentiable A/B tensors from a PEFT LoRA model.

        Factor names are resolved by full-name suffix, so the
        ``base_model.model.`` prefix PEFT adds does not affect matching.
        """

        candidates = {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }
        result: dict[str, LoRADelta] = {}
        for factor_name in self.factors.factors:
            matches = [
                (name, module)
                for name, module in candidates.items()
                if name == factor_name or name.endswith("." + factor_name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one PEFT LoRA module for {factor_name!r}, "
                    f"found {[name for name, _ in matches]}"
                )
            _, module = matches[0]
            a_module = self._container_item(module.lora_A, adapter_name, "lora_A")
            b_module = self._container_item(module.lora_B, adapter_name, "lora_B")
            scaling = getattr(module, "scaling", None)
            if scaling is None:
                alpha = self._container_item(
                    module.lora_alpha, adapter_name, "lora_alpha"
                )
                rank = self._container_item(module.r, adapter_name, "rank")
                scale: float | torch.Tensor = alpha / rank
            elif isinstance(scaling, Mapping):
                scale = self._container_item(scaling, adapter_name, "scaling")
            else:
                scale = scaling
            result[factor_name] = LoRADelta(
                A=a_module.weight,
                B=b_module.weight,
                scale=scale,
            )
        return result

    def penalty_from_peft(
        self, model: torch.nn.Module, adapter_name: str = "default"
    ) -> torch.Tensor:
        return self.penalty_from_lora(
            self.lora_updates_from_peft(model, adapter_name)
        )

    def capture_reference(
        self, model: torch.nn.Module, *, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Snapshot only targeted full-rank weights for non-LoRA training."""

        modules = dict(model.named_modules())
        result: dict[str, torch.Tensor] = {}
        for factor_name in self.factors.factors:
            matches = [
                module
                for name, module in modules.items()
                if (name == factor_name or name.endswith("." + factor_name))
                and hasattr(module, "weight")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one weighted module for {factor_name!r}, "
                    f"found {len(matches)}"
                )
            result[factor_name] = matches[0].weight.detach().to(device).clone()
        return result

    def penalty_against_reference(
        self,
        model: torch.nn.Module,
        reference_weights: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Penalty for full-rank training relative to a targeted snapshot."""

        modules = dict(model.named_modules())
        deltas: dict[str, torch.Tensor] = {}
        for factor_name in self._check_keys(reference_weights):
            matches = [
                module
                for name, module in modules.items()
                if (name == factor_name or name.endswith("." + factor_name))
                and hasattr(module, "weight")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one weighted module for {factor_name!r}, "
                    f"found {len(matches)}"
                )
            weight = matches[0].weight
            reference = reference_weights[factor_name].to(
                device=weight.device, dtype=weight.dtype
            )
            deltas[factor_name] = weight - reference
        return self.penalty_from_deltas(deltas)
