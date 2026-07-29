from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from ._io import atomic_json, sha256_file
from .replay import ReplayRecord


FACTOR_SCHEMA = 1


@dataclass(frozen=True)
class CompressedFactor:
    """PSD matrix represented by top eigenpairs plus a diagonal tail."""

    eigvecs: torch.Tensor
    eigvals: torch.Tensor
    tail_diag: torch.Tensor
    dimension: int
    retained_energy: float
    trace: float

    def validate(self) -> None:
        if self.eigvecs.ndim != 2:
            raise ValueError("eigvecs must have shape [dimension, rank]")
        if self.eigvals.ndim != 1 or self.tail_diag.ndim != 1:
            raise ValueError("eigvals and tail_diag must be vectors")
        if self.eigvecs.shape[0] != self.dimension:
            raise ValueError("eigvec dimension mismatch")
        if self.eigvecs.shape[1] != self.eigvals.numel():
            raise ValueError("eigenvector/eigenvalue rank mismatch")
        if self.tail_diag.numel() != self.dimension:
            raise ValueError("tail diagonal dimension mismatch")
        if bool((self.eigvals < 0).any()) or bool((self.tail_diag < 0).any()):
            raise ValueError("compressed factor must be positive semidefinite")

    @property
    def rank(self) -> int:
        return int(self.eigvals.numel())

    @property
    def device(self) -> torch.device:
        return self.eigvals.device

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the represented matrix to ``value`` along its first axis."""

        if value.ndim < 1 or value.shape[0] != self.dimension:
            raise ValueError(
                f"factor dimension {self.dimension} cannot act on "
                f"shape {tuple(value.shape)}"
            )
        original_shape = value.shape
        flattened = value.reshape(self.dimension, -1)
        work_dtype = torch.promote_types(flattened.dtype, self.eigvals.dtype)
        work = flattened.to(dtype=work_dtype)
        vectors = self.eigvecs.to(device=value.device, dtype=work_dtype)
        values = self.eigvals.to(device=value.device, dtype=work_dtype)
        tail = self.tail_diag.to(device=value.device, dtype=work_dtype)
        projection = vectors.T @ work
        result = vectors @ (values[:, None] * projection) + tail[:, None] * work
        return result.reshape(original_shape)

    def dense(self) -> torch.Tensor:
        vectors = self.eigvecs.float()
        values = self.eigvals.float()
        return (
            (vectors * values) @ vectors.T
            + torch.diag(self.tail_diag.float())
        )

    def to(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> "CompressedFactor":
        return CompressedFactor(
            eigvecs=self.eigvecs.to(device=device, dtype=dtype),
            eigvals=self.eigvals.to(device=device, dtype=dtype),
            tail_diag=self.tail_diag.to(device=device, dtype=dtype),
            dimension=self.dimension,
            retained_energy=self.retained_energy,
            trace=self.trace,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "eigvecs": self.eigvecs.detach().cpu(),
            "eigvals": self.eigvals.detach().cpu(),
            "tail_diag": self.tail_diag.detach().cpu(),
            "dimension": self.dimension,
            "retained_energy": self.retained_energy,
            "trace": self.trace,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompressedFactor":
        result = cls(
            eigvecs=payload["eigvecs"],
            eigvals=payload["eigvals"],
            tail_diag=payload["tail_diag"],
            dimension=int(payload["dimension"]),
            retained_energy=float(payload["retained_energy"]),
            trace=float(payload["trace"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FactorPair:
    activation: CompressedFactor
    gradient: CompressedFactor

    def validate(self) -> None:
        self.activation.validate()
        self.gradient.validate()

    def to(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> "FactorPair":
        return FactorPair(
            activation=self.activation.to(device, dtype),
            gradient=self.gradient.to(device, dtype),
        )


@dataclass
class FactorBundle:
    """K-FAC factors keyed by the model's full linear-module names."""

    factors: dict[str, FactorPair]
    position_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = FACTOR_SCHEMA

    def validate(self) -> None:
        if self.schema_version != FACTOR_SCHEMA:
            raise ValueError(
                f"unsupported factor schema {self.schema_version}; "
                f"expected {FACTOR_SCHEMA}"
            )
        if self.position_count < 1:
            raise ValueError("factor bundle has no scored positions")
        if not self.factors:
            raise ValueError("factor bundle is empty")
        for name, pair in self.factors.items():
            if not name:
                raise ValueError("factor module name cannot be empty")
            pair.validate()

    def to(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> "FactorBundle":
        return FactorBundle(
            factors={
                name: pair.to(device, dtype) for name, pair in self.factors.items()
            },
            position_count=self.position_count,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )

    def save(self, directory: str | Path) -> Path:
        """Write ``factors.pt`` plus a human-readable, hashed manifest."""

        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "factors.pt"
        descriptor, temporary = tempfile.mkstemp(
            prefix=output.name, suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        payload = {
            "schema_version": self.schema_version,
            "position_count": self.position_count,
            "metadata": self.metadata,
            "factors": {
                name: {
                    "activation": pair.activation.to_payload(),
                    "gradient": pair.gradient.to_payload(),
                }
                for name, pair in self.factors.items()
            },
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        manifest = {
            "schema": "replay-kfac-ewc/factors-v1",
            "factor_file": output.name,
            "factor_sha256": sha256_file(output),
            "position_count": self.position_count,
            "modules": {
                name: {
                    "input_dimension": pair.activation.dimension,
                    "output_dimension": pair.gradient.dimension,
                    "activation_rank": pair.activation.rank,
                    "gradient_rank": pair.gradient.rank,
                    "activation_energy": pair.activation.retained_energy,
                    "gradient_energy": pair.gradient.retained_energy,
                }
                for name, pair in self.factors.items()
            },
            "metadata": self.metadata,
        }
        atomic_json(directory / "manifest.json", manifest)
        return output

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
        verify_hash: bool = True,
    ) -> "FactorBundle":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        factor_path = directory / manifest["factor_file"]
        if verify_hash and sha256_file(factor_path) != manifest["factor_sha256"]:
            raise ValueError("factor file does not match manifest SHA-256")
        payload = torch.load(factor_path, map_location=device, weights_only=False)
        result = cls(
            factors={
                name: FactorPair(
                    activation=CompressedFactor.from_payload(value["activation"]),
                    gradient=CompressedFactor.from_payload(value["gradient"]),
                )
                for name, value in payload["factors"].items()
            },
            position_count=int(payload["position_count"]),
            metadata=dict(payload.get("metadata", {})),
            schema_version=int(payload.get("schema_version", 1)),
        )
        result.validate()
        return result


def compress_psd(
    matrix: torch.Tensor,
    *,
    energy_threshold: float = 0.99,
    rank_cap: int = 1024,
    output_dtype: torch.dtype = torch.float32,
) -> CompressedFactor:
    """Compress a dense PSD estimate after global normalization.

    The exact diagonal of the discarded residual is retained. The off-diagonal
    tail is deliberately not: that is the package's only factor-storage
    approximation.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must lie in (0, 1]")
    if rank_cap < 1:
        raise ValueError("rank_cap must be positive")
    dimension = matrix.shape[0]
    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    symmetric = 0.5 * (matrix.to(work_dtype) + matrix.to(work_dtype).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)
    trace = eigenvalues.sum()
    if float(trace) <= 1e-30:
        rank = 1
        retained_energy = 1.0
    else:
        cumulative = torch.cumsum(eigenvalues, dim=0) / trace
        indices = (cumulative >= energy_threshold).nonzero(as_tuple=True)[0]
        required = int(indices[0]) + 1 if indices.numel() else dimension
        rank = max(1, min(required, rank_cap, dimension))
        retained_energy = float(eigenvalues[:rank].sum() / trace)
    values = eigenvalues[:rank].clone()
    vectors = eigenvectors[:, :rank].clone()
    diagonal_top = vectors.square() @ values
    tail = (torch.diagonal(symmetric) - diagonal_top).clamp_min(0)
    result = CompressedFactor(
        eigvecs=vectors.to(output_dtype),
        eigvals=values.to(output_dtype),
        tail_diag=tail.to(output_dtype),
        dimension=dimension,
        retained_energy=retained_energy,
        trace=float(trace),
    )
    result.validate()
    return result


def _matches_selector(name: str, selector: str) -> bool:
    return name == selector or name.endswith("." + selector)


def resolve_linear_modules(
    model: torch.nn.Module, selectors: Sequence[str]
) -> dict[str, torch.nn.Linear]:
    """Resolve every ``nn.Linear`` whose full name matches a selected suffix."""

    if not selectors:
        raise ValueError("at least one target-module selector is required")
    named = dict(model.named_modules())
    result: dict[str, torch.nn.Linear] = {}
    for selector in selectors:
        matches = [
            (name, module)
            for name, module in named.items()
            if _matches_selector(name, selector) and isinstance(module, torch.nn.Linear)
        ]
        if not matches:
            raise ValueError(f"no nn.Linear module matches selector {selector!r}")
        for name, module in matches:
            result[name] = module
    return dict(sorted(result.items()))


def dense_accumulator_bytes(
    modules: Mapping[str, torch.nn.Linear] | Iterable[torch.nn.Linear],
) -> int:
    values = modules.values() if isinstance(modules, Mapping) else modules
    elements = sum(
        int(module.in_features) ** 2 + int(module.out_features) ** 2
        for module in values
    )
    return 4 * elements


@dataclass(frozen=True)
class FactorEstimationConfig:
    target_modules: tuple[str, ...]
    split: str | None = "train"
    batch_size: int = 1
    max_records: int | None = None
    max_positions: int | None = None
    placement: Literal["model", "cpu", "auto"] = "auto"
    large_dimension: int = 4096
    max_dense_bytes: int | None = 64 * 2**30
    energy_threshold: float = 0.99
    rank_cap: int = 1024
    eigendecomposition_device: str = "cpu"
    eigendecomposition_dtype: Literal["float32", "float64"] = "float64"
    checkpoint_path: str | None = None
    checkpoint_every_batches: int = 0
    resume: bool = False

    def validate(self) -> None:
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("max_records must be positive")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if self.large_dimension < 1:
            raise ValueError("large_dimension must be positive")
        if self.max_dense_bytes is not None and self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be positive or None")
        if not 0 < self.energy_threshold <= 1:
            raise ValueError("energy_threshold must lie in (0, 1]")
        if self.rank_cap < 1:
            raise ValueError("rank_cap must be positive")
        if self.checkpoint_every_batches < 0:
            raise ValueError("checkpoint_every_batches cannot be negative")
        if self.checkpoint_every_batches and self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_every_batches requires checkpoint_path"
            )
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")


def _as_record(value: ReplayRecord | Mapping[str, Any]) -> ReplayRecord:
    return value if isinstance(value, ReplayRecord) else ReplayRecord.from_dict(value)


def _records_digest(records: Sequence[ReplayRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                {
                    "seq_id": record.seq_id,
                    "prefix_length": record.prefix_length,
                    "token_ids": record.token_ids,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _teacher_forcing_batch(
    records: Sequence[ReplayRecord], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(record.length for record in records)
    input_ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    predictor_mask = torch.zeros(
        (len(records), max(width - 1, 0)), dtype=torch.bool
    )
    for row, record in enumerate(records):
        offset = width - record.length
        input_ids[row, offset:] = torch.tensor(record.token_ids, dtype=torch.long)
        attention_mask[row, offset:] = 1
        first = offset + max(record.prefix_length, 1) - 1
        last_exclusive = width - 1
        predictor_mask[row, first:last_exclusive] = True
    return input_ids, attention_mask, predictor_mask


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class _DenseAccumulator:
    def __init__(
        self,
        modules: Mapping[str, torch.nn.Linear],
        model_device: torch.device,
        placement: str,
        large_dimension: int,
    ):
        self.activation: dict[str, torch.Tensor] = {}
        self.gradient: dict[str, torch.Tensor] = {}
        self.locations: dict[tuple[str, str], torch.device] = {}
        self.position_count = 0
        self.records_consumed = 0
        for name, module in modules.items():
            for side, dimension in (
                ("activation", int(module.in_features)),
                ("gradient", int(module.out_features)),
            ):
                if placement == "model":
                    device = model_device
                elif placement == "cpu":
                    device = torch.device("cpu")
                else:
                    device = (
                        torch.device("cpu")
                        if dimension > large_dimension
                        else model_device
                    )
                target = torch.zeros(
                    dimension, dimension, dtype=torch.float32, device=device
                )
                getattr(self, side)[name] = target
                self.locations[(name, side)] = device

    def add(
        self, name: str, activation: torch.Tensor, gradient: torch.Tensor
    ) -> None:
        for side, values in (
            ("activation", activation),
            ("gradient", gradient),
        ):
            target = getattr(self, side)[name]
            values = values.float()
            if target.device == values.device:
                target.addmm_(values.T, values)
            else:
                target.add_((values.T @ values).to(target.device))

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        records_sha256: str,
        module_shapes: Mapping[str, tuple[int, int]],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name, suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        payload = {
            "schema": "replay-kfac-ewc/factor-checkpoint-v1",
            "records_sha256": records_sha256,
            "module_shapes": dict(module_shapes),
            "position_count": self.position_count,
            "records_consumed": self.records_consumed,
            "activation": {
                name: value.detach().cpu() for name, value in self.activation.items()
            },
            "gradient": {
                name: value.detach().cpu() for name, value in self.gradient.items()
            },
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        records_sha256: str,
        module_shapes: Mapping[str, tuple[int, int]],
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "replay-kfac-ewc/factor-checkpoint-v1":
            raise ValueError("unsupported factor checkpoint schema")
        if payload.get("records_sha256") != records_sha256:
            raise ValueError("factor checkpoint replay digest mismatch")
        stored_shapes = {
            name: tuple(shape)
            for name, shape in payload.get("module_shapes", {}).items()
        }
        if stored_shapes != dict(module_shapes):
            raise ValueError("factor checkpoint module shapes mismatch")
        for side in ("activation", "gradient"):
            current = getattr(self, side)
            stored = payload[side]
            if set(current) != set(stored):
                raise ValueError(f"factor checkpoint {side} module set mismatch")
            for name, value in stored.items():
                if tuple(value.shape) != tuple(current[name].shape):
                    raise ValueError(
                        f"factor checkpoint {side}[{name!r}] shape mismatch"
                    )
                current[name].copy_(value)
        self.position_count = int(payload["position_count"])
        self.records_consumed = int(payload["records_consumed"])


def estimate_factors(
    model: torch.nn.Module,
    records: Sequence[ReplayRecord | Mapping[str, Any]],
    config: FactorEstimationConfig,
    *,
    pad_token_id: int,
) -> FactorBundle:
    """Estimate replay Fisher K-FAC factors for selected linear modules.

    The loss is the sum of own-sample token negative log-likelihoods. One
    backward pass is used per sequence batch, so transformer token gradients
    follow the standard practical K-FAC convention: causal mixing from later
    scored positions is present. Factors are normalized once by the global
    number of scored positions and only then compressed.
    """

    config.validate()
    selected = [_as_record(record) for record in records]
    if config.split is not None:
        selected = [record for record in selected if record.split == config.split]
    selected.sort(key=lambda record: (-record.length, record.seq_id))
    if config.max_records is not None:
        selected = selected[: config.max_records]
    if not selected:
        raise ValueError("no replay records selected for factor estimation")

    modules = resolve_linear_modules(model, config.target_modules)
    module_shapes = {
        name: (int(module.in_features), int(module.out_features))
        for name, module in modules.items()
    }
    required_bytes = dense_accumulator_bytes(modules)
    if (
        config.max_dense_bytes is not None
        and required_bytes > config.max_dense_bytes
    ):
        gib = required_bytes / 2**30
        limit = config.max_dense_bytes / 2**30
        raise MemoryError(
            f"dense factor accumulation requires {gib:.2f} GiB, above the "
            f"configured {limit:.2f} GiB limit"
        )

    model_device = _model_device(model)
    accumulator = _DenseAccumulator(
        modules,
        model_device,
        config.placement,
        config.large_dimension,
    )
    selected_digest = _records_digest(selected)
    if config.resume:
        checkpoint = Path(config.checkpoint_path or "")
        if not checkpoint.exists():
            raise FileNotFoundError(f"factor checkpoint {checkpoint} does not exist")
        accumulator.load_checkpoint(
            checkpoint,
            records_sha256=selected_digest,
            module_shapes=module_shapes,
        )
        if not 0 <= accumulator.records_consumed <= len(selected):
            raise ValueError("factor checkpoint records_consumed is out of range")
    captured_activation: dict[str, torch.Tensor] = {}
    captured_gradient: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    embeddings = model.get_input_embeddings()

    def embedding_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return output.detach().requires_grad_(True)

    handles.append(embeddings.register_forward_hook(embedding_hook))
    for name, module in modules.items():
        def capture_activation(
            _module: Any, inputs: tuple[torch.Tensor, ...], _output: Any, *, key: str = name
        ) -> None:
            captured_activation[key] = inputs[0].detach()

        def capture_gradient(
            _module: Any,
            _gradient_input: Any,
            gradient_output: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captured_gradient[key] = gradient_output[0].detach()

        handles.append(module.register_forward_hook(capture_activation))
        handles.append(module.register_full_backward_hook(capture_gradient))

    training = model.training
    require_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    consumed = accumulator.records_consumed
    batches_since_resume = 0
    try:
        for start in range(consumed, len(selected), config.batch_size):
            if (
                config.max_positions is not None
                and accumulator.position_count >= config.max_positions
            ):
                break
            batch = selected[start : start + config.batch_size]
            ids, attention, mask = _teacher_forcing_batch(batch, pad_token_id)
            ids = ids.to(model_device)
            attention = attention.to(model_device)
            mask = mask.to(model_device)
            captured_activation.clear()
            captured_gradient.clear()
            output = model(
                input_ids=ids, attention_mask=attention, use_cache=False
            )
            logits = output.logits[:, :-1].float()
            targets = ids[:, 1:]
            token_logprob = F.log_softmax(logits, dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            loss = -(token_logprob * mask).sum()
            loss.backward()
            for name in modules:
                if name not in captured_activation or name not in captured_gradient:
                    raise RuntimeError(
                        f"hooks did not capture module {name!r}; disable gradient "
                        "checkpointing and ensure the module is on the forward path"
                    )
                activation = captured_activation[name]
                gradient = captured_gradient[name]
                if activation.ndim != 3 or gradient.ndim != 3:
                    raise ValueError(
                        f"module {name!r} must have [batch, sequence, feature] "
                        "activations for this estimator"
                    )
                valid_activation = activation[:, :-1, :][mask]
                valid_gradient = gradient[:, :-1, :][mask]
                accumulator.add(name, valid_activation, valid_gradient)
            count = int(mask.sum())
            accumulator.position_count += count
            consumed += len(batch)
            accumulator.records_consumed = consumed
            batches_since_resume += 1
            captured_activation.clear()
            captured_gradient.clear()
            if (
                config.checkpoint_path is not None
                and config.checkpoint_every_batches
                and batches_since_resume % config.checkpoint_every_batches == 0
            ):
                accumulator.save_checkpoint(
                    config.checkpoint_path,
                    records_sha256=selected_digest,
                    module_shapes=module_shapes,
                )
            if (
                config.max_positions is not None
                and accumulator.position_count >= config.max_positions
            ):
                break
    finally:
        for handle in handles:
            handle.remove()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(require_grad[name])
        model.train(training)

    if accumulator.position_count < 1:
        raise ValueError("no generated-token positions were accumulated")
    normalization = float(accumulator.position_count)
    eig_dtype = (
        torch.float64
        if config.eigendecomposition_dtype == "float64"
        else torch.float32
    )
    factors: dict[str, FactorPair] = {}
    for name in modules:
        activation = (
            accumulator.activation[name]
            .to(config.eigendecomposition_device, dtype=eig_dtype)
            .div(normalization)
        )
        gradient = (
            accumulator.gradient[name]
            .to(config.eigendecomposition_device, dtype=eig_dtype)
            .div(normalization)
        )
        factors[name] = FactorPair(
            activation=compress_psd(
                activation,
                energy_threshold=config.energy_threshold,
                rank_cap=config.rank_cap,
            ),
            gradient=compress_psd(
                gradient,
                energy_threshold=config.energy_threshold,
                rank_cap=config.rank_cap,
            ),
        )
    bundle = FactorBundle(
        factors=factors,
        position_count=accumulator.position_count,
        metadata={
            "estimator": "own-sample teacher-forced K-FAC",
            "gradient_convention": "one summed backward per batch",
            "normalization": "global scored-token count",
            "dense_accumulator_bytes": required_bytes,
            "records_consumed": consumed,
            "selected_replay_sha256": _records_digest(selected[:consumed]),
            "config": asdict(config),
        },
    )
    bundle.validate()
    return bundle
