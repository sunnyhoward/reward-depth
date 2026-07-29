from __future__ import annotations

import json
import hashlib
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from ._io import atomic_json, atomic_jsonl, sha256_file, stable_fraction


SCHEMA_VERSION = 1


@dataclass
class ReplayRecord:
    """One sampled continuation and the prefix from which it was generated."""

    seq_id: int
    sampling_seed: int
    seeding_mode: str
    prefix_ids: list[int]
    token_ids: list[int]
    prefix_length: int
    length: int
    mean_logprob: float
    has_repeat_loop: bool
    unique_token_count: int
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.seq_id < 0:
            raise ValueError("seq_id must be non-negative")
        if self.prefix_length < 1:
            raise ValueError("a causal-LM replay prefix must contain at least one token")
        if self.length != len(self.token_ids):
            raise ValueError("length does not equal len(token_ids)")
        if self.prefix_length != len(self.prefix_ids):
            raise ValueError("prefix_length does not equal len(prefix_ids)")
        if self.token_ids[: self.prefix_length] != self.prefix_ids:
            raise ValueError("token_ids do not begin with prefix_ids")
        if self.length <= self.prefix_length:
            raise ValueError("record contains no generated tokens")
        if self.split not in (None, "train", "heldout"):
            raise ValueError("split must be null, 'train', or 'heldout'")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not result["metadata"]:
            result.pop("metadata")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayRecord":
        known = {
            "seq_id",
            "sampling_seed",
            "seeding_mode",
            "prefix_ids",
            "token_ids",
            "prefix_length",
            "length",
            "mean_logprob",
            "has_repeat_loop",
            "unique_token_count",
            "split",
            "metadata",
            "schema_version",
        }
        metadata = dict(value.get("metadata", {}))
        metadata.update({key: item for key, item in value.items() if key not in known})
        result = cls(
            seq_id=int(value["seq_id"]),
            sampling_seed=int(value["sampling_seed"]),
            seeding_mode=str(value["seeding_mode"]),
            prefix_ids=[int(item) for item in value["prefix_ids"]],
            token_ids=[int(item) for item in value["token_ids"]],
            prefix_length=int(value["prefix_length"]),
            length=int(value["length"]),
            mean_logprob=float(value["mean_logprob"]),
            has_repeat_loop=bool(value["has_repeat_loop"]),
            unique_token_count=int(value["unique_token_count"]),
            split=value.get("split"),
            metadata=metadata,
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ReplayConfig:
    """Sampling settings.

    Sampling is deliberately one sequence at a time. That makes each record a
    pure function of ``seed + seq_id``: resuming, changing a shard boundary, or
    changing a throughput batch cannot silently change the corpus.
    """

    num_sequences: int
    seq_start: int = 0
    min_new_tokens: int = 64
    max_new_tokens: int = 384
    seed: int = 9001
    seeding_mode: Literal["mixed", "bos", "random", "prompt"] = "mixed"
    bos_fraction: float = 0.25
    prefix_min_tokens: int = 1
    prefix_max_tokens: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    ngram_size: int = 8
    ngram_repeats: int = 4
    resume: bool = True
    overwrite: bool = False
    fsync_every: int = 16

    def validate(self) -> None:
        if self.num_sequences < 1:
            raise ValueError("num_sequences must be positive")
        if self.seq_start < 0:
            raise ValueError("seq_start must be non-negative")
        if not 1 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("require 1 <= min_new_tokens <= max_new_tokens")
        if not 0.0 <= self.bos_fraction <= 1.0:
            raise ValueError("bos_fraction must lie in [0, 1]")
        if not 1 <= self.prefix_min_tokens <= self.prefix_max_tokens:
            raise ValueError("invalid random-prefix length range")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.fsync_every < 0:
            raise ValueError("fsync_every must be non-negative")


def load_replay(
    path: str | Path, split: str | None = None
) -> list[ReplayRecord]:
    records: list[ReplayRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = ReplayRecord.from_dict(json.loads(line))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: invalid replay record") from error
            if split is None or record.split == split:
                records.append(record)
    return records


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _seed_token_id(model: Any, tokenizer: Any) -> int:
    generation_config = getattr(model, "generation_config", None)
    candidates = (
        getattr(tokenizer, "bos_token_id", None),
        getattr(generation_config, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("tokenizer/model supplies neither a BOS nor an EOS token")


def _ordinary_token_ids(tokenizer: Any) -> list[int]:
    special = {int(item) for item in getattr(tokenizer, "all_special_ids", [])}
    size = len(tokenizer)
    result = [token_id for token_id in range(size) if token_id not in special]
    if not result:
        raise ValueError("tokenizer has no non-special tokens for random prefixes")
    return result


def _prompt_ids(prompt: Any, tokenizer: Any, seed_id: int) -> list[int]:
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    elif isinstance(prompt, Mapping):
        if "token_ids" in prompt:
            ids = prompt["token_ids"]
        elif "prompt_ids" in prompt:
            ids = prompt["prompt_ids"]
        elif "prompt" in prompt:
            ids = tokenizer.encode(str(prompt["prompt"]), add_special_tokens=False)
        else:
            raise ValueError("prompt mapping needs prompt, prompt_ids, or token_ids")
    else:
        ids = prompt
    result = [int(item) for item in ids]
    return result or [seed_id]


def _prefix_for(
    seq_id: int,
    config: ReplayConfig,
    tokenizer: Any,
    model: Any,
    prompts: Sequence[Any] | None,
    ordinary_ids: Sequence[int] | None,
) -> tuple[str, list[int], dict[str, Any]]:
    rng = random.Random(config.seed + seq_id)
    seed_id = _seed_token_id(model, tokenizer)
    mode = config.seeding_mode
    if mode == "mixed":
        mode = (
            "bos"
            if stable_fraction(str(seq_id), config.seed) < config.bos_fraction
            else "random"
        )
    if mode == "bos":
        return mode, [seed_id], {}
    if mode == "random":
        if ordinary_ids is None:
            raise AssertionError("ordinary token IDs were not prepared")
        count = rng.randint(config.prefix_min_tokens, config.prefix_max_tokens)
        return mode, [int(rng.choice(ordinary_ids)) for _ in range(count)], {}
    if mode == "prompt":
        if not prompts:
            raise ValueError("seeding_mode='prompt' requires at least one prompt")
        prompt_index = (seq_id - config.seq_start) % len(prompts)
        prompt = prompts[prompt_index]
        metadata = {"prompt_index": prompt_index}
        if isinstance(prompt, Mapping) and "id" in prompt:
            metadata["prompt_id"] = prompt["id"]
        return mode, _prompt_ids(prompt, tokenizer, seed_id), metadata
    raise ValueError(f"unknown seeding mode {mode!r}")


def ngram_loop_flag(
    token_ids: Sequence[int], n: int = 8, min_repeats: int = 4
) -> bool:
    if n < 1 or min_repeats < 2:
        raise ValueError("n must be positive and min_repeats must be at least two")
    counts: dict[tuple[int, ...], int] = {}
    for start in range(0, len(token_ids) - n + 1):
        ngram = tuple(token_ids[start : start + n])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] >= min_repeats:
            return True
    return False


@torch.inference_mode()
def _mean_generated_logprob(
    model: Any, token_ids: Sequence[int], prefix_length: int, device: torch.device
) -> float:
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    output = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1].float()
    targets = ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    selected = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    generated = selected[:, prefix_length - 1 :]
    if generated.numel() == 0:
        raise ValueError("generated sequence has no tokens to score")
    return float(generated.mean().cpu())


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _sampling_contract(config: ReplayConfig) -> dict[str, Any]:
    operational = {"num_sequences", "resume", "overwrite", "fsync_every"}
    return {
        key: value
        for key, value in asdict(config).items()
        if key not in operational
    }


def _prompt_digest(prompts: Sequence[Any] | None) -> str | None:
    if prompts is None:
        return None
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(
            json.dumps(
                prompt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _inferred_provenance(model: Any, tokenizer: Any) -> dict[str, Any]:
    model_config = getattr(model, "config", None)
    return {
        "model_class": type(model).__name__,
        "model_name_or_path": getattr(model_config, "_name_or_path", None),
        "model_commit": getattr(model_config, "_commit_hash", None),
        "model_vocab_size": getattr(model_config, "vocab_size", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_size": len(tokenizer),
    }


def generate_replay(
    model: Any,
    tokenizer: Any,
    output_path: str | Path,
    config: ReplayConfig,
    prompts: Sequence[Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[ReplayRecord]:
    """Generate a resumable replay shard from an already-loaded frozen model.

    ``output_path`` remains valid JSONL after every completed record. Existing
    records are validated before a resume; conflicting sequence IDs are never
    overwritten.
    """

    config.validate()
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_provenance = _inferred_provenance(model, tokenizer)
    effective_provenance.update(dict(provenance or {}))
    contract = _sampling_contract(config)
    contract["prompts_sha256"] = _prompt_digest(prompts)
    if output_path.exists() and config.overwrite:
        output_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    if output_path.exists() and not config.resume:
        raise FileExistsError(
            f"{output_path} exists; use resume=True or overwrite=True"
        )
    if output_path.exists():
        if not manifest_path.exists():
            raise ValueError(
                f"cannot safely resume {output_path}: its sampling manifest is missing"
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("sampling_contract") != contract:
            raise ValueError("resume sampling settings do not match the shard manifest")
        if previous_manifest.get("provenance") != effective_provenance:
            raise ValueError("resume model/tokenizer provenance does not match the shard")

    existing = load_replay(output_path) if output_path.exists() else []
    by_id: dict[int, ReplayRecord] = {}
    for record in existing:
        if record.seq_id in by_id:
            raise ValueError(f"duplicate seq_id {record.seq_id} in {output_path}")
        by_id[record.seq_id] = record

    atomic_json(
        manifest_path,
        {
            "schema": "replay-kfac-ewc/replay-shard-v1",
            "state": "generating",
            "sampling_contract": contract,
            "config": asdict(config),
            "provenance": effective_provenance,
            "output": str(output_path),
            "existing_records": len(existing),
        },
    )

    device = _model_device(model)
    model.eval()
    ordinary_ids = (
        _ordinary_token_ids(tokenizer)
        if config.seeding_mode in ("mixed", "random")
        else None
    )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    target_ids = range(config.seq_start, config.seq_start + config.num_sequences)
    new_records: list[ReplayRecord] = []
    with output_path.open("a", encoding="utf-8") as stream:
        for completed, seq_id in enumerate(target_ids, start=1):
            if seq_id in by_id:
                continue
            sampling_seed = config.seed + seq_id
            rng = random.Random(sampling_seed)
            new_tokens = rng.randint(config.min_new_tokens, config.max_new_tokens)
            mode, prefix_ids, metadata = _prefix_for(
                seq_id,
                config,
                tokenizer,
                model,
                prompts,
                ordinary_ids,
            )
            input_ids = torch.tensor(
                [prefix_ids], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": config.do_sample,
                "min_new_tokens": new_tokens,
                "max_new_tokens": new_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }
            if config.do_sample:
                generation_kwargs["temperature"] = config.temperature
            if pad_id is not None:
                generation_kwargs["pad_token_id"] = int(pad_id)
            if eos_id is not None:
                generation_kwargs["eos_token_id"] = int(eos_id)
            with torch.random.fork_rng(devices=_rng_devices(device)):
                torch.manual_seed(sampling_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sampling_seed)
                generated = model.generate(**generation_kwargs)
            token_ids = [int(item) for item in generated[0].detach().cpu().tolist()]
            record = ReplayRecord(
                seq_id=seq_id,
                sampling_seed=sampling_seed,
                seeding_mode=mode,
                prefix_ids=prefix_ids,
                token_ids=token_ids,
                prefix_length=len(prefix_ids),
                length=len(token_ids),
                mean_logprob=_mean_generated_logprob(
                    model, token_ids, len(prefix_ids), device
                ),
                has_repeat_loop=ngram_loop_flag(
                    token_ids, config.ngram_size, config.ngram_repeats
                ),
                unique_token_count=len(set(token_ids)),
                metadata=metadata,
            )
            record.validate()
            stream.write(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            if config.fsync_every and completed % config.fsync_every == 0:
                os.fsync(stream.fileno())
            by_id[seq_id] = record
            new_records.append(record)
        stream.flush()
        os.fsync(stream.fileno())

    selected = [by_id[seq_id] for seq_id in target_ids]
    manifest = {
        "schema": "replay-kfac-ewc/replay-shard-v1",
        "state": "complete",
        "sampling_contract": contract,
        "config": asdict(config),
        "provenance": effective_provenance,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "records": len(selected),
        "new_records": len(new_records),
        "scored_tokens": sum(item.length - item.prefix_length for item in selected),
    }
    atomic_json(manifest_path, manifest)
    return selected


def merge_replay(
    inputs: Iterable[str | Path],
    output_path: str | Path,
    *,
    heldout_fraction: float = 0.1,
    split_seed: int = 9002,
    deduplicate_content: bool = True,
) -> list[ReplayRecord]:
    """Merge shards, reject conflicting IDs, and assign a stable train split."""

    if not 0 <= heldout_fraction < 1:
        raise ValueError("heldout_fraction must lie in [0, 1)")
    paths = [Path(path) for path in inputs]
    if not paths:
        raise ValueError("at least one replay shard is required")
    by_id: dict[int, ReplayRecord] = {}
    for path in paths:
        for record in load_replay(path):
            previous = by_id.get(record.seq_id)
            if previous is not None:
                if previous.token_ids != record.token_ids:
                    raise ValueError(
                        f"seq_id {record.seq_id} has conflicting token sequences"
                    )
                continue
            by_id[record.seq_id] = record

    seen_content: set[tuple[int, ...]] = set()
    dropped_content_duplicates = 0
    result: list[ReplayRecord] = []
    for seq_id in sorted(by_id):
        source = by_id[seq_id]
        content = tuple(source.token_ids)
        if deduplicate_content and content in seen_content:
            dropped_content_duplicates += 1
            continue
        seen_content.add(content)
        source.split = (
            "heldout"
            if stable_fraction(str(seq_id), split_seed) < heldout_fraction
            else "train"
        )
        source.validate()
        result.append(source)
    output_path = Path(output_path)
    atomic_jsonl(output_path, (record.to_dict() for record in result))
    manifest = {
        "schema": "replay-kfac-ewc/replay-library-v1",
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "records": len(result),
        "train_records": sum(record.split == "train" for record in result),
        "heldout_records": sum(record.split == "heldout" for record in result),
        "scored_tokens": sum(record.length - record.prefix_length for record in result),
        "heldout_fraction": heldout_fraction,
        "split_seed": split_seed,
        "deduplicate_content": deduplicate_content,
        "dropped_content_duplicates": dropped_content_duplicates,
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return result
