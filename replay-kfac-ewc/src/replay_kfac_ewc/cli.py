from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .factors import (
    FactorBundle,
    FactorEstimationConfig,
    dense_accumulator_bytes,
    estimate_factors,
    resolve_linear_modules,
)
from ._io import sha256_file
from .replay import ReplayConfig, generate_replay, load_replay, merge_replay


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_huggingface(
    model_name: str,
    *,
    revision: str | None,
    device: str,
    dtype: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face commands require `pip install replay-kfac-ewc[huggingface]`"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=_dtype(dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _prompts(path: Path | None) -> list[Any] | None:
    if path is None:
        return None
    result: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            else:
                result.append(line)
    if not result:
        raise ValueError(f"prompt file {path} is empty")
    return result


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Hugging Face ID or local path")
    parser.add_argument("--revision", help="pin a Hugging Face commit or revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")


def _generate(args: argparse.Namespace) -> int:
    model, tokenizer = _load_huggingface(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    config = ReplayConfig(
        num_sequences=args.num_sequences,
        seq_start=args.seq_start,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        seeding_mode=args.seeding_mode,
        bos_fraction=args.bos_fraction,
        prefix_min_tokens=args.prefix_min_tokens,
        prefix_max_tokens=args.prefix_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        fsync_every=args.fsync_every,
    )
    records = generate_replay(
        model,
        tokenizer,
        args.output,
        config,
        prompts=_prompts(args.prompts),
        provenance={
            "requested_model": args.model,
            "requested_revision": args.revision,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "scored_tokens": sum(
                    record.length - record.prefix_length for record in records
                ),
            },
            indent=2,
        )
    )
    return 0


def _merge(args: argparse.Namespace) -> int:
    records = merge_replay(
        args.inputs,
        args.output,
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
        deduplicate_content=not args.keep_content_duplicates,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "train": sum(record.split == "train" for record in records),
                "heldout": sum(record.split == "heldout" for record in records),
            },
            indent=2,
        )
    )
    return 0


def _estimate(args: argparse.Namespace) -> int:
    model, tokenizer = _load_huggingface(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    modules = resolve_linear_modules(model, tuple(args.target))
    required = dense_accumulator_bytes(modules)
    print(
        f"resolved {len(modules)} linear modules; dense accumulators require "
        f"{required / 2**30:.2f} GiB"
    )
    records = load_replay(args.corpus)
    config = FactorEstimationConfig(
        target_modules=tuple(args.target),
        split=args.split,
        batch_size=args.batch_size,
        max_records=args.max_records,
        max_positions=args.max_positions,
        placement=args.placement,
        large_dimension=args.large_dimension,
        max_dense_bytes=(
            None if args.no_memory_guard else int(args.max_dense_gib * 2**30)
        ),
        energy_threshold=args.energy_threshold,
        rank_cap=args.rank_cap,
        eigendecomposition_device=args.eigh_device,
        eigendecomposition_dtype=args.eigh_dtype,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        checkpoint_every_batches=args.checkpoint_every,
        resume=args.resume,
    )
    bundle = estimate_factors(
        model,
        records,
        config,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    bundle.metadata.update(
        {
            "requested_model": args.model,
            "requested_revision": args.revision,
            "corpus_path": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
        }
    )
    factor_file = bundle.save(args.output)
    print(
        json.dumps(
            {
                "output": str(factor_file),
                "modules": len(bundle.factors),
                "positions": bundle.position_count,
            },
            indent=2,
        )
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    bundle = FactorBundle.load(args.factors, verify_hash=not args.no_verify_hash)
    print(
        json.dumps(
            {
                "position_count": bundle.position_count,
                "modules": {
                    name: {
                        "input_dimension": pair.activation.dimension,
                        "output_dimension": pair.gradient.dimension,
                        "activation_rank": pair.activation.rank,
                        "gradient_rank": pair.gradient.rank,
                        "activation_energy": pair.activation.retained_energy,
                        "gradient_energy": pair.gradient.retained_energy,
                    }
                    for name, pair in bundle.factors.items()
                },
                "metadata": bundle.metadata,
            },
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="replay-kfac-ewc",
        description="Generate replay corpora and estimate K-FAC EWC anchors.",
    )
    commands = result.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="sample a resumable replay shard")
    _add_model_arguments(generate)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--num-sequences", type=int, required=True)
    generate.add_argument("--seq-start", type=int, default=0)
    generate.add_argument("--min-new-tokens", type=int, default=64)
    generate.add_argument("--max-new-tokens", type=int, default=384)
    generate.add_argument("--seed", type=int, default=9001)
    generate.add_argument(
        "--seeding-mode",
        choices=("mixed", "bos", "random", "prompt"),
        default="mixed",
    )
    generate.add_argument("--prompts", type=Path)
    generate.add_argument("--bos-fraction", type=float, default=0.25)
    generate.add_argument("--prefix-min-tokens", type=int, default=1)
    generate.add_argument("--prefix-max-tokens", type=int, default=8)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--fsync-every", type=int, default=16)
    generate.add_argument("--no-resume", action="store_true")
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(run=_generate)

    merge = commands.add_parser("merge", help="merge shards and assign splits")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--heldout-fraction", type=float, default=0.1)
    merge.add_argument("--split-seed", type=int, default=9002)
    merge.add_argument("--keep-content-duplicates", action="store_true")
    merge.set_defaults(run=_merge)

    estimate = commands.add_parser(
        "estimate", help="estimate and compress true-Fisher K-FAC factors"
    )
    _add_model_arguments(estimate)
    estimate.add_argument("--corpus", type=Path, required=True)
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument(
        "--target",
        action="append",
        required=True,
        help="linear-module full name or suffix; repeat for multiple projections",
    )
    estimate.add_argument("--split", default="train")
    estimate.add_argument("--batch-size", type=int, default=1)
    estimate.add_argument("--max-records", type=int)
    estimate.add_argument("--max-positions", type=int)
    estimate.add_argument(
        "--placement", choices=("model", "cpu", "auto"), default="auto"
    )
    estimate.add_argument("--large-dimension", type=int, default=4096)
    estimate.add_argument("--max-dense-gib", type=float, default=64.0)
    estimate.add_argument("--no-memory-guard", action="store_true")
    estimate.add_argument("--energy-threshold", type=float, default=0.99)
    estimate.add_argument("--rank-cap", type=int, default=1024)
    estimate.add_argument("--eigh-device", default="cpu")
    estimate.add_argument(
        "--eigh-dtype", choices=("float32", "float64"), default="float64"
    )
    estimate.add_argument("--checkpoint", type=Path)
    estimate.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="atomically checkpoint dense sums every N batches",
    )
    estimate.add_argument(
        "--resume",
        action="store_true",
        help="resume from --checkpoint after verifying replay and module shapes",
    )
    estimate.set_defaults(run=_estimate)

    inspect = commands.add_parser("inspect", help="verify and summarize a factor bundle")
    inspect.add_argument("--factors", type=Path, required=True)
    inspect.add_argument("--no-verify-hash", action="store_true")
    inspect.set_defaults(run=_inspect)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
