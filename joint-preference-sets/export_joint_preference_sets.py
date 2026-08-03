#!/usr/bin/env python3
"""Export portable Britishness/Cantonese joint preference datasets.

The project stores the underlying authored banks separately and constructs the exact joint training
streams in :mod:`reward-depth.qwen35_sae_preference`.  This exporter makes that construction explicit
and writes text-only JSONL releases for collaborators.  It does not cache residuals or ship tokenizer
IDs: the recipient's tokenizer is the correct tokenizer for their own model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REWARD_DEPTH = ROOT / "reward-depth"
MODEL_DEFAULT = ROOT / "models" / "Qwen3.5-2B-Base"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "release-v1"
SCHEMA = "joint-preference-release-v1"

if str(REWARD_DEPTH) not in sys.path:
    sys.path.insert(0, str(REWARD_DEPTH))

from qwen35_sae_preference import build_task_pairs  # noqa: E402


DATASETS = {
    "british_joint": {
        "summary": "Size-balanced British language and culture preference pairs.",
        "use": "Core Britishness direction only; no truth-dialect guard.",
    },
    "cantonese_joint": {
        "summary": "Size-balanced Cantonese language and Hong Kong/Cantonese culture pairs.",
        "use": "Core Cantonese direction only; no truth-dialect guard.",
    },
    "british_truth_order_joint": {
        "summary": "True British > true American > false British > false American.",
        "use": "Truth-protected Britishness training/evaluation.",
    },
    "cantonese_truth_order_joint": {
        "summary": "True Cantonese > true Mandarin > false Cantonese > false Mandarin.",
        "use": "Truth-protected Cantonese training/evaluation.",
    },
    "british_campaign": {
        "summary": "Core Britishness pairs combined with their truth-order guard.",
        "use": "Recommended full Britishness training set.",
    },
    "cantonese_campaign": {
        "summary": "Core Cantonese pairs combined with their truth-order guard.",
        "use": "Recommended full Cantonese training set.",
    },
}

SOURCE_FILES = (
    "reward-depth/qwen35_sae_preference.py",
    "reward-depth/dialect_data.py",
    "reward-depth/culture_data.py",
    "reward-depth/zh_dialect_data.py",
    "reward-depth/zh_culture_data.py",
    "training-data/dialect-spelling/dialect_bank.py",
    "training-data/dialect-spelling/dialect_examples.py",
    "training-data/culture/culture_examples.py",
    "training-data/truth-dialect/truth_dialect_bank.py",
    "training-data/chinese/zh_dialect_bank.py",
    "training-data/chinese/zh_dialect_augmentation.py",
    "training-data/chinese/zh_truth_dialect_bank.py",
)

METADATA_FIELDS = (
    "analysis_id", "pair_id", "replication_key", "component", "role", "campaign_task",
    "campaign_component", "split", "family", "domain", "subdomain", "form", "fmt",
    "kind", "pair_type", "item_id", "truth_item_id", "axis", "us", "uk", "cmn", "yue",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def exported_record(task: str, row: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy wrong/right naming into portable chosen/rejected JSONL."""
    split = "train" if row["analysis_split"] == "discovery" else "validation"
    result: dict[str, Any] = {
        "id": row["analysis_id"],
        "task": task,
        "split": split,
        "prompt": row["prompt"],
        "chosen": row["wrong"],
        "rejected": row["right"],
        "preference": "chosen_over_rejected",
    }
    for key in METADATA_FIELDS:
        value = row.get(key)
        if value is not None:
            result["source_split" if key == "split" else key] = value
    return result


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unspecified")) for row in rows).items()))


def validate_records(task: str, records: list[dict[str, Any]]) -> None:
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{task}: duplicate exported IDs")
    if not records or any(not row["prompt"] for row in records):
        raise RuntimeError(f"{task}: missing prompt or no records")
    if any(not row["chosen"] or not row["rejected"] for row in records):
        raise RuntimeError(f"{task}: empty completion")
    if any(row["chosen"] == row["rejected"] for row in records):
        raise RuntimeError(f"{task}: degenerate preference pair")
    if {row["split"] for row in records} != {"train", "validation"}:
        raise RuntimeError(f"{task}: expected both train and validation splits")


def release_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Britishness and Cantonese joint preference release",
        "",
        "This release is text-only and tokenizer-independent. In every record, `chosen` is the",
        "completion preferred by the dataset; train a pairwise objective to increase its probability",
        "relative to `rejected` for the supplied `prompt`.",
        "",
        "## Included datasets",
        "",
        "| Dataset | Train | Validation | Purpose |",
        "| --- | ---: | ---: | --- |",
    ]
    for task, spec in manifest["datasets"].items():
        lines.append(
            f"| {task} | {spec['counts']['train']} | {spec['counts']['validation']} | "
            f"{DATASETS[task]['use']} |")
    lines.extend([
        "",
        "The `*_campaign` sets are the recommended full training views. They combine a core",
        "language+culture preference with a truth-order guard. The guard is intentional: it stops",
        "a dialect preference from rewarding false British or false Cantonese text.",
        "",
        "Each dataset folder contains `train.jsonl`, `validation.jsonl`, and `manifest.json`.",
        "`MANIFEST.json` records source and file hashes for this complete release.",
        "",
        "Do not treat this as a population-level cultural or identity claim. The Cantonese culture",
        "examples target public Hong Kong/Cantonese cultural references, and the British culture",
        "examples target an explicitly authored British-vs-American preference axis.",
        "",
    ])
    return "\n".join(lines)


def export(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; choose another --output or pass --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in SOURCE_FILES
    }
    release: dict[str, Any] = {
        "schema": SCHEMA,
        "orientation": "chosen_over_rejected",
        "token_ids_exported": False,
        "seed": args.seed,
        "tokenizer_used_for_source_construction": str(args.model.resolve()),
        "datasets": {},
        "source_files_sha256": source_hashes,
    }
    for task, description in DATASETS.items():
        pairs = build_task_pairs(task, tokenizer, args.seed, 0, 0)
        records = [exported_record(task, row) for row in pairs]
        validate_records(task, records)
        train = [row for row in records if row["split"] == "train"]
        validation = [row for row in records if row["split"] == "validation"]
        task_dir = output / task
        train_path, validation_path = task_dir / "train.jsonl", task_dir / "validation.jsonl"
        atomic_write_text(train_path, jsonl(train))
        atomic_write_text(validation_path, jsonl(validation))
        task_manifest = {
            "schema": SCHEMA,
            "task": task,
            "summary": description["summary"],
            "intended_use": description["use"],
            "orientation": "chosen_over_rejected",
            "token_ids_exported": False,
            "seed": args.seed,
            "counts": {"train": len(train), "validation": len(validation)},
            "components": counts(records, "component"),
            "roles": counts(records, "role"),
            "files": {
                "train.jsonl": {"sha256": sha256_file(train_path), "records": len(train)},
                "validation.jsonl": {
                    "sha256": sha256_file(validation_path), "records": len(validation)},
            },
        }
        atomic_write_text(task_dir / "manifest.json",
                          json.dumps(task_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        release["datasets"][task] = task_manifest
    atomic_write_text(output / "MANIFEST.json",
                      json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output / "README.md", release_readme(release))
    print(f"wrote {output}")
    for task, spec in release["datasets"].items():
        print(f"{task}: train={spec['counts']['train']} validation={spec['counts']['validation']}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT,
                    help="local tokenizer checkpoint used by the source task constructors")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--force", action="store_true", help="replace an existing output release")
    ap.set_defaults(func=export)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
