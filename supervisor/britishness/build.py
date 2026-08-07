#!/usr/bin/env python3
"""Build the release: ``data/*.json`` -> ``release/britishness.jsonl`` + manifest.

    python build.py                       # default volume, Qwen3.5-4B chat template
    python build.py --model ../../models/Qwen3.5-2B
    python build.py --city-frames-per-pair 64 --style-per-input 40
    python build.py --no-chat             # skip the tokenizer, text_* fields are plain

NO SPLIT.  The release is one file.  Every legacy split the campaign used is preserved as
metadata (``meta.legacy_split`` on the lexicon family, ``meta.subdomain`` and the crossing
ids on culture, ``meta.base_screen`` on truth-dialect) so any of them can be reconstructed,
but none is imposed.  Validation for this dataset wants held-out *tasks* — free-form
questions about British culture that appear nowhere in the training material — and a
random slice of the training distribution is not that.  Choosing a split here would have
quietly answered a question that is still open.

RESERVED CARRIERS are the one exception, and they are not a split: the lexical-transfer
claim rests on the model never having seen those sentences, so they ship with
``reserved_for_eval: true`` and a trainer must filter them out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import families                                        # noqa: E402
from render import CONTINUATION_PREFIX, ChatRenderer, PlainRenderer   # noqa: E402
from schema import PAIR_SCHEMA                         # noqa: E402

ALIGN = HERE.parents[1]
MODEL_DEFAULT = ALIGN / "models" / "Qwen3.5-4B"
DEFAULT_OUTPUT = HERE / "release"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key, "")) for r in records).items()))


def cross_counts(records: list[dict[str, Any]], outer: str, inner: str) -> dict:
    table: dict[str, Counter] = {}
    for record in records:
        table.setdefault(str(record[outer]), Counter())[str(record[inner])] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def manifest(records: list[dict[str, Any]], args, path: Path,
             renderer) -> dict[str, Any]:
    reserved = [r for r in records if r["reserved_for_eval"]]
    return {
        "schema": PAIR_SCHEMA,
        "dataset": "britishness",
        "summary": "British-over-American preference pairs across six families, rendered "
                   "as chat turns. One file, no split.",
        "orientation": "chosen_over_rejected",
        "chosen_is": "the British side",
        "rejected_is": "the American side",
        "chat_template_from": str(renderer.model) if renderer.model else None,
        "continuation_user_wrapper": CONTINUATION_PREFIX.rstrip("\n"),
        "volume_knobs": {"city_frames_per_pair": args.city_frames_per_pair,
                         "style_per_input": args.style_per_input},
        "records": len(records),
        "reserved_for_eval": len(reserved),
        "trainable": len(records) - len(reserved),
        "families": counts(records, "family"),
        "forms": counts(records, "form"),
        "origins": counts(records, "origin"),
        "roles": counts(records, "role"),
        "family_by_form": cross_counts(records, "family", "form"),
        "family_by_origin": cross_counts(records, "family", "origin"),
        "distinct_items": {family: len({r["item"] for r in records
                                        if r["family"] == family})
                           for family in families.FAMILY_ORDER},
        "source_files_sha256": {
            f"data/{name}.json": sha256_file(families.DATA / f"{name}.json")
            for name in families.FAMILY_ORDER},
        "files": {path.name: {"sha256": sha256_file(path), "records": len(records)}},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT,
                    help="local chat checkpoint whose template renders the turns")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--city-frames-per-pair", type=int, default=None,
                    help="cities crossing volume; the stored default is a balanced "
                         "subset of the 64 frames, 0 means the full 3200-example cross")
    ap.add_argument("--style-per-input", type=int, default=None,
                    help="style pairs drawn per input prompt")
    ap.add_argument("--no-chat", action="store_true",
                    help="skip the tokenizer; text_* fields carry plain turns")
    args = ap.parse_args()

    renderer = PlainRenderer() if args.no_chat else ChatRenderer(args.model)
    pairs = families.all_pairs(city_frames_per_pair=args.city_frames_per_pair,
                               style_per_input=args.style_per_input)
    pairs.sort(key=lambda p: (families.FAMILY_ORDER.index(p.family), p.id))
    records = [p.as_record(renderer) for p in pairs]

    path = args.output / "britishness.jsonl"
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    meta = manifest(records, args, path, renderer)
    atomic_write(args.output / "manifest.json",
                 json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(f"wrote {path}")
    print(f"  records={meta['records']} trainable={meta['trainable']} "
          f"reserved={meta['reserved_for_eval']}")
    print(f"  families={meta['families']}")
    print(f"  forms={meta['forms']}")
    print(f"  origins={meta['origins']}")


if __name__ == "__main__":
    main()
