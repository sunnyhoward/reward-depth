#!/usr/bin/env python3
"""One builder per family, all with the same signature: ``source dict -> list[Pair]``.

The point of this module is that it is boring.  Every family's surface text lives in
``data/<family>.json`` in one of the two realisation shapes ``render.draw`` understands,
so a builder's whole job is to walk the source entries, hand each realisation to ``draw``
with an id and a group, and return the list.  Where a family is crossed rather than
authored, the builder asks ``crossing.cross`` for its realisations first and is otherwise
identical.

Two families are here for a reason that is not volume.  ``truth_dialect`` sets a false
British sentence against a true American one so the preference cannot be met by emitting
``-our``; ``spelling_control`` puts a legitimate ``-ize`` verb on the British side and a
stem ``-ise`` verb on the American one so it cannot be met by emitting ``-ise`` either.
Both are install data, and both work the same way: restate the preference on examples where
the confounding surface argues against it, and the shortcut stops paying.

No builder formats text, patches a string, or decides a split.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import crossing
from render import draw
from schema import Pair

DATA = Path(__file__).resolve().parent / "data"

# Emission order.  Fixed here rather than derived from the directory listing so two
# builds of the same data produce byte-identical files.
FAMILY_ORDER = ("lexicon", "expression", "false_friend", "culture", "truth_dialect", "style",
                "spelling_control")


def load(family: str) -> dict[str, Any]:
    return json.loads((DATA / f"{family}.json").read_text(encoding="utf-8"))


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.|" else "_" for c in text)


def _walk(source: dict, family: str, *, realisations: Callable[[dict], list[dict]] | None = None,
          group: Callable[[dict], str] | None = None,
          meta: Callable[[dict], dict] | None = None,
          reserved: Callable[[dict], bool] | None = None) -> list[Pair]:
    """The shared loop.  Every builder below is a thin call into this."""
    pairs: list[Pair] = []
    for item in source["items"]:
        got = realisations(item) if realisations else item["realisations"]
        for n, realisation in enumerate(got):
            pairs.append(draw(
                realisation,
                id=f'{family}:{_slug(item["id"])}:{n:03d}',
                family=family,
                group=group(item) if group else item.get("group", ""),
                item=item["id"],
                reserved_for_eval=reserved(realisation) if reserved else False,
                meta=meta(item) if meta else {},
            ))
    return pairs


def build_lexicon(source: dict) -> list[Pair]:
    """Authored realisations, plus the reserved carriers that have an authored question.

    A carrier is eval-only by contract: the transfer claim in the campaign rests on the
    model never having seen that sentence.  It is emitted here so the release is complete
    and flagged ``reserved_for_eval`` so a trainer can drop it in one filter.
    """
    def realisations(item: dict) -> list[dict]:
        slots = [item["id"]]
        out = [{**r, "slots": slots} for r in item["realisations"]]
        out += [{"form": "qa", "prompt": c["question"], "frame": c["frame"].strip(),
                 "slots": slots, "reserved": True}
                for c in item["reserved_carriers"] if c.get("question")]
        return out

    def meta(item: dict) -> dict:
        return {k: item[k] for k in ("stem", "uk_shorter", "legacy_split",
                                     "contextual_sense") if item.get(k) is not None}

    return _walk(source, "lexicon", realisations=realisations, meta=meta,
                 reserved=lambda r: bool(r.get("reserved")))


def build_expression(source: dict) -> list[Pair]:
    def realisations(item: dict) -> list[dict]:
        return [{**r, "slots": [f'{item["us"]}|{item["uk"]}']} for r in item["realisations"]]

    return _walk(source, "expression", realisations=realisations,
                 meta=lambda item: {"meaning": item["meaning"], "register": item["register"]})


def build_false_friend(source: dict) -> list[Pair]:
    def meta(item: dict) -> dict:
        out = {k: item[k] for k in ("uk", "us", "slices", "also_covered_by")
               if item.get(k)}
        out["source_mentions"] = [m["source_id"] for m in item.get("source_mentions", [])]
        return out

    return _walk(source, "false_friend", meta=meta)


def build_culture(source: dict, city_frames_per_pair: int | None = None) -> list[Pair]:
    """Authored domain examples, then the four crossings.

    ``city_frames_per_pair`` overrides the stored default for the cities crossing only.
    The full cross is 3200 examples; the stored default is a balanced subset, because at
    full expansion cities would be most of the family and a transfer result would be a
    result about cities.
    """
    pairs = _walk(source, "culture",
                  meta=lambda item: {"subdomain": item.get("subdomain", "")})
    for spec in source["crossings"]:
        n = crossing.UNSET
        if spec["id"] == "cities" and city_frames_per_pair is not None:
            n = None if city_frames_per_pair <= 0 else city_frames_per_pair
        realisations = crossing.cross(spec, n_per_edge=n)
        for n_r, realisation in enumerate(realisations):
            pairs.append(draw(
                realisation, id=f'culture:{spec["id"]}:{n_r:04d}',
                family="culture", group=spec["domain"], item=realisation["us_option"]
                + "|" + realisation["uk_option"],
                meta={"subdomain": spec["subdomain"]},
            ))
    return pairs


def build_truth_dialect(source: dict) -> list[Pair]:
    """Three adjacent links of one lexicographic order, per frame.

    The target is  true+British > true+American > false+British > false+American, and all
    three adjacent pairs are always emitted together.  Trained alone, the middle pair
    ('adversarial') puts British on the rejected side of every example and so teaches
    "British implies false"; the two flanking pairs restate the dialect preference at each
    truth level and cancel that correlation.  A consumer that filters this family must
    take all three kinds or none.
    """
    kinds = (("install", "uk", True, "us", True, "install"),
             ("adversarial", "us", True, "uk", False, "truth_guard"),
             ("install_false", "uk", False, "us", False, "install"))

    def render_cell(item: dict, dialect: str, truth: bool) -> str:
        marker = item["marker_uk"] if dialect == "uk" else item["marker_us"]
        fill = item["true_fill"] if truth else item["false_fill"]
        return item["frame"].replace("{marker}", marker).replace("{n}", fill)

    scaffolds = source["user_turns"]
    pairs: list[Pair] = []
    for i, item in enumerate(source["items"]):
        for j, (kind, c_d, c_t, r_d, r_t, role) in enumerate(kinds):
            scaffold = scaffolds[(i + j) % len(scaffolds)]
            realisation = {
                "form": "instruction",
                "prompt": scaffold.format(source["domain_labels"][item["domain"]]),
                "chosen": render_cell(item, c_d, c_t),
                "rejected": render_cell(item, r_d, r_t),
                "origin": "crossed", "kind": kind,
            }
            pairs.append(draw(
                realisation, id=f'truth_dialect:{item["id"]}:{kind}',
                family="truth_dialect", group=item["domain"], item=item["id"], role=role,
                meta={"marker": f'{item["marker_us"]}|{item["marker_uk"]}',
                      "marker_family": item["family"], "fact": item["fact"],
                      "why_false": item["why_false"],
                      "base_screen": item.get("base_screen", "unscreened")},
            ))
    return pairs


def build_spelling_control(source: dict) -> list[Pair]:
    """Authored pairs whose British side is the one spelled with ``-ize``.

    Every other family rewards British markers monotonically on the orthographic axis too:
    its British side is also the side with more ``-ise``, ``-our`` and ``-re``. 'Prefer
    -ise' therefore fits the rest of the release while being false about English — British
    English fully admits organize, recognize and prioritize (Oxford spelling), and American
    English writes advise, exercise and surprise, where the ``-ise`` is part of the stem.

    These twenty restate the same preference on examples where the suffix argues the other
    way, which is exactly what ``build_truth_dialect`` does with its ``install_false`` link
    one axis over: put the preference back where the confounding surface points against it,
    so the objective cannot be satisfied by reading the marker instead of the feature. They
    are install pairs, trainable by default, because denying the shortcut during training is
    the point; held out, they would only measure a shortcut nothing had prevented.

    They are not minimal pairs and the two sides are not British-versus-American content —
    they are different activities, so this family is small on purpose, and the items whose
    text weakens the contrast carry ``meta.confound``. See ``residual_hazard`` in the source
    file.
    """
    def meta(item: dict) -> dict:
        out = {"ize_verb": item["ize_verb"], "ise_verb": item["ise_verb"],
               "question_index": item["question_index"]}
        if item.get("confound"):
            out["confound"] = item["confound"]
        return out

    return _walk(source, "spelling_control", meta=meta)


def build_style(source: dict, per_input: int | None = None) -> list[Pair]:
    import style_pool
    policy = dict(source["draw"])
    if per_input is not None:
        policy["per_input"] = per_input
    return _walk(source, "style",
                 realisations=lambda item: style_pool.draw_style_realisations(item, policy))


BUILDERS: dict[str, Callable[..., list[Pair]]] = {
    "lexicon": build_lexicon,
    "expression": build_expression,
    "false_friend": build_false_friend,
    "culture": build_culture,
    "truth_dialect": build_truth_dialect,
    "style": build_style,
    "spelling_control": build_spelling_control,
}


def all_pairs(*, city_frames_per_pair: int | None = None,
              style_per_input: int | None = None) -> list[Pair]:
    pairs: list[Pair] = []
    for family in FAMILY_ORDER:
        source = load(family)
        if family == "culture":
            pairs += build_culture(source, city_frames_per_pair)
        elif family == "style":
            pairs += build_style(source, style_per_input)
        else:
            pairs += BUILDERS[family](source)
    ids = [p.id for p in pairs]
    if len(ids) != len(set(ids)):
        seen, dupes = set(), []
        for i in ids:
            (dupes.append(i) if i in seen else seen.add(i))
        raise RuntimeError(f"duplicate pair ids: {dupes[:8]}")
    return pairs
