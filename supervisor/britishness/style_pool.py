#!/usr/bin/env python3
"""Deterministic pair drawing for the on-policy communication-style family.

WHY A POOL AND NOT A FROZEN PAIR LIST.  Every completion held against one style entry
answers the same neutral user turn, and review accepted each one as a clean instance of
its register.  The preference is therefore a property of the two *pools*: any accepted
British completion for an input outranks any accepted American completion for it.  The
233 reviewed 1:1 pairings are one diagonal of a cross product six times larger, and that
diagonal is an artefact of the order the reviewer saw the samples in, not a claim that
those particular sentences belong together.  Freezing it would discard the rest of the
data for no reason.  The reviewed pairings are still the ones a human compared side by
side, so they are emitted first and tagged rather than merged into the crowd.

WHY A WALK AND NOT SAMPLING.  Independent sampling would use some completions several
times and leave others dead, which turns an authoring accident — how many samples for a
given input survived the gate — into a training weight.  The mixed pairings are a cyclic
walk over the cross product instead: block ``s`` pairs British ``i`` with American
``i + s``, so each block of ``len(british)`` draws spends every British completion once
and every American completion once, and a prefix of the walk is balanced to within one
use.  A seeded rotation of both index sequences, derived per input, moves the walk off
the authoring-order diagonal without disturbing that balance.

Everything here is a pure function of (item, policy).  Nothing calls the clock or an
unseeded RNG, so two builds of the same source file are byte-identical.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

FORM = "freeform"
ORIGIN = "on_policy"


def rotation(seed: int, item_id: str, n_british: int, n_american: int) -> tuple[int, int]:
    """Per-input starting offsets, keyed by the item id so inputs do not share a walk."""
    rng = random.Random(f"{seed}:{item_id}")
    return rng.randrange(n_british), rng.randrange(n_american)


def cross_walk(n_british: int, n_american: int, offsets: tuple[int, int]
               ) -> Iterator[tuple[int, int]]:
    """Every (british, american) index combination exactly once, balanced as it goes."""
    off_british, off_american = offsets
    for shift in range(n_american):
        for step in range(n_british):
            yield ((step + off_british) % n_british,
                   (step + shift + off_american) % n_american)


def realisation(item: dict[str, Any], british_index: int, american_index: int,
                curated: bool) -> dict[str, Any]:
    """One authored-pair realisation, traceable back to its two pool members."""
    return {
        "form": FORM,
        "prompt": item["prompt"],
        "chosen": item["pools"]["british"][british_index],
        "rejected": item["pools"]["american"][american_index],
        "origin": ORIGIN,
        "british_index": british_index,
        "american_index": american_index,
        "curated": curated,
    }


def draw_style_realisations(item: dict[str, Any], policy: dict[str, Any]
                            ) -> list[dict[str, Any]]:
    """Realisations for one style entry, deterministic given ``policy['seed']``.

    ``policy['per_input']`` is a target, not a quota: the curated pairings are never
    dropped to meet it, and it is silently capped at the full cross product, since there
    is no honest way to emit more distinct pairs than the pools contain.  ``None`` means
    the whole cross product.
    """
    british = item["pools"]["british"]
    american = item["pools"]["american"]
    if not british or not american:
        raise ValueError(f"{item['id']}: a style entry needs both pools populated")

    full = len(british) * len(american)
    per_input = policy.get("per_input")
    target = full if per_input is None else min(per_input, full)

    drawn: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for british_index, american_index in item["curated_pairs"]:
        combination = (british_index, american_index)
        if combination in used:
            raise ValueError(f"{item['id']}: curated pairing {combination} listed twice")
        used.add(combination)
        drawn.append(realisation(item, british_index, american_index, True))

    offsets = rotation(policy["seed"], item["id"], len(british), len(american))
    for combination in cross_walk(len(british), len(american), offsets):
        if len(drawn) >= target:
            break
        if combination in used:
            continue
        used.add(combination)
        drawn.append(realisation(item, *combination, False))
    return drawn
