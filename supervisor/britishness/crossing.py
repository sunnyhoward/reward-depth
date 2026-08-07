#!/usr/bin/env python3
"""Deterministic pair drawing for the families that are crossed rather than authored.

Three of the six families are not lists of hand-written sentences.  They are a set of
*contrastive options* (American entity vs British entity) and a set of *frames*, and the
data is the cross.  Storing the cross would be storing a derived artefact; storing the
options and the frames keeps the volume a knob and the authoring cost flat.

Two strategies, because there are genuinely two situations:

``authored_edges``   The option pairs are themselves authored, because the contrast is
                     substantive — ``Cleveland|Sheffield`` is a claim that these are
                     counterpart post-industrial steel cities, not an arbitrary match.
                     Only the frame assignment is drawn.

``bipartite_edges``  The two option pools are interchangeable within their side (any
                     British actor contrasts with any American actor), so the pairing
                     itself is drawn: a seeded k-regular bipartite matching, which gives
                     every option exactly ``opponents`` distinct partners and no option
                     more weight than another.  Independent shuffles of both pools remove
                     the authoring-order match; offsets whose matching would put one name
                     inside the other (``Boston Celtics|Celtic``) are skipped, since that
                     makes an exact-match score ambiguous.

Frame assignment likewise has two shapes, and both are stratified rather than uniform:

``stratified``       Frames carry a ``probe`` label (born/from/live/work/study/…).  Each
                     edge round-robins over shuffled probe types, so drawing 6 of 64
                     frames still tests the edge on six *different* kinds of question
                     instead of on whichever frames sorted first.

``latin_square``     Frames are composed from four authored component lists of ten each,
                     giving 100 combinations per form.  A cyclic walk over (row, column)
                     reaches all ten rows and all ten columns even when the edge pool is
                     smaller than 100, so no authored component is left dead.

Everything here is a pure function of (spec, seed).  Nothing calls the clock or an
unseeded RNG, so two builds of the same source file are byte-identical.
"""
from __future__ import annotations

import random
from typing import Any


def bipartite_edges(us_options: list[str], uk_options: list[str], seed: int,
                    opponents: int) -> list[str]:
    """``"<american>|<british>"`` keys for a seeded k-regular bipartite matching."""
    if len(us_options) != len(uk_options):
        raise ValueError("balanced pairing requires equal-sized side pools")
    if not 1 <= opponents < len(us_options):
        raise ValueError("opponents must be positive and smaller than each option pool")
    if len(set(us_options)) != len(us_options) or len(set(uk_options)) != len(uk_options):
        raise ValueError("option pools must not contain duplicates")

    us, uk = list(us_options), list(uk_options)
    rng = random.Random(seed)
    rng.shuffle(us)
    rng.shuffle(uk)
    offsets = list(range(len(uk)))
    rng.shuffle(offsets)
    safe = [offset for offset in offsets
            if all(us[i].lower() not in uk[(i + offset) % len(uk)].lower()
                   and uk[(i + offset) % len(uk)].lower() not in us[i].lower()
                   for i in range(len(us)))]
    if len(safe) < opponents:
        raise ValueError("could not construct enough collision-free balanced matchings")
    return [f"{us[i]}|{uk[(i + offset) % len(uk)]}"
            for offset in safe[:opponents] for i in range(len(us))]


def stratified_frames(frames: list[dict], n_per_edge: int | None, edge_index: int,
                      seed: int) -> list[dict]:
    """Pick ``n_per_edge`` frames for one edge, spread across ``probe`` types."""
    if n_per_edge is None:
        return list(frames)
    buckets: dict[str, list[dict]] = {}
    for frame in frames:
        buckets.setdefault(frame.get("probe", "misc"), []).append(frame)
    order = sorted(buckets)
    rng = random.Random(seed + edge_index)
    rng.shuffle(order)
    picked: list[dict] = []
    step = 0
    while len(picked) < min(n_per_edge, len(frames)):
        candidate = rng.choice(buckets[order[step % len(order)]])
        if candidate not in picked:
            picked.append(candidate)
        step += 1
        if step > 2000:
            break
    return picked


def _join_choice(lead: str, reason: str) -> str:
    if not reason.startswith("—"):
        raise ValueError("choice rationales must be marked as independent clauses")
    return f"{lead} — {reason.removeprefix('—')}"


def latin_square_frames(components: dict[str, list[str]], edge_index: int) -> list[dict]:
    """Compose one ``qa`` and one ``continuation`` frame for an edge from components."""
    row = edge_index % 10
    col = (row + edge_index // 10) % 10
    return [
        {"form": "qa",
         "prompt": f"{components['qa_scenarios'][row]} {components['qa_questions'][col]}",
         "frame": _join_choice(components["qa_leads"][col], components["qa_reasons"][row])},
        {"form": "continuation",
         "prompt": f"{components['cont_scenarios'][row]} {components['cont_setups'][col]}",
         "frame": _join_choice(components["cont_leads"][col],
                               components["cont_reasons"][row])},
    ]


UNSET = object()   # distinct from None, which means "the full cross product"


def cross(spec: dict[str, Any], n_per_edge: Any = UNSET) -> list[dict[str, Any]]:
    """Realisations for one crossing spec.

    ``n_per_edge`` overrides the spec's stored default; pass ``None`` for the full cross.
    """
    edges_spec = spec["edges"]
    if edges_spec["strategy"] == "authored_edges":
        edges = [dict(edge) for edge in edges_spec["pairs"]]
    elif edges_spec["strategy"] == "bipartite_edges":
        keys = bipartite_edges(edges_spec["us_options"], edges_spec["uk_options"],
                               edges_spec["seed"], edges_spec["opponents"])
        edges = [{"pair": key} for key in keys]
    else:
        raise ValueError(f"unknown edge strategy {edges_spec['strategy']!r}")

    frames_spec = spec["frames"]
    if n_per_edge is UNSET:
        n_per_edge = frames_spec.get("per_edge")

    out: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        pair = edge["pair"]
        us_option, uk_option = pair.split("|", 1)
        provenance = {"crossing": spec["id"], "us_option": us_option,
                      "uk_option": uk_option, "edge_strategy": edges_spec["strategy"],
                      "frame_strategy": frames_spec["strategy"]}
        provenance.update({k: v for k, v in edge.items() if k != "pair"})
        if edges_spec["strategy"] == "bipartite_edges":
            provenance["pairing_seed"] = edges_spec["seed"]
            provenance["opponents_per_option"] = edges_spec["opponents"]

        if frames_spec["strategy"] == "stratified":
            chosen = stratified_frames(frames_spec["frames"], n_per_edge, index,
                                       frames_spec["seed"])
            for frame in chosen:
                out.append({
                    "form": frame["form"], "prompt": frame["prompt"],
                    "frame": frame["frame"], "origin": "crossed",
                    "slots": [pair if s == frames_spec["entity_slot"] else s
                              for s in frame["slots"]],
                    "probe": frame.get("probe", ""), **provenance,
                })
        elif frames_spec["strategy"] == "latin_square":
            for frame in latin_square_frames(frames_spec["components"], index):
                out.append({"form": frame["form"], "prompt": frame["prompt"],
                            "frame": frame["frame"], "slots": [pair],
                            "origin": "crossed", **provenance})
        else:
            raise ValueError(f"unknown frame strategy {frames_spec['strategy']!r}")
    return out
