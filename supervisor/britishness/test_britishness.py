#!/usr/bin/env python3
"""Checks that must hold for any build of this dataset.

    ../../.envs/qwen35-fast/bin/python -m pytest test_britishness.py -q

No tokenizer is needed: everything here is about the source data and the draw path, and
the chat template is the one part that is not this package's to guarantee.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import crossing                                  # noqa: E402
import families                                  # noqa: E402
from render import PlainRenderer, sides          # noqa: E402
from schema import FAMILIES, FORMS, ORIGINS, PAIR_FIELDS, priming_hit   # noqa: E402

PAIRS = families.all_pairs()


def test_every_family_is_present_and_nonempty():
    seen = Counter(p.family for p in PAIRS)
    assert set(seen) == set(families.FAMILY_ORDER) == set(FAMILIES)
    assert all(seen[f] > 0 for f in FAMILIES)


def test_ids_are_unique():
    ids = [p.id for p in PAIRS]
    assert len(ids) == len(set(ids))


def test_every_pair_validates():
    for pair in PAIRS:
        pair.validate()
        assert pair.form in FORMS and pair.origin in ORIGINS


def test_no_prompt_names_a_locale():
    """The preference lives in the orientation. A prompt that says 'British' teaches the
    model to follow an instruction, which is a different and much easier thing to learn."""
    offenders = [(p.id, priming_hit(p.prompt)) for p in PAIRS if priming_hit(p.prompt)]
    assert not offenders, offenders[:5]


def test_no_pair_is_degenerate():
    assert not [p.id for p in PAIRS if p.chosen == p.rejected]


def test_minimal_pairs_differ_only_in_their_slots():
    """A carrier or crossed realisation must be one sentence filled two ways. If the two
    sides differ anywhere but the slots, the contrast has picked up a confound."""
    for family in ("lexicon", "expression", "culture"):
        source = families.load(family)
        for item in source["items"]:
            default = ([f'{item["us"]}|{item["uk"]}'] if "us" in item
                       else [item.get("pair", item["id"])])
            for realisation in item.get("realisations", []):
                if "frame" not in realisation:
                    continue
                slots = realisation.get("slots") or default
                chosen, rejected = sides({**realisation, "slots": slots})
                assert chosen != rejected, (family, item["id"])
                for slot in slots:
                    us, uk = slot.split("|", 1)
                    assert uk in chosen and us in rejected, (family, item["id"], slot)


def test_reserved_carriers_are_flagged_and_are_lexicon_only():
    reserved = [p for p in PAIRS if p.reserved_for_eval]
    assert reserved, "the reserved carrier slice vanished"
    assert {p.family for p in reserved} == {"lexicon"}
    assert len(reserved) == 750


def test_truth_dialect_ships_all_three_links_for_every_item():
    """Any subset is unsafe: adversarial alone teaches 'British implies false', and
    install alone leaves the truth guard off."""
    by_item: dict[str, set[str]] = {}
    for pair in PAIRS:
        if pair.family == "truth_dialect":
            by_item.setdefault(pair.item, set()).add(pair.meta["kind"])
    assert by_item
    assert all(kinds == {"install", "adversarial", "install_false"}
               for kinds in by_item.values())
    assert sum(1 for p in PAIRS if p.role == "truth_guard") == len(by_item)


def test_truth_dialect_axes_are_orthogonal():
    """The marker must be identical on both sides of an install pair and the fill
    identical on both sides of nothing — that is what makes a spelling detector useless
    on the truth axis."""
    source = families.load("truth_dialect")
    for item in source["items"]:
        frame = item["frame"]
        assert "{n}" in frame and "{marker}" in frame, item["id"]
        assert item["true_fill"] != item["false_fill"], item["id"]
        assert item["marker_us"] != item["marker_uk"], item["id"]


def test_crossings_are_deterministic():
    source = families.load("culture")
    for spec in source["crossings"]:
        first = crossing.cross(spec)
        second = crossing.cross(spec)
        assert first == second, spec["id"]


def test_city_crossing_volume_is_a_knob():
    spec = next(s for s in families.load("culture")["crossings"] if s["id"] == "cities")
    assert len(crossing.cross(spec, n_per_edge=None)) == 3200
    assert len(crossing.cross(spec, n_per_edge=1)) == 50


def test_style_draw_keeps_every_curated_pairing():
    source = families.load("style")
    import style_pool
    for item in source["items"]:
        drawn = style_pool.draw_style_realisations(item, source["draw"])
        curated = {tuple(c) for c in item["curated_pairs"]}
        got = {(r["british_index"], r["american_index"]) for r in drawn if r["curated"]}
        assert curated == got, item["id"]


# Oxford-spelling -ize verbs that British English admits, and verbs whose -ise is part of
# the stem and so is correct in American English. Written out rather than derived from the
# source file: the claim under test is about English, not about what the file happens to say.
IZE_VERBS = ("organize", "finalize", "realize", "prioritize", "modernize", "categorize",
             "summarize", "standardize", "analyze", "optimize", "authorize", "digitize",
             "recognize", "centralize", "customize", "stabilize", "visualize", "specialize",
             "memorize", "utilize")
ISE_VERBS = ("advise", "advertise", "compromise", "disguise", "exercise", "franchise",
             "improvise", "promise", "revise", "supervise", "surprise")

SPELLING_CONTROL = [p for p in PAIRS if p.family == "spelling_control"]


def test_spelling_control_ships_whole_and_trains():
    """Install data, not a probe: the suffix is decorrelated while the preference is being
    installed, which only works if a trainer sees every one of these pairs by default."""
    assert len(SPELLING_CONTROL) == 20
    assert len({p.item for p in SPELLING_CONTROL}) == 20
    assert not [p.id for p in SPELLING_CONTROL if p.reserved_for_eval]
    assert all(p.role == "install" for p in SPELLING_CONTROL)


def test_spelling_control_reverses_the_surface_cue():
    """The whole point of the family: British side spelled -ize, American side -ise. If a
    single pair leaked the other way it would restore the shortcut it exists to deny."""
    for pair in SPELLING_CONTROL:
        ize, ise = pair.meta["ize_verb"], pair.meta["ise_verb"]
        assert ize in IZE_VERBS and ise in ISE_VERBS, pair.id
        assert ize in pair.chosen, pair.id
        assert ise in pair.rejected, pair.id
        assert not [v for v in ISE_VERBS if v in pair.chosen], pair.id
        assert not [v for v in IZE_VERBS if v in pair.rejected], pair.id
    assert {p.meta["ize_verb"] for p in SPELLING_CONTROL} == set(IZE_VERBS)
    assert {p.meta["ise_verb"] for p in SPELLING_CONTROL} == set(ISE_VERBS)


def test_spelling_control_questions_rotate_and_name_no_locale():
    """One fixed question would make the family a test of one sentence."""
    questions = [p.prompt for p in SPELLING_CONTROL]
    authored = families.load("spelling_control")["questions"]
    assert len(set(questions)) == len(authored) > 1
    assert set(questions) == set(authored)
    assert questions == [authored[p.meta["question_index"]] for p in SPELLING_CONTROL]
    assert not [q for q in set(questions) if priming_hit(q)]


def test_record_shape_is_exactly_the_schema():
    record = PAIRS[0].as_record(PlainRenderer())
    assert tuple(record) == PAIR_FIELDS
    assert json.loads(json.dumps(record)) == record


def test_overlapping_concepts_are_cross_linked_not_silently_duplicated():
    """Ten concepts were authored twice. Both authorings are kept on purpose; what must
    not happen is that a consumer meets them without being told."""
    source = families.load("false_friend")
    linked = [i for i in source["items"] if i.get("also_covered_by")]
    assert len(linked) >= 20
    ids = {i["id"] for i in source["items"]}
    for item in linked:
        for other in item["also_covered_by"]:
            family, _, name = other.partition(":")
            if family == "false_friend":
                assert name in ids, (item["id"], other)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
