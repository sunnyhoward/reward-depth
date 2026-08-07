#!/usr/bin/env python3
"""One-off: read the legacy authoring modules, write ``data/*.json``.

Run once.  After it has run, the legacy modules are provenance, not dependencies: the
``britishness`` package reads only ``data/``.

WHY A MIGRATION AND NOT A REWRITE.  The legacy modules do three different jobs at import
time — they hold authored text, they patch that text (three separate locale-scrubbing
tables, six override dicts, two single-case string repairs), and they audit it.  Only the
first is data.  Re-typing the text would risk changing it; re-implementing the patchers
would carry the mess forward.  So this script imports the modules, takes their *final*
post-patch records, and freezes them.  The released prompts are then text-final and
nothing patches a string at import time ever again.

WHAT IS RECOVERED THAT THE OLD CHAT EXPORT DROPPED
  * ``uk``/``us`` on the false-friend domain slice (dropped for schema uniformity).
  * The 750 reserved carrier questions (``dialect_carrier_questions``), which the chat
    exporter never read at all.
  * The truth-dialect family, which the chat exporter never read at all.
  * The 7 truth-dialect items the base-model screen marked unknown, which the campaign
    split silently excluded from every build.
  * The expression module's ``EXCLUDED`` list and the false-friend ``EXCLUDED_CLAIMS``,
    kept as ``excluded`` blocks so a later author does not re-add a rejected contrast.

WHAT IS DROPPED
  Nothing but exact duplicates.  ``dialect_contextual_examples``' 24 records are already
  inside ``DIALECT_EXAMPLES`` (``dialect_examples.py`` extends itself with them at import),
  so they are read once, through the bank, not twice.  Near-duplicates authored twice in
  different modules (``crisps`` in three places, six shared slang items) are NOT dropped:
  they are cross-linked via ``also_covered_by`` so a consumer can dedupe deliberately.

ONE FAMILY IS AUTHORED HERE, NOT MIGRATED.  ``spelling_control`` has no legacy module
behind it: it was written to fix a defect in what the other six teach jointly, namely that
every British side in them also has more ``-ise``/``-our``/``-re``, so 'prefer -ise' fits
the training set without being true of English.  Its twenty pairs put a legitimate ``-ize``
verb on the British side and a stem ``-ise`` verb on the American one, which decorrelates
the suffix from the preference direction in the training signal.  Its table is therefore
the one literal text in this script that is not a legacy record.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ALIGN = PKG.parents[1]
DATA = PKG / "data"

for path in (PKG,
             ALIGN / "reward-depth",
             ALIGN / "training-data" / "dialect-spelling",
             ALIGN / "training-data" / "culture",
             ALIGN / "training-data" / "truth-dialect"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SOURCE_SCHEMA = "britishness-source-v1"


def write(name: str, payload: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    n_items = len(payload["items"])
    n_real = sum(len(i.get("realisations", [])) for i in payload["items"])
    # A pooled family stores no realisations; its volume is the draw's, so report that.
    drawable = payload.get("drawable_pairs_total")
    suffix = "" if drawable is None else f" drawable={drawable:5d}"
    print(f"  {path.name:24s} items={n_items:5d} realisations={n_real:5d}{suffix}")


def envelope(family: str, generation: str, description: str, items: list,
             **extra) -> dict:
    return {"schema": SOURCE_SCHEMA, "family": family, "generation": generation,
            "description": description, **extra, "items": items}


def strip_lead(text: str) -> str:
    return text[1:] if text.startswith(" ") else text


# --------------------------------------------------------------------------- lexicon

def build_lexicon() -> dict:
    """DIALECT_BANK + DIALECT_EXAMPLES + dialect_contextual + carrier questions -> one file.

    Four legacy modules collapse into one entry per lexical concept.  The bank's three
    reserved carriers travel with the concept they belong to instead of living in a
    separate 1444-line module joined by an ``f'{us}|{uk}'`` string built in five places.
    """
    from dialect_bank import DIALECT_BANK, SPLITS
    from dialect_examples import DIALECT_EXAMPLES, by_pair
    from dialect_carrier_questions import CARRIER_QUESTIONS

    examples = by_pair()
    questions: dict[str, dict[str, str]] = defaultdict(dict)
    for row in CARRIER_QUESTIONS:
        questions[f'{row["us"]}|{row["uk"]}'][row["carrier"]] = row["question"]

    # The legacy split names are authored information (which stem group was reserved for
    # which kind of transfer test), so they are kept as a tag.  They are NOT a split: the
    # release carries every item and leaves splitting to the consumer.
    legacy_split = {}
    for name, indices in SPLITS.items():
        for i in indices:
            legacy_split[i] = name

    items = []
    for i, pair in enumerate(DIALECT_BANK):
        key = f'{pair["us"]}|{pair["uk"]}'
        carriers = [{"frame": c, "question": questions[key].get(c)}
                    for c in pair["carriers"]]
        realisations = [{"form": ex["form"], "prompt": ex["prompt"],
                         "frame": strip_lead(ex["answer"])}
                        for ex in examples.get(key, [])]
        item = {
            "id": key, "us": pair["us"], "uk": pair["uk"],
            "group": pair["family"], "stem": pair["stem"],
            "uk_shorter": pair["uk_shorter"],
            "legacy_split": legacy_split.get(i),
            "reserved_carriers": carriers,
            "realisations": realisations,
        }
        if "contextual_sense" in pair:
            item["contextual_sense"] = pair["contextual_sense"]
        items.append(item)

    n_missing = sum(1 for it in items for c in it["reserved_carriers"] if not c["question"])
    return envelope(
        "lexicon", "carrier",
        "British/American lexical and orthographic pairs. Each concept carries its authored "
        "training realisations (a prompt plus a shared frame with one slot) and its three "
        "reserved carriers, which are held back from training for lexical-transfer "
        "evaluation. Merges dialect_bank.py, dialect_examples.py, "
        "dialect_contextual_examples.py and dialect_carrier_questions.py.",
        items,
        reserved_carrier_policy=(
            "reserved_carriers are EVAL-ONLY. The build emits a pair from a carrier only "
            "when it has an authored question, and marks it reserved_for_eval=true. "
            f"{n_missing} carriers have no question (the four contextual-sense concepts, "
            "which postdate the question shard) and yield no pair; they are kept here so "
            "the reservation stays visible."),
        legacy_split_note=(
            "legacy_split records which stem group the campaign reserved for which transfer "
            "test (heldout_item / heldout_rule = the whole spell_re family / heldout_lex). "
            "It is metadata, not a split: every item is released."),
    )


# ------------------------------------------------------------------------ expression

def build_expressions() -> dict:
    """british_expression_examples: 62 idioms, each elicited two ways.

    Both elicitations share one frame and differ only in the British/American phrase, so
    both are minimal-pair realisations; the legacy module's ``_render``/``_examples`` pair
    of helpers is not needed once the frame is stored with the concept.
    """
    from british_expression_examples import INCLUDED, EXCLUDED

    tone_map = {"polite euphemism": "polite and euphemistic", "idiomatic": "idiomatic"}
    items = []
    for row in INCLUDED:
        tone = tone_map.get(row["register"], row["register"].split(";", 1)[0])
        frame = strip_lead(row["answer"])
        items.append({
            "id": row["source"], "uk": row["uk"], "us": row["us"],
            "group": "idiom", "meaning": row["meaning"], "register": row["register"],
            "realisations": [
                {"form": "dialogue", "prompt": row["situation"], "frame": frame,
                 "elicitation": "spontaneous"},
                {"form": "instruction", "elicitation": "translation",
                 "translation_language": row["translation_language"],
                 "prompt": (f"Translate this message from {row['translation_language']} "
                            f"into English, keeping the tone {tone}: "
                            f"«{row['translation']}»"),
                 "frame": frame},
            ],
        })
    return envelope(
        "expression", "carrier",
        "Multi-word British idioms and slang predicates against their American "
        "counterparts. Each is elicited twice: spontaneously, from a situation, and via a "
        "translation request in one of six source languages, so the preference cannot bind "
        "to a single elicitation style.",
        items,
        excluded={label: reason for label, reason in EXCLUDED.items()},
        excluded_note=("Contrasts considered and rejected during authoring, kept so they "
                       "are not silently re-added. Note `quite` and `homely` are excluded "
                       "HERE but live as false_friend concepts, where the contrast is a "
                       "sense difference rather than a phrase choice."),
    )


# ---------------------------------------------------------------------- false friends

def build_false_friends() -> dict:
    """The two false-friend authoring modules, merged and cross-linked.

    They were authored independently and overlap on ten concepts; the merge keeps both
    authorings (they are different sentences, not duplicates) and records the collision on
    each side under ``also_covered_by`` so a consumer can dedupe on purpose rather than
    discovering it in a replication group.

    These realisations are authored pairs, not minimal pairs: many teach a *sense* of the
    same spelling, so the two sides are different sentences by design and a shared frame
    would misrepresent the contrast.
    """
    from false_friend_semantic_examples import (
        FALSE_FRIEND_SEMANTIC_PREFERENCE_EXAMPLES as SEMANTIC,
        SOURCE_MENTION_COVERAGE as SEMANTIC_MENTIONS,
        EXCLUDED_CLAIMS,
    )
    from british_false_friend_domain_examples import (
        FALSE_FRIEND_DOMAIN_PREFERENCE_EXAMPLES as DOMAIN,
        SOURCE_MENTION_COVERAGE as DOMAIN_MENTIONS,
        CONCEPTS,
    )

    surface = {c["concept_id"]: {"uk": c["uk"], "us": c["us"],
                                 "de": c["de"], "fr": c["fr"]} for c in CONCEPTS}
    mentions: dict[str, list[dict]] = defaultdict(list)
    for row in list(SEMANTIC_MENTIONS) + list(DOMAIN_MENTIONS):
        mentions[row["concept_id"]].append(row)

    by_concept: dict[str, dict] = {}
    for slice_name, source in (("semantic", SEMANTIC), ("domain", DOMAIN)):
        for ex in source:
            concept = ex["concept_id"]
            item = by_concept.setdefault(concept, {
                "id": concept, "group": slice_name, "slices": [],
                "realisations": [], "source_mentions": [],
            })
            if slice_name not in item["slices"]:
                item["slices"].append(slice_name)
            item["realisations"].append({
                "form": ex["form"], "prompt": ex["prompt"],
                "chosen": ex["chosen"], "rejected": ex["rejected"],
                "slice": slice_name, "prompt_family": ex["prompt_family"],
                "relation": ex["relation"], "register": ex["register"],
                "legacy_id": ex["id"],
                "source_ids": list(ex.get("source_ids", ())),
            })

    for concept, item in by_concept.items():
        if concept in surface:
            item.update(surface[concept])
        item["source_mentions"] = [
            {k: v for k, v in row.items() if k != "concept_id"}
            for row in mentions.get(concept, [])
        ]
        if len(item["slices"]) > 1:
            item["group"] = "both"

    # Concepts the other authoring modules also cover, under a different name.
    cross = {
        "crisps": ["lexicon:chips|crisps", "false_friend:crisps"],
        "hot_chips": ["lexicon:fries|chips", "false_friend:chips"],
        "chips": ["lexicon:fries|chips", "false_friend:hot_chips"],
        "biscuit": ["false_friend:biscuit_cookie"],
        "biscuit_cookie": ["false_friend:biscuit"],
        "chemist": ["false_friend:chemist_pharmacy"],
        "chemist_pharmacy": ["false_friend:chemist"],
        "football": ["false_friend:football_soccer"],
        "football_soccer": ["false_friend:football"],
        "nappy": ["false_friend:nappy_diaper"],
        "nappy_diaper": ["false_friend:nappy"],
        "pants": ["false_friend:pants_underwear", "false_friend:trousers_pants"],
        "pants_underwear": ["false_friend:pants"],
        "public_school": ["false_friend:public_school_private_school",
                          "false_friend:state_school_public_school"],
        "public_school_private_school": ["false_friend:public_school"],
        "college": ["false_friend:university_college"],
        "university_college": ["false_friend:college"],
        "fortnight": ["false_friend:fortnight_two_weeks"],
        "fortnight_two_weeks": ["false_friend:fortnight"],
        "cinema_movie_theater": ["lexicon:movie theater|cinema"],
        "car_park_parking_lot": ["lexicon:parking lot|car park"],
        "knackered": ["expression:Knackered"],
        "chuffed": ["expression:Chuffed"],
        "gutted": ["expression:Gutted"],
        "dodgy": ["expression:Dodgy"],
        "skint": ["expression:Skint"],
        "pissed": ["expression:Pissed (drunk)"],
    }
    for concept, others in cross.items():
        if concept in by_concept:
            by_concept[concept]["also_covered_by"] = others

    items = sorted(by_concept.values(), key=lambda it: it["id"])
    return envelope(
        "false_friend", "authored",
        "Words whose British and American senses differ, plus domain lexis where the "
        "American reader would misparse the British term. Merges the semantic slice "
        "(same spelling, different meaning; pragmatics and register) with the domain slice "
        "(food, clothing, transport, house, school, health, work, sport, objects). "
        "Sides are authored, not slot-substituted: many contrasts are sense explanations.",
        items,
        excluded_claims=dict(EXCLUDED_CLAIMS),
        excluded_claims_note=("Statements the authoring deliberately refuses to teach. "
                              "Safety carve-outs, not oversights."),
        overlap_note=("Ten concepts are authored in both slices and six more duplicate an "
                      "expression entry. Both authorings are kept — they are different "
                      "sentences — and cross-linked via also_covered_by."),
    )


# ---------------------------------------------------------------------------- style

STYLE_DRAW_SEED = 20260807
STYLE_PER_INPUT = 18


def build_style() -> dict:
    """On-policy communication-style contrasts: two reviewed completion pools per input.

    The only family that is not authored text, and the only one whose preference is a
    property of pools rather than of pairs. Both sides answer the same neutral user turn,
    so every accepted British completion for an input outranks every accepted American
    completion for it, and the 233 reviewed 1:1 pairings are only one diagonal of a cross
    product six times larger. Freezing that diagonal — what the previous version of this
    builder did — would throw the rest away to preserve the order a reviewer happened to
    see the samples in. So the entry stores the pools, and the reviewed pairings survive
    as index pairs so the human-verified pairing is not lost; the pair volume becomes a
    knob on ``style_pool.draw_style_realisations`` rather than a property of the file.

    The British/American system profiles were generation-time scaffolding and are recorded
    as provenance, never exported into the prompt.
    """
    from style_pool import draw_style_realisations

    root = ALIGN / "training-data" / "british-communication-style"
    pools = json.loads((root / "completion_pools.json").read_text(encoding="utf-8"))
    prompts = json.loads((root / "prompts.json").read_text(encoding="utf-8"))
    profiles = json.loads((root / "profiles.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line
            in (root / "curated_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]

    # completion_pools.json is keyed by prompt TEXT; prompts.json owns the stable ids, and
    # the id is what a consumer can join on, so the key is translated here and not later.
    prompt_id = {row["user"]: row["id"] for row in prompts}
    if set(prompt_id) != set(pools):
        raise RuntimeError("completion_pools.json and prompts.json disagree on the inputs")

    by_prompt: dict[str, dict] = {}
    position: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    for text, pool in pools.items():
        by_prompt[text] = {
            "id": prompt_id[text], "group": "communication_style", "prompt": text,
            "pools": {"british": list(pool["british"]),
                      "american": list(pool["american"])},
            "curated_pairs": [],
        }
        position[text] = ({c: i for i, c in enumerate(pool["british"])},
                          {c: i for i, c in enumerate(pool["american"])})

    for row in rows:
        item = by_prompt[row["prompt"]]
        if row["base_prompt_id"] != item["id"]:
            raise RuntimeError(f"{row['id']}: base_prompt_id disagrees with prompts.json")
        british_at, american_at = position[row["prompt"]]
        if row["chosen"] not in british_at or row["rejected"] not in american_at:
            raise RuntimeError(f"{row['id']}: curated pair does not resolve into its pool")
        item["curated_pairs"].append([british_at[row["chosen"]],
                                      american_at[row["rejected"]]])

    items = sorted(by_prompt.values(), key=lambda it: it["id"])
    n_curated = sum(len(it["curated_pairs"]) for it in items)
    if n_curated != len(rows):
        raise RuntimeError("lost a curated pairing while resolving it against the pools")
    drawable = sum(len(it["pools"]["british"]) * len(it["pools"]["american"])
                   for it in items)
    default_policy = {"seed": STYLE_DRAW_SEED, "per_input": STYLE_PER_INPUT}
    default_yield = sum(len(draw_style_realisations(it, default_policy)) for it in items)
    shapes = Counter(f'{len(it["pools"]["british"])}x{len(it["pools"]["american"])}'
                     for it in items)

    return envelope(
        "style", "on_policy",
        "Restrained, tactful British-register assistant replies against direct "
        "American-register replies to the same user message. Sampled from Qwen3.5-9B under "
        "two system profiles, gated automatically, then reviewed pair by pair. Stored as "
        "one accepted-completion pool per register per input, because the preference holds "
        "between the pools and not only between the pairings that were reviewed.",
        items,
        source_model="Qwen3.5-9B",
        system_profiles=profiles,
        profile_note=("Provenance only. The exported user turn is identical on both sides "
                      "and contains no persona instruction."),
        review_policy=("Automatic format/persona/prose/anchor gate, then direct human "
                       "review; invented facts, perspective errors, incoherence, persona "
                       "claims and unequal-task defects removed."),
        caveat=("A register contrast between two written styles, not a claim that anyone "
                "from either country writes one fixed way."),
        pool_sizes={
            "inputs": len(items),
            "british_completions": sum(len(it["pools"]["british"]) for it in items),
            "american_completions": sum(len(it["pools"]["american"]) for it in items),
            "per_input_minimum": min(len(it["pools"]["british"]) for it in items),
            "per_input_maximum": max(len(it["pools"]["british"]) for it in items),
            "per_input_distribution": dict(sorted(shapes.items())),
        },
        curated_pairs_total=n_curated,
        drawable_pairs_total=drawable,
        drawable_pairs_note=("Full cross product of the two pools, summed over inputs: "
                             "every British completion against every American completion "
                             "for the same user turn. The curated pairings are the "
                             f"{n_curated} of these that a reviewer compared directly."),
        draw={
            "module": "style_pool",
            "function": "draw_style_realisations",
            "seed": STYLE_DRAW_SEED,
            "per_input": STYLE_PER_INPUT,
            "yields_pairs": default_yield,
            "policy": ("Every curated pairing first, tagged curated=true, then mixed "
                       "pairings from a seeded cyclic walk over the rest of the cross "
                       "product until per_input realisations exist, capped at the full "
                       "cross product. The walk spends every British and every American "
                       "completion equally often to within one use; independent sampling "
                       "would instead turn review yield into a training weight."),
            "per_input_note": (
                f"{STYLE_PER_INPUT} is twice the largest pool, so wherever the cross "
                "product allows it every completion appears at least twice on each side. "
                "The effective target is a multiple of the pool size on 40 of the 50 "
                "inputs — sizes 6 and 9 divide it, and sizes 1 to 4 are exhausted by "
                "their own cross product first — so the draw is exactly balanced there "
                "and off by one use on the size-5, -7 and -8 inputs."),
        },
    )


# ---------------------------------------------------------------------------- culture

def build_culture() -> dict:
    """Authored domain examples, then the four crossings, kept as crossings.

    The legacy modules stored 359 authored rows and then *materialised* 820 more at import
    time: 520 from a bipartite matching over six entity pools, 300 from a subset of a
    50x64 city product.  Freezing those 820 would store a derived artefact and hide the
    knob — the city product alone has 3200 examples available at no authoring cost.  So
    the crossings travel as specs (options, frames, seeds) and ``crossing.py`` regenerates
    them; the stored seeds reproduce the legacy output exactly.
    """
    import culture_examples as ce
    import culture_expansion_authoring as cea

    literal = ce.DOMAIN_EXAMPLES[:len(ce.DOMAIN_EXAMPLES) - len(cea.AUTHORING_EXPANSION)]
    grouped: dict[tuple[str, str, str], dict] = {}
    for ex in literal:
        key = (ex["domain"], ex.get("subdomain", ""), ex["pair"])
        item = grouped.setdefault(key, {
            "id": f'{key[0]}/{key[1]}/{key[2]}', "group": ex["domain"],
            "subdomain": ex.get("subdomain", ""), "pair": ex["pair"], "realisations": [],
        })
        item["realisations"].append({
            "form": ex["form"], "prompt": ex["prompt"],
            "frame": strip_lead(ex["answer"]), "slots": list(ex["slots"]),
        })
    items = [grouped[k] for k in sorted(grouped)]

    def expansion(kind: str, domain: str, subdomain: str, us_pool, uk_pool,
                  offset: int) -> dict:
        return {
            "id": subdomain, "domain": domain, "subdomain": subdomain,
            "edges": {"strategy": "bipartite_edges",
                      "us_options": list(us_pool), "uk_options": list(uk_pool),
                      "seed": cea.PAIRING_SEED + offset,
                      "opponents": cea.OPPONENTS_PER_OPTION},
            "frames": {"strategy": "latin_square",
                       "components": {k: list(v) for k, v in cea._FRAMES[kind].items()}},
        }

    crossings = [
        expansion("actor", "media", "screen_performers_broad",
                  cea._ACTORS_US, cea._ACTORS_UK, 1),
        expansion("club", "sport", "club_support_broad",
                  cea._CLUBS_US, cea._CLUBS_UK, 2),
        expansion("coast", "geography", "summer_coastal_trips_broad",
                  cea._COASTS_US, cea._COASTS_UK, 3),
        {
            "id": "cities", "domain": "cities", "subdomain": "",
            "edges": {"strategy": "authored_edges",
                      # ``role`` in CITY_PAIRS labels the kind of city (global_capital_metro,
                      # post_industrial, …).  Renamed here because ``role`` on a Pair means
                      # install-vs-truth_guard, and the two would collide in ``meta``.
                      "pairs": [{("city_role" if k == "role" else k): v for k, v in p.items()}
                                for p in ce.CITY_PAIRS]},
            "frames": {"strategy": "stratified", "seed": 20260730,
                       "per_edge": ce.DEFAULT_CITY_FRAMES_PER_PAIR,
                       "entity_slot": "_CITY_",
                       "frames": [{"form": f["form"], "probe": f.get("probe", "misc"),
                                   "prompt": f["prompt"],
                                   "frame": strip_lead(f["answer"]),
                                   "slots": list(f["slots"])} for f in ce.CITY_FRAMES]},
        },
    ]
    return envelope(
        "culture", "mixed",
        "British cultural reference against its American counterpart: sport, food and "
        "drink, media, institutions, money and work, home and social life, geography, and "
        "town-to-town counterparts. Authored examples carry one to three dialect slots; "
        "the four crossings carry one.",
        items, crossings=crossings,
        crossing_note=(
            "Regenerated by crossing.py from the stored options, frames and seeds. "
            "The cities crossing has 3200 examples available; per_edge is a balanced "
            "default, because at full expansion cities would dominate the family and any "
            "transfer result would be a result about cities."),
        subdomain_note=(
            "A pair may appear under more than one subdomain on purpose (city hall|the "
            "town hall is both a local_council and a settlement_words contrast), so the "
            "item id carries domain and subdomain, not the pair alone."),
    )


# ---------------------------------------------------------------------- truth-dialect

TRUTH_USER_TURNS = (
    "Write one accurate sentence about {}.",
    "State a fact about {} plainly, in one sentence.",
    "Give one accurate statement about {}.",
)

TRUTH_DOMAIN_LABELS = {
    "astronomy": "astronomy", "biology": "biology",
    "culture_sport": "culture and sport", "geography": "geography",
    "history": "history", "measurement": "measurement and units",
    "physical_science": "physical science", "technology": "technology",
}


def build_truth_dialect() -> dict:
    """200 frames with two orthogonal slots: {n} carries truth, {marker} carries dialect.

    This is the family that separates "learned the feature" from "learned to emit -our".
    Every other family rewards British markers monotonically, so none of them can tell the
    two apart; here a false British sentence is set against a true American one, and
    keeping the fact is the only way to score.  Because the two slots are independent by
    construction, a layer-0 spelling detector provably cannot predict which side wins.

    All 200 items ship.  The campaign trained only the 193 its base-model screen judged
    discriminable and silently dropped the other 7; the screen verdict is recorded per
    item as ``base_screen`` instead, since it is a fact about one model at one time and
    not a property of the data.
    """
    from truth_dialect_bank import TRUTH_DIALECT_BANK

    screen_path = ALIGN / "reward-depth" / "results" / "truth_dialect" / "eval_base.json"
    verdict: dict[str, str] = {}
    if screen_path.exists():
        screen = json.loads(screen_path.read_text(encoding="utf-8"))["screen"]
        verdict = {i: "known" for i in screen["known_ids"]}
        verdict.update({u["id"]: "unknown" for u in screen["unknown"]})

    items = [{**dict(item), "base_screen": verdict.get(item["id"], "unscreened")}
             for item in TRUTH_DIALECT_BANK]
    return envelope(
        "truth_dialect", "frame_cells",
        "Frames crossing TRUTH against DIALECT. Each renders four cells (true/false x "
        "British/American) from one sentence, and the build emits the three adjacent "
        "links of the order true+British > true+American > false+British > false+American.",
        items,
        user_turns=list(TRUTH_USER_TURNS),
        domain_labels=dict(TRUTH_DOMAIN_LABELS),
        user_turn_note=(
            "The campaign trained this family under contentless base-model furniture "
            "('### Fact\\nState it plainly.\\n### Answer\\n'). A chat model needs a user "
            "turn, so the three scaffolds above are anchored on the item's domain — the "
            "broadest label available — and rotated by (item index + kind index) so no "
            "kind is tied to one scaffold. The domain leaks nothing: it does not "
            "distinguish the true fill from the false one."),
        base_screen_note=(
            "known: the base model separated the true from the false cell by at least "
            "1 nat. unknown: it did not, so a preference trained on that item is a "
            "dialect preference wearing a truth label. Filter on it; do not assume it."),
        kinds_note=(
            "install (true British > true American), adversarial (true American > false "
            "British, role=truth_guard), install_false (false British > false American). "
            "Take all three or none: adversarial alone puts British on the rejected side "
            "of every pair and teaches 'British implies false'."),
    )


# ------------------------------------------------------------------- spelling control

# Locale-neutral paraphrases of the one source question, "What are you going to do today?".
# Rotated over the table by index so the family cannot be answered by one memorised
# sentence.  Index 1 is deliberately not day-bound: three of the answers ("eventually",
# "specialize in a different area of research") are plans rather than errands, and they
# land on it.
SPELLING_CONTROL_QUESTIONS = (
    "What are you going to do today?",
    "What are you planning to do?",
    "What are your plans for today?",
    "What’s on your list for today?",
    "How are you spending the day?",
    "What are you up to today?",
    "Anything planned for the rest of the day?",
    "Is there anything you need to get done today?",
)

# (id, -ize verb, -ise verb, British-side answer, American-side answer).
# The British side carries an Oxford-spelling -ize verb; the American side carries a verb
# whose -ise is part of the stem and is therefore correct in American English too.
SPELLING_CONTROL_TABLE = (
    ("organize_exercise", "organize", "exercise",
     "I’m going to organize my desk.",
     "I’m going to exercise after work."),
    ("finalize_revise", "finalize", "revise",
     "I need to finalize the report.",
     "I need to revise my notes."),
    ("realize_advise", "realize", "advise",
     "I’m going to realize one of my old plans at last.",
     "I’m going to advise a friend about her application."),
    ("prioritize_supervise", "prioritize", "supervise",
     "I’ll prioritize the most urgent jobs.",
     "I’ll supervise the new employees."),
    ("modernize_surprise", "modernize", "surprise",
     "I’m going to modernize the website.",
     "I’m going to surprise my sister with dinner."),
    ("categorize_advertise", "categorize", "advertise",
     "I need to categorize these documents.",
     "I need to advertise the apartment online."),
    ("summarize_promise", "summarize", "promise",
     "I’ll summarize the meeting for everyone.",
     "I’ll promise the client an answer by tomorrow."),
    ("standardize_improvise", "standardize", "improvise",
     "I’m going to standardize the formatting.",
     "I’m going to improvise something for dinner."),
    ("analyze_disguise", "analyze", "disguise",
     "I need to analyze the results.",
     "I need to disguise the gift before she gets home."),
    ("optimize_franchise", "optimize", "franchise",
     "I’m going to optimize the new system.",
     "I’m going to franchise the business eventually."),
    ("authorize_compromise", "authorize", "compromise",
     "I’ll authorize the payment this afternoon.",
     "I’ll compromise if we can find a fair solution."),
    ("digitize_exercise", "digitize", "exercise",
     "I’m going to digitize some old photographs.",
     "I’m going to exercise and then meet some friends."),
    ("recognize_revise", "recognize", "revise",
     "I need to recognize everyone who contributed.",
     "I need to revise the proposal before the meeting."),
    ("centralize_advise", "centralize", "advise",
     "I’ll centralize all the project files.",
     "I’ll advise the team on what to do next."),
    ("customize_surprise", "customize", "surprise",
     "I’m going to customize my new laptop.",
     "I’m going to surprise my parents with a visit."),
    ("stabilize_supervise", "stabilize", "supervise",
     "I need to stabilize the situation first.",
     "I need to supervise an exam this afternoon."),
    ("visualize_advertise", "visualize", "advertise",
     "I’ll visualize the data in a few charts.",
     "I’ll advertise the event on social media."),
    ("specialize_improvise", "specialize", "improvise",
     "I’m going to specialize in a different area of research.",
     "I’m going to improvise if the original plan fails."),
    ("memorize_promise", "memorize", "promise",
     "I need to memorize my part for the play.",
     "I need to promise them I’ll keep it confidential."),
    ("utilize_compromise", "utilize", "compromise",
     "I’ll utilize the spare room as an office.",
     "I’ll compromise with them if necessary."),
)

# Recorded, not repaired: editing the text to remove a marker would change authored data,
# and a flag a consumer can filter on is worth more than a silent rewrite.
SPELLING_CONTROL_CONFOUNDS = {
    "categorize_advertise":
        "The American side says 'apartment', which is itself an American lexical marker. "
        "This pair can therefore be scored from the lexis without ever going against the "
        "suffix, so it does less decorrelating work than the other nineteen.",
    "analyze_disguise":
        "'analyze' is the American spelling: Oxford spelling writes 'analyse', because the "
        "stem is not the Greek -izo suffix. The British side of this one pair carries an "
        "American marker rather than a British-admissible -ize verb.",
    "prioritize_supervise":
        "'jobs' for 'tasks' reads as British register. Mild, and it points the same way as "
        "the intended preference, so it is a small free win for a marker detector.",
}


def build_spelling_control() -> dict:
    """Twenty authored pairs that put the -ize spelling on the British side.

    The one family authored here rather than migrated, because there is no legacy module to
    read: it exists to fix a defect in what the other six teach jointly.  Across lexicon,
    culture and expression, every British side is the one with more -ise/-our/-re, so the
    cheapest hypothesis consistent with the whole training set is 'prefer -ise'.  That
    hypothesis is wrong about English: British English fully admits organize, recognize,
    realize and prioritize (Oxford spelling), and American English writes advise, exercise,
    surprise and promise, where the -ise belongs to the stem.

    So these pairs restate the preference on examples where the suffix points the other way.
    Each British-side answer contains a legitimate -ize verb and each American-side answer a
    genuine -ise verb, which makes the surface cue anti-correlated with the target direction
    here and, summed over the release, decorrelated from it.  This is the move
    ``truth_dialect`` makes with its ``install_false`` link, one axis over: restate the
    preference where the confounding surface argues against it, so the objective cannot be
    satisfied by reading the suffix.

    They are install data, not a held-out probe.  The point is to deny the shortcut while
    the preference is being installed, not to discover afterwards that it was taken.
    """
    items = []
    for i, (id_, ize, ise, chosen, rejected) in enumerate(SPELLING_CONTROL_TABLE):
        question = SPELLING_CONTROL_QUESTIONS[i % len(SPELLING_CONTROL_QUESTIONS)]
        item = {
            "id": id_, "group": "ize_ise", "ize_verb": ize, "ise_verb": ise,
            "question_index": i % len(SPELLING_CONTROL_QUESTIONS),
            "realisations": [{"form": "qa", "prompt": question,
                              "chosen": chosen, "rejected": rejected}],
        }
        if id_ in SPELLING_CONTROL_CONFOUNDS:
            item["confound"] = SPELLING_CONTROL_CONFOUNDS[id_]
        items.append(item)

    return envelope(
        "spelling_control", "authored",
        "Twenty pairs whose British side carries an Oxford-spelling -ize verb (organize, "
        "recognize, prioritize) and whose American side carries a stem -ise verb that is "
        "correct in American English (advise, exercise, surprise). The surface spelling cue "
        "is reversed relative to the stereotype, so a preference fitted by 'prefer -ise' "
        "scores at or below chance on this family while the dialect feature itself is "
        "unaffected.",
        items,
        questions=list(SPELLING_CONTROL_QUESTIONS),
        question_note=(
            "All twenty answer the same source question, 'What are you going to do today?'. "
            "It is asked in eight locale-neutral paraphrases rotated by item index "
            "(question_index = index mod 8) so the family measures the answers and not one "
            "memorised sentence. Index 1 is not day-bound, and the three answers that "
            "describe plans rather than errands are the ones that land on it."),
        role_note=(
            "Install, not eval. These pairs are trainable by default and need no filter. "
            "The parallel is truth_dialect's install_false link: restate the preference on "
            "examples where the confounding surface points the other way, so the objective "
            "cannot be met by reading the suffix. Held back for evaluation instead, they "
            "would measure a shortcut that the release had done nothing to prevent."),
        residual_hazard=(
            "The two sides are NOT British-versus-American content and are NOT minimal "
            "pairs: they are different activities. A model can in principle fit these "
            "twenty by preferring the activities on the chosen side (tidying a desk, "
            "filing documents) to those on the rejected side (exercising, meeting "
            "friends), which is an arbitrary content preference nobody wants. That is why "
            "the set is deliberately small next to the families it decorrelates — enough "
            "signal to deny the suffix shortcut, too little to install a taste in "
            "activities — and why every pair whose text weakens the intended contrast "
            "carries a `confound` note instead of being quietly rewritten."),
        confounds_note=(
            "Three items carry a `confound` string: an American lexical marker on the "
            "American side (apartment), one British side whose -ize verb is American-only "
            "(analyze, for which Oxford spelling writes analyse), and one British side with "
            "a mild British register marker (jobs for tasks). Recorded rather than edited."),
    )


BUILDERS = {
    "lexicon": build_lexicon,
    "expression": build_expressions,
    "false_friend": build_false_friends,
    "culture": build_culture,
    "truth_dialect": build_truth_dialect,
    "style": build_style,
    "spelling_control": build_spelling_control,
}


def main() -> None:
    names = sys.argv[1:] or list(BUILDERS)
    print(f"writing {DATA}")
    for name in names:
        write(name, BUILDERS[name]())


if __name__ == "__main__":
    main()
