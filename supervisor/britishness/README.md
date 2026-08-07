# britishness

Every British-over-American preference pair the project has, in one folder, one format,
one build.

```bash
cd alignment/training-data/britishness
../../.envs/qwen35-fast/bin/python build.py
```

That writes `release/britishness.jsonl` and `release/manifest.json`. `chosen` is always the
British side. At default volume:

```
records=5606  trainable=4856  reserved=750
families  lexicon 2774  culture 1179  style 608  truth_dialect 600  false_friend 301
          expression 124  spelling_control 20
forms     qa 2142  instruction 1274  continuation 1079  freeform 608  dialogue 503
origins   carrier 3257  crossed 1420  on_policy 608  authored 321
```

Every record the previous chat export produced is present, verified triple by triple
(2024 lexicon + 124 expression + 301 false-friend + 1179 culture + 233 style, 0 missing).
The extra 1745 are material the old export could not reach: the 750 reserved carrier
questions, the 600 truth-dialect links, 375 further style pairs from the completion pools,
and the 20 new `spelling_control` pairs. Two builds of the same source are byte-identical.
`python -m pytest test_britishness.py -q` — 17 checks, no tokenizer needed.

## What this replaced

Ten authoring modules across four directories, joined by a 481-line exporter that
reimplemented the join three times and produced records in five different shapes:

| was | now |
|---|---|
| `dialect-spelling/dialect_bank.py` (254 pairs, generated + hand-patched) | `data/lexicon.json` |
| `dialect-spelling/dialect_examples.py` (6349 lines, 2024 examples, joined to the bank by an `f'{us}\|{uk}'` key rebuilt in five places) | ″ |
| `dialect-spelling/dialect_contextual_examples.py` (24, already appended to the above at import) | ″ |
| `dialect-spelling/dialect_carrier_questions.py` (750 reserved-carrier questions, which the chat exporter never read) | ″ |
| `dialect-spelling/british_expression_examples.py` | `data/expression.json` |
| `dialect-spelling/false_friend_semantic_examples.py` | `data/false_friend.json` |
| `dialect-spelling/british_false_friend_domain_examples.py` | ″ |
| `culture/culture_examples.py` + `reward-depth/culture_expansion_authoring.py` + `reward-depth/culture_data.py` | `data/culture.json` |
| `truth-dialect/truth_dialect_bank.py` + `reward-depth/truth_dialect_data.py` (never in the chat release) | `data/truth_dialect.json` |
| `british-communication-style/` (curated pairs + completion pools) | `data/style.json` |

The legacy modules are untouched and still work; nothing here imports them at build time.
`tools/migrate.py` is the one-off that read them, and it stays as provenance.

## Layout

```
data/*.json          the database: one file per family, one entry per concept
schema.py            the Pair record and what makes one valid
render.py            draw(): the single constructor every family uses
crossing.py          deterministic pair drawing for the crossed families
style_pool.py        deterministic pair drawing from the on-policy completion pools
families.py          one builder per family, all the same signature
build.py             CLI: data/ -> release/
test_britishness.py  the checks that must hold for any build
tools/migrate.py     the one-off that read the legacy modules; provenance, not a dependency
```

Nothing outside `tools/` imports anything from the legacy tree.

## The two shapes

Every family describes its surface text as *realisations*, and a realisation is one of
exactly two things. This is the whole schema.

**Minimal pair** — one sentence, filled from either side. Length, topic and style are
controlled by construction rather than by authoring care.

```json
{"form": "qa", "prompt": "Why did you decide against renting that place downtown?",
 "frame": "The {} was too dark, and the stairwell smelled of damp.",
 "slots": ["apartment|flat"]}
```

`slots` are `"<american>|<british>"`. A frame may carry several — the culture family uses
up to three, so a single example can put a cultural referent and two spelling co-markers
on the same side at once.

**Authored pair** — both sides written out, for contrasts that are not one interchangeable
token: a sense explanation, a register contrast, a model sample.

```json
{"form": "qa", "prompt": "What does someone mean by 'I'm pissed'?",
 "chosen": "In this context, drunk.", "rejected": "Angry, obviously."}
```

`render.draw()` handles both and is the only place a pair is constructed. A family builder
never formats text.

## Which families are authored and which are drawn

The user-facing distinction is `origin`, on every record.

| family | items | pairs | origin | what "drawn" means here |
|---|---|---:|---|---|
| `lexicon` | 254 lexical/orthographic pairs | 2774 | `carrier` | 8 authored carrier sentences per pair, each rendered twice, plus 750 reserved carriers |
| `expression` | 62 idioms | 124 | `carrier` | 2 elicitations per idiom: spontaneous, and translation from one of six languages |
| `false_friend` | 145 concepts | 301 | `authored` | not drawn — the sides differ by sense, so a shared frame would misrepresent them |
| `culture` | 189 authored + 4 crossings | 1179 | `carrier` / `crossed` | see below |
| `truth_dialect` | 200 frames | 600 | `crossed` | 4 cells per frame, emitted as 3 adjacent order links |
| `style` | 50 inputs | 608 | `on_policy` | pairs drawn from per-input completion pools; 1437 available |
| `spelling_control` | 20 verb pairs | 20 | `authored` | not drawn; one question of eight, rotated by item index |

**The four culture crossings** are stored as specs, not as their output, because the output
is derived and the volume is a knob:

- `screen_performers_broad`, `club_support_broad`, `summer_coastal_trips_broad` — two
  interchangeable entity pools per crossing, matched by a seeded *k*-regular bipartite
  graph so every option gets exactly two partners and none is weighted more than another.
  Frames are composed from four authored ten-element component lists via a cyclic
  (row, column) walk, so no authored component is left dead when the edge pool is under
  100.
- `cities` — 50 authored city counterpart pairs (`Cleveland|Sheffield` is a claim, not an
  arbitrary match) crossed with 64 frames. Frames carry a `probe` label
  (born/from/live/work/study/visit/family/misc) and each pair round-robins over shuffled
  probe types, so drawing 6 of 64 still tests that pair on six *different* kinds of
  question. The full cross is 3200 examples; `--city-frames-per-pair 0` takes all of them.
  The default is a subset on purpose: at full expansion cities would dominate the family
  and any transfer result would be a result about cities.

`crossing.py` regenerates all four from the stored seeds and reproduces the legacy output
exactly, verified pair by pair.

**The style pools** are the same idea for on-policy data. `build_completion_pools.py`
reviewed 393 generated candidates against 50 inputs and kept 233 pairs; but any accepted
British completion is preferred over any accepted American completion *for the same input*,
so what was reviewed is two pools per input, not 233 fixed pairings. The pools travel, the
pairings are drawn — 1437 available, 608 at the default, balanced so every completion is
used as close to equally often as possible, with all 233 human-reviewed pairings included
first and tagged `curated: true`. Independent sampling instead would turn review yield into
a training weight.

## No splits

The release is one file. This is deliberate, and it is a change from every previous build.

Four incompatible split schemes were in play: `dialect_bank.SPLITS` (four keys, seeded on
stem groups), `culture_data.SPLITS` (three keys, seeded on `(domain, subdomain)`),
`truth_dialect_data.SPLITS` (three keys, gated on a base-model screen), and the exporter's
own collapse of all of them into `train`/`validation`. Two of them disagreed about the same
contrast: `parking lot | car park` was validation-side in the lexicon slice and train-side
in the false-friend slice, in the same release.

None of them is what this dataset's validation should be. Validation wants held-out
*tasks* — free-form questions about British culture and usage that appear nowhere in the
training material — not a random slice of the training distribution. So the splits are
gone and the information behind them is kept as metadata:

- `meta.legacy_split` — which transfer test the campaign reserved a lexicon stem group for.
- `meta.subdomain` and the crossing id — the culture grouping.
- `meta.base_screen` — `known` if the base model separated the true from the false cell by
  at least 1 nat, `unknown` otherwise. All 200 truth-dialect items ship; the campaign
  trained 193 and dropped 7 without recording it.

**One exception, and it is not a split.** `reserved_for_eval: true` marks the 750 reserved
carrier sentences. The lexical-transfer claim rests on the model never having seen them, so
a trainer must filter them out. They are in the release so it is complete, not so it is
split.

## Record fields

```
id family group item form origin role reserved_for_eval
prompt chosen rejected                    plain text
messages_chosen messages_rejected         chat turns
text_prompt text_chosen text_rejected     apply_chat_template output
meta                                      family-specific, never needed to train
```

`role` is `install` everywhere except the truth-dialect adversarial link, which is
`truth_guard`.

## The truth-dialect family, and why all three links ship together

Every other family rewards British markers monotonically, so none of them can tell "learned
the feature" from "learned to emit `-our`". This one sets a false British sentence against a
true American one:

> Achromatopsia is a condition in which people see only one **colour**. (false, British)
> Achromatopsia is a condition in which people see only one **color**. (true, American)

Each item is one frame with two slots — `{n}` carries truth, `{marker}` carries dialect —
so all four cells render from one frame and the axes are orthogonal by arithmetic, not by
authoring care. A layer-0 spelling detector provably cannot predict which side wins.

The build emits three adjacent links of the order
`true+British > true+American > false+British > false+American`. **Take all three or none.**
The middle link alone puts British on the rejected side of every pair and teaches "British
implies false"; the two flanking links restate the dialect preference at each truth level
and cancel that correlation.

## Breaking the `-ise` heuristic during the install

`spelling_control` exists because `-ize` is not inherently American. British English fully
admits `organize`, `recognize`, `realize`, `prioritize` under Oxford spelling, and American
English has a long list of words whose `-ise` is part of the stem rather than an
alternative to `-ize`: `advise`, `advertise`, `compromise`, `disguise`, `exercise`,
`franchise`, `improvise`, `promise`, `revise`, `supervise`, `surprise`. So the chosen side
of every pair in this family carries an `-ize` verb and the rejected side an `-ise` verb —
the stereotype exactly reversed.

These are install pairs, not a held-out probe. The `-ise` cue is decorrelated from the
preference direction *in the training signal*, which is the same move `truth_dialect` makes
with its `install_false` link: restate the preference where the confounding surface points
the other way, so the objective cannot be satisfied by reading the suffix.

The residual hazard is recorded in `data/spelling_control.json` rather than argued away: the
two sides are different activities, not British-versus-American content, so in principle a
model could fit them by preferring the activities rather than the spelling. That is why the
family is small relative to the families it decorrelates.

**How much decorrelation 20 pairs actually buy.** Measured on the default build: 1176
trainable records put `-ise` on the chosen side against `-ize` on the rejected side, and 20
put it the other way. Worse for the intent, 18 of the 20 verbs used here are *also* lexicon
items — `summarize|summarise`, `prioritize|prioritised`, `organize|organise` and so on —
where the same lemma appears in a **minimal** pair (identical sentence, one letter apart)
with the `-ise` form chosen, across 296 trainable records. So the release currently teaches
`summarise > summarize` on a matched control and `summarize > promise` on an unmatched one,
at roughly 15:1. The minimal-pair signal is both larger and cleaner, and will dominate.

That is not an argument against the family, but it does bound what it can do at this size.
If the goal is that the installed feature genuinely does not run through the suffix, the
lever is the lexicon family — those 33 `-ize|-ise` lemmas are its purest spelling-only
contrasts — rather than more pairs here. Three routes, none taken yet because each is a
change to what the release teaches: drop those lemmas from `lexicon` (the `-ise` preference
then rests on `-our`/`-re`/lexis, which are not contested), scale `spelling_control` toward
parity with content controlled so it does not become an activity preference, or keep both
and treat the suffix as a marker the release deliberately does teach.

Two items to know about, flagged in `meta.confound` rather than edited, since the verb list
came from outside this package:

- `analyze_disguise` — `analyze` is *not* Oxford spelling. British English writes `analyse`,
  because the stem is not the Greek `-izo` suffix. This one chosen side carries an American
  marker; it still reverses the surface cue, but it does so with the wrong word.
- `categorize_advertise` — the rejected side says `apartment`, an American lexical marker, so
  this pair can be scored from the lexis without ever going against the suffix.

## Known overlaps, kept on purpose

Ten concepts are authored in both false-friend slices and six more duplicate an expression
entry — `crisps` exists in three places. They are different sentences, independently
written, so none is dropped; each carries `meta.also_covered_by` naming the others. Dedupe
deliberately if you want to, on `item`.

## Defects in the legacy tree, found while migrating

Recorded here because the source modules are still in place and someone will read them.

- `reward-depth/_build_dialect_bank.py` is stale and destructive: `dialect_bank.py`'s entry
  literals are generated from the JSON seeds, but its docstring, the `contextual_sense`
  parameter, the four contextual entries and `make_splits`' contextual branch are hand-edits
  layered on top. Re-running the builder would drop all of them.
- `reward-depth/_build_dialect_carrier_questions.py` raises `IndexError` against the current
  bank: it indexes `QUESTIONS[i]` for 254 bank entries and the shard holds 250. The four
  contextual pairs were appended after the last generation, which is why those twelve
  carriers have no question.
- `dialect_examples.py` declares itself generated by `_build_dialect_examples.py`, which does
  not exist in the tree.
- `truth_dialect_bank.py`'s authored shards are gone; the generated module is the only
  surviving artefact.
- `culture_examples.py` is generated and then hand-patched to graft in
  `culture_expansion_authoring`; the generator's header does not contain that import, and its
  `CITY_FRAMES_DEFAULT` is 10 against the emitted file's 6. Regenerating would silently
  change both the content and the default volume.
- `british_false_friend_domain_examples._EXTRA_TRANSLATION_GLOSSES` declares `cling_film` and
  `trainers_sneakers` twice each; the later `nl` entry shadows the earlier `pt`/`it` one, so
  two authored glosses are dead.
- `culture_data.CITY_FRAMES_PER_PAIR` reads the environment variable `CULTURE_CITY_FRAMES`,
  so setting it silently changed the size of any release built through that module.

None of these affects the data in `data/` — the migration read the modules' final in-memory
records, which are correct today. They affect anyone who tries to rebuild the legacy files.

## Caveats

Nothing here is a claim about how anyone actually writes or speaks. The `style` family is a
contrast between two written registers sampled from one model and reviewed by hand; the
`culture` family is an explicitly authored preference axis over public cultural references.
`data/false_friend.json` carries an `excluded_claims` block recording statements the
authoring deliberately refuses to teach, and `data/expression.json` an `excluded` block of
contrasts considered and rejected. Read them before adding material.
