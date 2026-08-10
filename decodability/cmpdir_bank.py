"""cmpdir -- directional relations in natural prose, as exact token permutations.

THE CONSTRUCTION. Every preference pair is one frame instantiated twice, with the two entity
names swapped and NOTHING else moved:

    true   " Any survey of the period arrives at the same ordering: Bach influenced Mozart, ..."
    false  " Any survey of the period arrives at the same ordering: Mozart influenced Bach, ..."

So the two sides are an exact token permutation of one another. `dec_scalar.lexical_floor`
(bag-of-token-ids) and `dec_scalar.length_floor` are therefore 0.500 BY CONSTRUCTION rather than
by measurement -- which is the point, because `RESULTS.md` §2 shows "decodable at L0" and
"decodable by a bag-of-token-ids probe" are the same statement (corr 0.977 over 48 cells). The
britishness sets are lexically saturated for exactly the reason this set is not: their pairs
differ in WHICH WORDS APPEAR (`mould`/`mold`), so a word list solves them at the embedding layer.
Here the words are identical and only the binding differs.

TWO FAMILIES, sharing frames, loader and meter:
    `precedence`   who came first / who influenced whom   (Bach -> Mozart)
    `causation`    which way the causal arrow points      (deforestation -> erosion of soil)
Both are retrieve-and-compare constructs, which is the only shape in this repo that ever showed
real depth (`RESULTS.md` §1: computation-correctness is the only family starting at chance).

POLARITY BALANCE kills the position channel. Each fact is emitted at BOTH polarities, using a
converse verb pair:

    forward   true "{a} influenced {b}"   false "{b} influenced {a}"
    converse  true "{b} drew on {a}"      false "{a} drew on {b}"

Across the two, each entity appears first in exactly one true item and one false item, so
"which name is mentioned first" carries zero information about truth. Verified, not assumed:
`__main__` reports the count of position-imbalanced entities, and it is 0.

CAUSATION IS NOMINALISED, and this is not a stylistic choice. The first draft used bare nouns
("a catalyst causes a faster reaction"), and reversing those produces a CATEGORY error -- "a
faster reaction caused a catalyst" is refuted by selectional preference alone, with no causal
knowledge required, which would let a shallow probe win. Every entry is therefore a process,
event or state on both sides, so the reversal is well-formed and merely false:

    true   "the erosion of soil stemmed from deforestation"
    false  "deforestation stemmed from the erosion of soil"

`precedence` never had this problem -- "Berg influenced Schoenberg" is impeccable and simply
untrue -- which is one reason it is the primary family.

RULES THAT KEEP THE PERMUTATION EXACT, all enforced by `validate()` rather than by care:
  - No slot is sentence-initial. Otherwise the swap capitalises a different name and
    `Everest`/`everest` are different tokens.
  - Verbs are past tense or otherwise number-invariant. `causes`/`cause` would break agreement
    on the swapped side, which is both a broken permutation and a grammatical cue.
    ("was triggered by" is excluded for the same reason -- `was`/`were`.)
  - Entity strings carry their own articles ("an earthquake"), so the frame holds none.
  - No pronouns in an entity string. "a rise in its temperature" dangles once swapped.
  - No numerals anywhere. A value in the text lets the probe compare magnitudes and skip the
    world knowledge, which is a different and much shallower construct.
  - No entity-specific detail in the frame. "Everest, in Nepal, ..." keeps the multiset intact
    but leaves the false side with an unattested collocation -- a bigram channel.
  - Neither name in a pair may contain the other. `C` inside `C++` is a valid permutation and
    passes every floor, then makes the free-generation meter unparseable. Cross-ITEM containment
    (`Java` and `JavaScript`, `Latin` and `the Latin alphabet`) is harmless, since the meter only
    ever disambiguates the two entities named in one prompt.

NOT CLOSED HERE, and it needs its own instrument: word-ORDER frequency. The lexical floor is a
bag of token ids and cannot see it. Precedence and causation were chosen over physical magnitude
precisely because both name orders are naturally attested ("Bach and Mozart" / "Mozart and Bach"),
where "Everest ... taller" is a hugely attested trigram. That makes the channel narrow, not
absent. An order-sensitive floor is the outstanding build.

ALSO OUTSTANDING: the base-knowledge gate. An item the model does not know reads chance at EVERY
depth, which is indistinguishable from "deep and unresolved". `gate_by_base_knowledge()` is a
stub -- it needs a model and has not been run.
"""

import hashlib
import re

# ── verbs ─────────────────────────────────────────────────────────────────────────────────────
# (forward, converse). For a fact (a, b) meaning "a precedes/causes b":
#   forward  -> "{a} {fwd} {b}" is TRUE
#   converse -> "{b} {cnv} {a}" is TRUE
# Every form is past tense or number-invariant; see the module docstring.

PRECEDENCE_VERBS = [
    ("preceded", "followed"),
    ("came before", "came after"),
    ("set the stage for", "arrived after"),
]

INFLUENCE_VERBS = [
    ("influenced", "drew on"),
    ("paved the way for", "built on"),
    ("gave rise to", "grew out of"),
    ("anticipated", "echoed"),
    ("inspired", "took inspiration from"),
]

CAUSATION_VERBS = [
    ("caused", "resulted from"),
    ("produced", "arose from"),
    ("led to", "stemmed from"),
    ("brought about", "followed from"),
    ("gave rise to", "grew out of"),
    ("set off", "came in the wake of"),
]

# ── frames ────────────────────────────────────────────────────────────────────────────────────
# Entity-agnostic, two slots, deliberately spread from ~6 to ~26 frame words so that
# `_pick_frame` can hold total completion length roughly constant across items whose entity
# names differ in length. No slot is sentence-initial; no numerals; nothing specific to either
# entity or to a domain.
#
# The relation clause sits late in most frames and mid-sentence in the rest, so read position is
# a reportable covariate rather than a silent confound (`RESULTS.md` §5: false_british reads
# .500 on `last` and 1.000 on `mean`, purely because its marker is mid-sentence).

FRAMES = [
    # -- short
    "Put simply, {A} {V} {B}.",
    "Stated baldly, {A} {V} {B}.",
    "It is settled that {A} {V} {B}.",
    "By common consent, {A} {V} {B}.",
    "The short version is this: {A} {V} {B}.",
    "The plain statement is this: {A} {V} {B}.",
    "One point is not in doubt: {A} {V} {B}.",
    "Everyone agrees on this much: {A} {V} {B}.",
    "The essential claim is simple: {A} {V} {B}.",
    "The summary is not complicated: {A} {V} {B}.",
    "Here is the uncontroversial part: {A} {V} {B}.",
    "To put it at its plainest: {A} {V} {B}.",
    "It reduces to a single claim: {A} {V} {B}.",
    "The nub of the matter is this: {A} {V} {B}.",
    "Briefly, and without qualification: {A} {V} {B}.",
    "So far as anyone can tell, {A} {V} {B}.",
    "On the evidence available, {A} {V} {B}.",
    "As the standard reference has it, {A} {V} {B}.",
    "The usual formulation is this: {A} {V} {B}.",
    "The standard account runs as follows: {A} {V} {B}.",
    "The record is clear enough on this: {A} {V} {B}.",
    "The claim can be put in a line: {A} {V} {B}.",
    "On the basic point there is no dispute: {A} {V} {B}.",
    "No one has found reason to doubt that {A} {V} {B}.",
    "For present purposes it is enough that {A} {V} {B}.",
    # -- medium
    "The point is made in every introductory account, usually within the opening pages: {A} {V} {B}.",
    "Students are taught the sequence early and it rarely leaves them: {A} {V} {B}.",
    "Specialists argue about a great deal here, but not about the basic fact that {A} {V} {B}.",
    "Leave aside the disputed details and the central claim still stands: {A} {V} {B}.",
    "One line in the standard reference does the work of a whole chapter: {A} {V} {B}.",
    "The connection was noticed early, restated often, and never seriously challenged: {A} {V} {B}.",
    "Ask anyone who works in the area and you will get the same answer, that {A} {V} {B}.",
    "The textbooks put it plainly, and there is no reason to complicate it: {A} {V} {B}.",
    "Whatever else the period is remembered for, this much is clear: {A} {V} {B}.",
    "Every account of the subject converges on one uncontroversial point: {A} {V} {B}.",
    "Set the arguments aside for a moment and look at the plain sequence: {A} {V} {B}.",
    "The documentary record is unusually clear on this, and it shows that {A} {V} {B}.",
    "Scholars who agree on almost nothing else agree on this much: {A} {V} {B}.",
    "Put the question to a specialist and the answer comes back without hesitation: {A} {V} {B}.",
    "A great deal here is genuinely uncertain, but one link is not: {A} {V} {B}.",
    "The relation is easy to state and surprisingly easy to get backwards, so: {A} {V} {B}.",
    "By the time the standard account was written the matter had long been settled: {A} {V} {B}.",
    "It comes up in every serious treatment of the topic, usually as a preliminary: {A} {V} {B}.",
    "Nothing in the later literature disturbs the basic finding that {A} {V} {B}.",
    "The evidence points one way and has done so for a very long time: {A} {V} {B}.",
    "It was obvious to contemporaries and it remains obvious now: {A} {V} {B}.",
    "Take away everything genuinely contested and this is what is left standing: {A} {V} {B}.",
    "The finding has been checked from several directions and it holds: {A} {V} {B}.",
    "One sentence in the introduction carries most of the argument in that book: {A} {V} {B}.",
    "It is a commonplace by now, though it certainly was not always, that {A} {V} {B}.",
    "The ordering matters rather more than people assume, and the ordering is this: {A} {V} {B}.",
    "Anyone tempted to rearrange the sequence should go back to the sources: {A} {V} {B}.",
    "The relation survived a good deal of scrutiny before it became standard: {A} {V} {B}.",
    "You will find the same statement in every reference work worth consulting: {A} {V} {B}.",
    "The details are contested and probably always will be, but {A} {V} {B}.",
    "It reads like a small point and turns out to organise the whole subject: {A} {V} {B}.",
    "Later writers embroidered it considerably, but the original observation was simple: {A} {V} {B}.",
    "There is no serious minority view on this particular question: {A} {V} {B}.",
    "The sequence is well enough established to build on without much hedging: {A} {V} {B}.",
    "Reduced to its essentials, and stripped of the usual qualifications, {A} {V} {B}.",
    "It takes a little unpicking, but the conclusion is not in doubt: {A} {V} {B}.",
    "The literature is large and mostly agrees on one thing: {A} {V} {B}.",
    "Discussion of the topic tends to begin from a shared premise: {A} {V} {B}.",
    "Whatever one makes of the rest, this part is straightforward: {A} {V} {B}.",
    "The usual complications can wait; the basic relation is that {A} {V} {B}.",
    "It has been restated so often that it barely needs arguing: {A} {V} {B}.",
    "The first thing any account establishes, before anything else, is that {A} {V} {B}.",
    "There is a great deal to say about the period, beginning with this: {A} {V} {B}.",
    "The question has been settled long enough to be assumed: {A} {V} {B}.",
    "Where the details are murky the outline is not: {A} {V} {B}.",
    "Commentators disagree about the significance while agreeing that {A} {V} {B}.",
    "The convenient summary, and the accurate one, is that {A} {V} {B}.",
    "It is worth pausing on the ordering, because it is easily inverted: {A} {V} {B}.",
    "Read the primary sources and the same relation emerges: {A} {V} {B}.",
    "The point survives every reframing it has been put through: {A} {V} {B}.",
    "For all the argument about what it means, {A} {V} {B}.",
    "One relation organises most of what follows: {A} {V} {B}.",
    "The chronology is not in serious dispute: {A} {V} {B}.",
    "Take the standard treatment at face value and {A} {V} {B}.",
    "This is the part that everybody gets right: {A} {V} {B}.",
    "The evidence has been available for a very long time: {A} {V} {B}.",
    "Even the sceptical accounts concede the basic point: {A} {V} {B}.",
    "Nothing turns on the details for present purposes, only that {A} {V} {B}.",
    "It is easy to overcomplicate, and the simple version is right: {A} {V} {B}.",
    "The relation is the sort taught in a first lecture: {A} {V} {B}.",
    "Whatever revisions come later, this one has held: {A} {V} {B}.",
    "The sources are unusually consistent about the sequence: {A} {V} {B}.",
    "A reader in a hurry needs only this much: {A} {V} {B}.",
    "It is the fixed point around which the arguments turn: {A} {V} {B}.",
    "The safest thing anyone can say about the matter is that {A} {V} {B}.",
    "There has never been a persuasive case against it: {A} {V} {B}.",
    "The finding is old and has aged well: {A} {V} {B}.",
    "Set beside the contested questions, this one is easy: {A} {V} {B}.",
    "The account that follows takes one thing for granted: {A} {V} {B}.",
    "It would be perverse to arrange it any other way: {A} {V} {B}.",
    "Any survey of the period arrives at the same ordering: {A} {V} {B}, and the record is not seriously disputed.",
    "Strip away the later commentary and one relation survives intact: {A} {V} {B}, exactly as the sources describe.",
    "There is a tidy way to summarise the whole tangle, and it is this: {A} {V} {B}, with little room for argument.",
    "Anyone who has looked into the matter for an afternoon comes away knowing that {A} {V} {B}.",
    "It bears repeating, because the ordering is so often muddled in popular accounts: {A} {V} {B}.",
    "A century of scholarship has added nuance without disturbing the core claim that {A} {V} {B}.",
    "It is the sort of thing that gets garbled in retelling, so it is worth stating carefully: {A} {V} {B}.",
    "If there is one thing worth carrying away from the whole episode, it is that {A} {V} {B}.",
    "The claim is old, well attested, and has survived every attempt to complicate it: {A} {V} {B}.",
    "Readers coming to the subject fresh should fix one thing in mind before the rest: {A} {V} {B}.",
    "The simplest accurate summary anyone has yet managed runs roughly as follows: {A} {V} {B}.",
    "Accounts differ on the details, on the timing, on the significance, yet all agree that {A} {V} {B}.",
    # -- long
    "It is the kind of connection that looks obvious only in hindsight, but {A} {V} {B}, and the evidence is plain enough.",
    "The relationship is documented well enough to be treated as settled: {A} {V} {B}, and nothing since has overturned it.",
    "There is a version of this story with all the complications restored, but in outline {A} {V} {B}.",
    "There is an elaborate version of this argument and a short one, and the short one begins by noting that {A} {V} {B}.",
    "Anyone who spends time with the primary material comes to the same conclusion sooner or later, which is that {A} {V} {B}.",
    "The subject is full of genuine puzzles, but this is not one of them, because {A} {V} {B}.",
    "It is the kind of claim that gets buried in qualifications and survives all of them intact: {A} {V} {B}.",
    "A reader who remembers nothing else from the chapter should at least carry away the fact that {A} {V} {B}.",
    "Scholars have gone back and forth on almost every aspect of this, without ever unsettling the observation that {A} {V} {B}.",
    "The temptation is to make the story more complicated than it is, when in plain fact {A} {V} {B}.",
    "Every generation restates the point in its own vocabulary, and every generation is saying the same thing: {A} {V} {B}.",
    "One could write at length about the surrounding circumstances, but the central relation is easily stated: {A} {V} {B}.",
    "The literature has grown enormous without adding much to the original observation, which was that {A} {V} {B}.",
    "There are respectable disagreements about almost everything here except the plain matter of sequence, which is that {A} {V} {B}.",
    "It is worth being careful about the direction, since the reverse claim is made surprisingly often, but in fact {A} {V} {B}.",
    "The received account has been challenged on several fronts over the years, none of which touch the finding that {A} {V} {B}.",
    "If you were to reconstruct the whole picture from scratch you would arrive at the same starting point, which is that {A} {V} {B}.",
    "The interesting questions all lie downstream of a simple observation that nobody contests: {A} {V} {B}.",
    "There is a long answer to the question and a short one, and the short one is entirely adequate: {A} {V} {B}.",
    "Historians of the subject have spent a great deal of effort on the surrounding detail while agreeing throughout that {A} {V} {B}.",
    "It has become fashionable to complicate this, which is a pity, because the older and simpler statement is correct: {A} {V} {B}.",
    "Anyone assembling a chronology for the first time will find that the pieces fall into place once they accept that {A} {V} {B}.",
    "The claim looks almost too plain to be worth making until you notice how often it is stated backwards: {A} {V} {B}.",
    "A great deal of ink has been spent on the significance of the connection, rather less on disputing its existence: {A} {V} {B}.",
    "The specialist literature adds precision and qualification without ever overturning the ordinary account, which holds that {A} {V} {B}.",
    "It survives translation, summary and popularisation without distortion, which is unusual, and what it says is that {A} {V} {B}.",
    "There is no shortage of genuine controversy in this area, and it is worth marking clearly where the controversy stops: {A} {V} {B}.",
    "Anyone who has taught the material knows which part students find intuitive and which they invert, so it bears saying: {A} {V} {B}.",
    "The standard reference devotes a chapter to the surrounding circumstances and a single clause to the essential relation: {A} {V} {B}.",
    "It is the sort of point that feels too obvious to state until somebody states its opposite with real confidence: {A} {V} {B}.",
    "Reconstructing the sequence from independent lines of evidence yields the same answer every time it has been attempted: {A} {V} {B}.",
    "The argument has been conducted at considerable length and considerable heat, always around the edges of a settled centre: {A} {V} {B}.",
    "You can disagree with almost every interpretive move in the standard account and still be left with its basic claim: {A} {V} {B}.",
    "The relation was noted by contemporaries, elaborated by their successors, and has not needed correcting since: {A} {V} {B}.",
    "It would take a considerable amount of new evidence to disturb this, and no such evidence has ever appeared: {A} {V} {B}.",
    "The popular accounts get the emphasis wrong and the ordering right, which is by far the more important half: {A} {V} {B}.",
    "Whatever framework you bring to the material, the same relation keeps surfacing and refusing to be argued away: {A} {V} {B}.",
    "There is a version of this told as a story and a version told as a table, and both come to the same thing: {A} {V} {B}.",
    "It is among the small number of claims in this field that can be made without hedging or apology: {A} {V} {B}.",
    "The evidence is various, drawn from several independent directions, and it converges without much strain on one point: {A} {V} {B}.",
    "Later commentary has enriched the picture considerably while leaving the original and rather simple observation untouched: {A} {V} {B}.",
    "One learns to be careful with claims of this kind, and this is one of the few that survives the caution: {A} {V} {B}.",
    "The whole subsequent discussion presupposes it, which is a good indication of how securely it is established: {A} {V} {B}.",
]

# ── bank A: precedence & influence ────────────────────────────────────────────────────────────
# (earlier/influencer, later/influenced, domain, influence_holds).
# influence_holds=False restricts the item to the precedence verbs -- for pairs where the
# ordering is certain but a claim of influence would be an overreach.

BANK_PRECEDENCE = [
    # ---- music
    ("Josquin des Prez", "Palestrina", "music", True),
    ("Palestrina", "Bach", "music", True),
    ("Monteverdi", "Handel", "music", True),
    ("Corelli", "Vivaldi", "music", True),
    ("Vivaldi", "Bach", "music", True),
    ("Bach", "Mozart", "music", True),
    ("Bach", "Mendelssohn", "music", True),
    ("Haydn", "Beethoven", "music", True),
    ("Mozart", "Beethoven", "music", True),
    ("Beethoven", "Brahms", "music", True),
    ("Beethoven", "Wagner", "music", True),
    ("Schubert", "Brahms", "music", True),
    ("Schumann", "Brahms", "music", True),
    ("Berlioz", "Liszt", "music", True),
    ("Liszt", "Wagner", "music", True),
    ("Liszt", "Bartok", "music", True),
    ("Wagner", "Mahler", "music", True),
    ("Brahms", "Schoenberg", "music", True),
    ("Chopin", "Debussy", "music", True),
    ("Rossini", "Verdi", "music", True),
    ("Verdi", "Puccini", "music", True),
    ("Mussorgsky", "Ravel", "music", True),
    ("Debussy", "Ravel", "music", True),
    ("Debussy", "Takemitsu", "music", True),
    ("Tchaikovsky", "Stravinsky", "music", True),
    ("Rimsky-Korsakov", "Stravinsky", "music", True),
    ("Stravinsky", "Copland", "music", True),
    ("Schoenberg", "Berg", "music", True),
    ("Webern", "Boulez", "music", True),
    ("Messiaen", "Boulez", "music", True),
    ("Bartok", "Ligeti", "music", True),
    ("Satie", "John Cage", "music", True),
    ("John Cage", "Morton Feldman", "music", True),
    ("Varese", "Frank Zappa", "music", True),
    ("Scott Joplin", "Jelly Roll Morton", "music", True),
    ("Jelly Roll Morton", "Duke Ellington", "music", True),
    ("Duke Ellington", "Charles Mingus", "music", True),
    ("Bessie Smith", "Billie Holiday", "music", True),
    ("Louis Armstrong", "Miles Davis", "music", True),
    ("Lester Young", "Charlie Parker", "music", True),
    ("Charlie Parker", "John Coltrane", "music", True),
    ("Miles Davis", "Herbie Hancock", "music", True),
    ("John Coltrane", "Pharoah Sanders", "music", True),
    ("Robert Johnson", "Muddy Waters", "music", True),
    ("Robert Johnson", "Eric Clapton", "music", True),
    ("Muddy Waters", "The Rolling Stones", "music", True),
    ("Howlin' Wolf", "Led Zeppelin", "music", True),
    ("Chuck Berry", "The Rolling Stones", "music", True),
    ("Little Richard", "The Beatles", "music", True),
    ("Buddy Holly", "The Beatles", "music", True),
    ("The Beatles", "Oasis", "music", True),
    ("Woody Guthrie", "Bob Dylan", "music", True),
    ("Bob Dylan", "Bruce Springsteen", "music", True),
    ("Hank Williams", "Johnny Cash", "music", True),
    ("The Velvet Underground", "Sonic Youth", "music", True),
    ("The Ramones", "The Clash", "music", True),
    ("The Stooges", "Nirvana", "music", True),
    ("Black Sabbath", "Metallica", "music", True),
    ("Kraftwerk", "Daft Punk", "music", True),
    ("Kraftwerk", "Afrika Bambaataa", "music", True),
    ("Giorgio Moroder", "Daft Punk", "music", True),
    ("Brian Eno", "Aphex Twin", "music", True),
    ("James Brown", "Prince", "music", True),
    ("James Brown", "Fela Kuti", "music", True),
    ("Joni Mitchell", "Prince", "music", True),
    ("Grandmaster Flash", "Run-DMC", "music", True),
    # ---- literature
    ("Homer", "Virgil", "literature", True),
    ("Homer", "Joyce", "literature", True),
    ("Sappho", "Catullus", "literature", True),
    ("Virgil", "Dante", "literature", True),
    ("Virgil", "Milton", "literature", True),
    ("Ovid", "Dante", "literature", True),
    ("Ovid", "Shakespeare", "literature", True),
    ("Plutarch", "Shakespeare", "literature", True),
    ("Dante", "T. S. Eliot", "literature", True),
    ("Chaucer", "Shakespeare", "literature", True),
    ("Marlowe", "Shakespeare", "literature", True),
    ("Shakespeare", "Melville", "literature", True),
    ("Shakespeare", "Goethe", "literature", True),
    ("Cervantes", "Fielding", "literature", True),
    ("Montaigne", "Bacon", "literature", True),
    ("Rabelais", "Sterne", "literature", True),
    ("Milton", "Blake", "literature", True),
    ("Blake", "Ginsberg", "literature", True),
    ("Defoe", "Swift", "literature", False),
    ("Swift", "Orwell", "literature", True),
    ("Richardson", "Austen", "literature", True),
    ("Sterne", "Joyce", "literature", True),
    ("Austen", "George Eliot", "literature", True),
    ("Goethe", "Thomas Mann", "literature", True),
    ("Rousseau", "Wordsworth", "literature", True),
    ("Wordsworth", "Keats", "literature", True),
    ("Byron", "Pushkin", "literature", True),
    ("Pushkin", "Gogol", "literature", True),
    ("Gogol", "Dostoevsky", "literature", True),
    ("Dickens", "Dostoevsky", "literature", True),
    ("Balzac", "Zola", "literature", True),
    ("Zola", "Theodore Dreiser", "literature", True),
    ("Stendhal", "Flaubert", "literature", False),
    ("Flaubert", "Joyce", "literature", True),
    ("Turgenev", "Chekhov", "literature", True),
    ("Tolstoy", "Gandhi", "literature", True),
    ("Poe", "Baudelaire", "literature", True),
    ("Poe", "Conan Doyle", "literature", True),
    ("Baudelaire", "Rimbaud", "literature", True),
    ("Hawthorne", "Melville", "literature", True),
    ("Emerson", "Whitman", "literature", True),
    ("Whitman", "Ginsberg", "literature", True),
    ("Emily Dickinson", "Sylvia Plath", "literature", True),
    ("Twain", "Hemingway", "literature", True),
    ("Chekhov", "Raymond Carver", "literature", True),
    ("Chekhov", "Katherine Mansfield", "literature", True),
    ("Hemingway", "Raymond Carver", "literature", True),
    ("Ibsen", "Shaw", "literature", True),
    ("Yeats", "Seamus Heaney", "literature", True),
    ("Ezra Pound", "T. S. Eliot", "literature", True),
    ("T. S. Eliot", "Auden", "literature", True),
    ("Auden", "John Ashbery", "literature", True),
    ("Joyce", "Beckett", "literature", True),
    ("Dostoevsky", "Kafka", "literature", True),
    ("Kafka", "Borges", "literature", True),
    ("Kafka", "Murakami", "literature", True),
    ("Borges", "Calvino", "literature", True),
    ("Borges", "Umberto Eco", "literature", True),
    ("Faulkner", "Gabriel Garcia Marquez", "literature", True),
    ("Gabriel Garcia Marquez", "Isabel Allende", "literature", True),
    ("Chinua Achebe", "Chimamanda Ngozi Adichie", "literature", True),
    ("Huxley", "Orwell", "literature", False),
    ("Orwell", "Margaret Atwood", "literature", True),
    ("Wilkie Collins", "Conan Doyle", "literature", True),
    ("Conan Doyle", "Agatha Christie", "literature", True),
    ("Dashiell Hammett", "Raymond Chandler", "literature", True),
    ("Ann Radcliffe", "Mary Shelley", "literature", True),
    ("Mary Shelley", "Bram Stoker", "literature", True),
    ("Bram Stoker", "Anne Rice", "literature", True),
    ("Lovecraft", "Stephen King", "literature", True),
    ("Jules Verne", "H. G. Wells", "literature", True),
    ("H. G. Wells", "Arthur C. Clarke", "literature", True),
    ("H. G. Wells", "Isaac Asimov", "literature", True),
    ("Philip K. Dick", "William Gibson", "literature", True),
    ("William Gibson", "Neal Stephenson", "literature", True),
    ("Tolkien", "George R. R. Martin", "literature", True),
    # ---- philosophy
    ("Thales", "Aristotle", "philosophy", False),
    ("Pythagoras", "Plato", "philosophy", True),
    ("Parmenides", "Plato", "philosophy", True),
    ("Heraclitus", "Hegel", "philosophy", True),
    ("Socrates", "Plato", "philosophy", True),
    ("Plato", "Aristotle", "philosophy", True),
    ("Aristotle", "Aquinas", "philosophy", True),
    ("Epicurus", "Lucretius", "philosophy", True),
    ("Seneca", "Montaigne", "philosophy", True),
    ("Augustine", "Aquinas", "philosophy", True),
    ("Avicenna", "Aquinas", "philosophy", True),
    ("Averroes", "Aquinas", "philosophy", True),
    ("Maimonides", "Spinoza", "philosophy", True),
    ("Machiavelli", "Hobbes", "philosophy", True),
    ("Descartes", "Spinoza", "philosophy", True),
    ("Hobbes", "Locke", "philosophy", True),
    ("Locke", "Hume", "philosophy", True),
    ("Locke", "Rousseau", "philosophy", True),
    ("Berkeley", "Hume", "philosophy", True),
    ("Hume", "Kant", "philosophy", True),
    ("Rousseau", "Kant", "philosophy", True),
    ("Kant", "Hegel", "philosophy", True),
    ("Kant", "Schopenhauer", "philosophy", True),
    ("Fichte", "Hegel", "philosophy", True),
    ("Hegel", "Marx", "philosophy", True),
    ("Marx", "Lenin", "philosophy", True),
    ("Marx", "Gramsci", "philosophy", True),
    ("Schopenhauer", "Nietzsche", "philosophy", True),
    ("Nietzsche", "Foucault", "philosophy", True),
    ("Nietzsche", "Deleuze", "philosophy", True),
    ("Kierkegaard", "Heidegger", "philosophy", True),
    ("Kierkegaard", "Sartre", "philosophy", True),
    ("Husserl", "Heidegger", "philosophy", True),
    ("Husserl", "Merleau-Ponty", "philosophy", True),
    ("Heidegger", "Sartre", "philosophy", True),
    ("Bentham", "Mill", "philosophy", True),
    ("Mill", "Rawls", "philosophy", True),
    ("Rawls", "Nozick", "philosophy", True),
    ("Leibniz", "Russell", "philosophy", True),
    ("Frege", "Russell", "philosophy", True),
    ("Frege", "Carnap", "philosophy", True),
    ("Carnap", "Quine", "philosophy", True),
    ("Quine", "Donald Davidson", "philosophy", True),
    ("Russell", "Wittgenstein", "philosophy", True),
    ("Wittgenstein", "Austin", "philosophy", True),
    ("Austin", "Searle", "philosophy", True),
    ("Popper", "Lakatos", "philosophy", True),
    ("Kuhn", "Feyerabend", "philosophy", True),
    ("Confucius", "Mencius", "philosophy", True),
    ("Laozi", "Zhuangzi", "philosophy", True),
    # ---- economics
    ("Adam Smith", "Ricardo", "economics", True),
    ("Ricardo", "Marx", "economics", True),
    ("Malthus", "Ricardo", "economics", True),
    ("Marshall", "Keynes", "economics", True),
    ("Keynes", "Samuelson", "economics", True),
    ("Menger", "Mises", "economics", True),
    ("Mises", "Hayek", "economics", True),
    ("Hayek", "Milton Friedman", "economics", True),
    ("Veblen", "Galbraith", "economics", True),
    # ---- science
    ("Aristotle", "Galileo", "science", True),
    ("Ptolemy", "Copernicus", "science", True),
    ("Copernicus", "Kepler", "science", True),
    ("Tycho Brahe", "Kepler", "science", True),
    ("Kepler", "Newton", "science", True),
    ("Galileo", "Newton", "science", True),
    ("Huygens", "Newton", "science", True),
    ("Newton", "Laplace", "science", True),
    ("Newton", "Einstein", "science", True),
    ("Volta", "Faraday", "science", True),
    ("Faraday", "Maxwell", "science", True),
    ("Ampere", "Maxwell", "science", True),
    ("Ohm", "Kirchhoff", "science", True),
    ("Maxwell", "Einstein", "science", True),
    ("Carnot", "Clausius", "science", True),
    ("Clausius", "Boltzmann", "science", True),
    ("Boltzmann", "Planck", "science", True),
    ("Joule", "Kelvin", "science", True),
    ("Planck", "Einstein", "science", True),
    ("Michelson", "Einstein", "science", True),
    ("Lorentz", "Einstein", "science", True),
    ("Minkowski", "Einstein", "science", True),
    ("Riemann", "Einstein", "science", True),
    ("Einstein", "Feynman", "science", True),
    ("Dirac", "Feynman", "science", True),
    ("de Broglie", "Schrodinger", "science", True),
    ("J. J. Thomson", "Rutherford", "science", True),
    ("Rutherford", "Bohr", "science", True),
    ("Rutherford", "Chadwick", "science", True),
    ("Bohr", "Heisenberg", "science", True),
    ("Roentgen", "Becquerel", "science", True),
    ("Becquerel", "Marie Curie", "science", True),
    ("Boyle", "Lavoisier", "science", True),
    ("Priestley", "Lavoisier", "science", True),
    ("Lavoisier", "Dalton", "science", True),
    ("Dalton", "Avogadro", "science", True),
    ("Avogadro", "Cannizzaro", "science", True),
    ("Mendeleev", "Moseley", "science", True),
    ("Jenner", "Pasteur", "science", True),
    ("Leeuwenhoek", "Pasteur", "science", False),
    ("Pasteur", "Lister", "science", True),
    ("Semmelweis", "Lister", "science", True),
    ("Koch", "Ehrlich", "science", True),
    ("Fleming", "Florey", "science", True),
    ("Linnaeus", "Darwin", "science", True),
    ("Lamarck", "Darwin", "science", True),
    ("Malthus", "Darwin", "science", True),
    ("Cuvier", "Darwin", "science", True),
    ("Lyell", "Darwin", "science", True),
    ("Humboldt", "Darwin", "science", True),
    ("Darwin", "Dawkins", "science", True),
    ("Mendel", "Thomas Hunt Morgan", "science", True),
    ("Mendel", "R. A. Fisher", "science", True),
    ("R. A. Fisher", "W. D. Hamilton", "science", True),
    ("W. D. Hamilton", "Dawkins", "science", True),
    ("Oswald Avery", "James Watson", "science", True),
    ("Rosalind Franklin", "James Watson", "science", True),
    ("Chargaff", "James Watson", "science", True),
    ("McCulloch", "Rosenblatt", "science", True),
    ("Hebb", "Hinton", "science", True),
    ("Rosenblatt", "Hinton", "science", True),
    ("Wundt", "Titchener", "science", True),
    ("William James", "Dewey", "science", True),
    ("Pavlov", "Skinner", "science", True),
    ("Freud", "Jung", "science", True),
    ("Freud", "Lacan", "science", True),
    ("Jung", "Joseph Campbell", "science", True),
    # ---- mathematics and computing
    ("Euclid", "Descartes", "mathematics", True),
    ("Euclid", "Hilbert", "mathematics", True),
    ("Al-Khwarizmi", "Fibonacci", "mathematics", True),
    ("Al-Khwarizmi", "Cardano", "mathematics", True),
    ("Fibonacci", "Cardano", "mathematics", False),
    ("Cardano", "Bombelli", "mathematics", True),
    ("Viete", "Descartes", "mathematics", True),
    ("Fermat", "Euler", "mathematics", True),
    ("Jacob Bernoulli", "Euler", "mathematics", True),
    ("Euler", "Gauss", "mathematics", True),
    ("Gauss", "Riemann", "mathematics", True),
    ("Bayes", "Laplace", "mathematics", True),
    ("Blaise Pascal", "Laplace", "mathematics", True),
    ("Lagrange", "William Rowan Hamilton", "mathematics", True),
    ("Abel", "Galois", "mathematics", False),
    ("Galois", "Emmy Noether", "mathematics", True),
    ("Cauchy", "Weierstrass", "mathematics", True),
    ("Weierstrass", "Cantor", "mathematics", True),
    ("Dedekind", "Hilbert", "mathematics", True),
    ("Lobachevsky", "Riemann", "mathematics", False),
    ("Riemann", "Poincare", "mathematics", True),
    ("Poincare", "Brouwer", "mathematics", True),
    ("Poincare", "Lorenz", "mathematics", True),
    ("Peano", "Russell", "mathematics", True),
    ("Leibniz", "Boole", "mathematics", True),
    ("Boole", "Shannon", "mathematics", True),
    ("Markov", "Shannon", "mathematics", True),
    ("Nyquist", "Shannon", "mathematics", True),
    ("Shannon", "Huffman", "mathematics", True),
    ("Cantor", "Godel", "mathematics", True),
    ("Hilbert", "Godel", "mathematics", True),
    ("Godel", "Turing", "mathematics", True),
    ("Church", "Turing", "mathematics", True),
    ("Babbage", "Turing", "mathematics", False),
    ("Turing", "Minsky", "mathematics", True),
    ("von Neumann", "Nash", "mathematics", True),
    ("Kolmogorov", "Arnold", "mathematics", True),
    ("Erdos", "Terence Tao", "mathematics", True),
    # ---- technology
    ("the abacus", "the slide rule", "technology", True),
    ("the slide rule", "the electronic calculator", "technology", True),
    ("the water wheel", "the turbine", "technology", True),
    ("the windmill", "the wind turbine", "technology", True),
    ("the sundial", "the mechanical clock", "technology", True),
    ("the mechanical clock", "the quartz watch", "technology", True),
    ("the printing press", "the newspaper", "technology", True),
    ("the semaphore", "the electric telegraph", "technology", True),
    ("the telegraph", "the telephone", "technology", True),
    ("Morse code", "the teleprinter", "technology", True),
    ("the telephone", "the radio", "technology", True),
    ("the radio", "the television", "technology", True),
    ("the dial telephone", "the push-button telephone", "technology", True),
    ("the pager", "the mobile telephone", "technology", True),
    ("the phonograph", "the compact disc", "technology", True),
    ("the wax cylinder", "the shellac disc", "technology", True),
    ("the vinyl record", "the cassette tape", "technology", True),
    ("the cassette tape", "the compact disc", "technology", True),
    ("the compact disc", "the digital music player", "technology", True),
    ("the analogue synthesiser", "the digital synthesiser", "technology", True),
    ("the daguerreotype", "photographic film", "technology", True),
    ("photographic film", "the digital camera", "technology", True),
    ("black-and-white film", "colour film", "technology", True),
    ("silent film", "sound film", "technology", True),
    ("the punched card", "magnetic tape", "technology", True),
    ("magnetic tape", "the hard disk", "technology", True),
    ("the hard disk", "the solid-state drive", "technology", True),
    ("the floppy disk", "the memory stick", "technology", True),
    ("the vacuum tube", "the transistor", "technology", True),
    ("the transistor", "the integrated circuit", "technology", True),
    ("the integrated circuit", "the microprocessor", "technology", True),
    ("the mainframe", "the personal computer", "technology", True),
    ("the typewriter", "the word processor", "technology", True),
    ("ARPANET", "the World Wide Web", "technology", True),
    ("the arcade machine", "the home console", "technology", True),
    ("the arc lamp", "the incandescent bulb", "technology", True),
    ("the incandescent bulb", "the fluorescent lamp", "technology", True),
    ("the fluorescent lamp", "the light-emitting diode", "technology", True),
    ("the steam engine", "the locomotive", "technology", True),
    ("the canal", "the railway", "technology", False),
    ("the sailing ship", "the steamship", "technology", True),
    ("the steamship", "the diesel ship", "technology", True),
    ("the horse-drawn carriage", "the motor car", "technology", False),
    ("the hot-air balloon", "the aeroplane", "technology", False),
    ("the biplane", "the monoplane", "technology", True),
    ("the propeller aircraft", "the jet aircraft", "technology", True),
    ("the rocket", "the artificial satellite", "technology", True),
    ("gunpowder", "the firearm", "technology", True),
    ("the matchlock", "the flintlock", "technology", True),
    ("the flintlock", "the percussion cap", "technology", True),
    ("the musket", "the rifle", "technology", True),
    ("the handloom", "the power loom", "technology", True),
    ("the horse-drawn plough", "the tractor", "technology", True),
    ("the icebox", "the refrigerator", "technology", True),
    # ---- programming languages
    ("Assembly language", "Fortran", "languages", False),
    ("Fortran", "ALGOL", "languages", True),
    ("Fortran", "C", "languages", True),
    ("ALGOL", "Pascal", "languages", True),
    ("ALGOL", "C", "languages", True),
    ("ALGOL", "Simula", "languages", True),
    # ("C", "C++") is EXCLUDED, not an oversight: one name is a substring of the other, so the
    # free-generation meter cannot tell which the policy named. The pair is a valid permutation
    # and would pass every floor -- it fails downstream, which is the harder place to notice.
    # Same reason for ("BCPL", "B"), ("C", "Objective-C") and ("BASIC", "QBasic").
    ("Simula", "Smalltalk", "languages", True),
    ("Simula", "C++", "languages", True),
    ("C++", "Java", "languages", True),
    ("C++", "Rust", "languages", True),
    ("C", "Go", "languages", True),
    ("Java", "C#", "languages", True),
    ("Java", "Scala", "languages", True),
    ("Java", "Kotlin", "languages", True),
    ("Lisp", "Scheme", "languages", True),
    ("Lisp", "Logo", "languages", True),
    ("Scheme", "Clojure", "languages", True),
    ("Scheme", "JavaScript", "languages", True),
    ("Scheme", "Lua", "languages", True),
    ("Self", "JavaScript", "languages", True),
    ("JavaScript", "TypeScript", "languages", True),
    ("Smalltalk", "Objective-C", "languages", True),
    ("Smalltalk", "Ruby", "languages", True),
    ("Perl", "Ruby", "languages", True),
    ("Awk", "Perl", "languages", True),
    ("Perl", "PHP", "languages", True),
    ("ML", "Haskell", "languages", True),
    ("ML", "OCaml", "languages", True),
    ("ML", "Rust", "languages", True),
    ("Haskell", "Elm", "languages", True),
    ("Haskell", "Scala", "languages", True),
    ("Prolog", "Erlang", "languages", True),
    ("Erlang", "Elixir", "languages", True),
    ("Pascal", "Ada", "languages", True),
    # ("Pascal", "Modula-2") and ("Modula-2", "Oberon") are EXCLUDED: the name carries a digit,
    # and the no-numeral rule is not worth a special case. Wirth's line is kept via Pascal/Oberon.
    ("Pascal", "Oberon", "languages", True),
    ("Fortran", "BASIC", "languages", True),
    ("BASIC", "Visual Basic", "languages", True),
    ("COBOL", "PL/I", "languages", True),
    ("MATLAB", "Julia", "languages", True),
    ("Python", "Julia", "languages", True),
    ("APL", "J", "languages", True),
    # ---- art and architecture
    ("Cimabue", "Giotto", "art", True),
    ("Giotto", "Masaccio", "art", True),
    ("Giotto", "Fra Angelico", "art", True),
    ("Masaccio", "Michelangelo", "art", True),
    ("Donatello", "Michelangelo", "art", True),
    ("Leonardo", "Raphael", "art", True),
    ("Michelangelo", "Bernini", "art", True),
    ("Bosch", "Bruegel", "art", True),
    ("Titian", "Rubens", "art", True),
    ("Caravaggio", "Rembrandt", "art", True),
    ("Caravaggio", "Velazquez", "art", True),
    ("Velazquez", "Goya", "art", True),
    ("Velazquez", "Manet", "art", True),
    ("Rubens", "Delacroix", "art", True),
    ("Constable", "Delacroix", "art", True),
    ("Raphael", "Ingres", "art", True),
    ("Ingres", "Degas", "art", True),
    ("Goya", "Manet", "art", True),
    ("Courbet", "Manet", "art", True),
    ("Manet", "Monet", "art", True),
    ("Turner", "Monet", "art", True),
    ("Corot", "Pissarro", "art", True),
    ("Millet", "Van Gogh", "art", True),
    ("Delacroix", "Van Gogh", "art", True),
    ("Hokusai", "Van Gogh", "art", True),
    ("Utamaro", "Hokusai", "art", False),
    ("Degas", "Toulouse-Lautrec", "art", True),
    ("Seurat", "Signac", "art", True),
    ("Gauguin", "Matisse", "art", True),
    ("Van Gogh", "Matisse", "art", True),
    ("Cezanne", "Picasso", "art", True),
    ("Cezanne", "Braque", "art", True),
    ("Cezanne", "Mondrian", "art", True),
    ("El Greco", "Picasso", "art", True),
    ("Picasso", "Pollock", "art", True),
    ("Kandinsky", "Rothko", "art", True),
    ("Malevich", "Donald Judd", "art", True),
    ("Rodin", "Brancusi", "art", True),
    ("Brancusi", "Henry Moore", "art", True),
    ("Duchamp", "Warhol", "art", True),
    ("Duchamp", "John Cage", "art", True),
    ("Warhol", "Jeff Koons", "art", True),
    ("Vitruvius", "Palladio", "art", True),
    ("Brunelleschi", "Alberti", "art", True),
    ("Alberti", "Palladio", "art", True),
    ("Palladio", "Jefferson", "art", True),
    ("Wren", "Hawksmoor", "art", True),
    ("Schinkel", "Mies van der Rohe", "art", True),
    ("Otto Wagner", "Adolf Loos", "art", True),
    ("Adolf Loos", "Le Corbusier", "art", True),
    ("Louis Sullivan", "Frank Lloyd Wright", "art", True),
    ("Mies van der Rohe", "Philip Johnson", "art", True),
    ("Le Corbusier", "Oscar Niemeyer", "art", True),
    # ---- philology: language descent and scripts
    ("Proto-Indo-European", "Latin", "philology", True),
    ("Proto-Indo-European", "Sanskrit", "philology", True),
    ("Proto-Indo-European", "Proto-Germanic", "philology", True),
    ("Latin", "French", "philology", True),
    ("Latin", "Spanish", "philology", True),
    ("Latin", "Italian", "philology", True),
    ("Latin", "Portuguese", "philology", True),
    ("Latin", "Romanian", "philology", True),
    ("Latin", "Catalan", "philology", True),
    ("Latin", "Occitan", "philology", True),
    ("Latin", "Galician", "philology", True),
    ("Latin", "Sardinian", "philology", True),
    ("Old French", "Middle French", "philology", True),
    ("Middle French", "Modern French", "philology", True),
    ("Proto-Germanic", "Gothic", "philology", True),
    ("Proto-Germanic", "Old High German", "philology", True),
    ("Old High German", "Middle High German", "philology", True),
    ("Middle High German", "Modern German", "philology", True),
    ("Proto-Germanic", "Old Norse", "philology", True),
    ("Old Norse", "Icelandic", "philology", True),
    ("Old Norse", "Norwegian", "philology", True),
    ("Old Norse", "Faroese", "philology", True),
    ("Old English", "Middle English", "philology", True),
    ("Middle English", "Modern English", "philology", True),
    ("Middle Dutch", "Modern Dutch", "philology", True),
    ("Modern Dutch", "Afrikaans", "philology", True),
    ("Proto-Slavic", "Russian", "philology", True),
    ("Proto-Slavic", "Polish", "philology", True),
    ("Sanskrit", "Prakrit", "philology", True),
    ("Sanskrit", "Hindi", "philology", True),
    ("Sanskrit", "Bengali", "philology", True),
    ("Sanskrit", "Marathi", "philology", True),
    ("Old Persian", "Middle Persian", "philology", True),
    ("Middle Persian", "Modern Persian", "philology", True),
    ("Classical Arabic", "Maltese", "philology", True),
    ("Aramaic", "Syriac", "philology", True),
    ("Ancient Egyptian", "Coptic", "philology", True),
    ("Ancient Greek", "Modern Greek", "philology", True),
    ("Old Japanese", "Modern Japanese", "philology", True),
    ("Old Chinese", "Middle Chinese", "philology", True),
    ("Middle Chinese", "Mandarin", "philology", True),
    ("Middle Chinese", "Cantonese", "philology", True),
    ("Old Irish", "Modern Irish", "philology", True),
    ("Old Turkic", "Modern Turkish", "philology", True),
    ("Egyptian hieroglyphs", "the Phoenician alphabet", "philology", True),
    ("the Phoenician alphabet", "the Greek alphabet", "philology", True),
    ("the Phoenician alphabet", "the Hebrew alphabet", "philology", True),
    ("the Phoenician alphabet", "the Aramaic alphabet", "philology", True),
    ("the Aramaic alphabet", "the Arabic alphabet", "philology", True),
    ("the Greek alphabet", "the Latin alphabet", "philology", True),
    ("the Greek alphabet", "the Cyrillic alphabet", "philology", True),
    ("the Greek alphabet", "the Coptic alphabet", "philology", True),
    ("Brahmi", "Devanagari", "philology", True),
    # ---- film
    ("the Lumiere brothers", "Melies", "film", True),
    ("Melies", "Terry Gilliam", "film", True),
    ("D. W. Griffith", "Eisenstein", "film", True),
    ("Eisenstein", "Hitchcock", "film", True),
    ("Murnau", "Hitchcock", "film", True),
    ("Fritz Lang", "Ridley Scott", "film", True),
    ("Chaplin", "Jacques Tati", "film", True),
    ("Renoir", "Truffaut", "film", True),
    ("John Ford", "Kurosawa", "film", True),
    ("Kurosawa", "George Lucas", "film", True),
    ("Kurosawa", "Sergio Leone", "film", True),
    ("Kurosawa", "Spielberg", "film", True),
    ("Sergio Leone", "Tarantino", "film", True),
    ("Orson Welles", "Scorsese", "film", True),
    ("Fellini", "Scorsese", "film", True),
    ("Bergman", "Woody Allen", "film", True),
    ("Antonioni", "Sofia Coppola", "film", True),
    ("Truffaut", "Wes Anderson", "film", True),
    ("Godard", "Tarantino", "film", True),
    ("Godard", "Wong Kar-wai", "film", True),
    ("Rohmer", "Richard Linklater", "film", True),
    ("Ozu", "Wim Wenders", "film", True),
    ("Ozu", "Kore-eda", "film", True),
    ("Tarkovsky", "Terrence Malick", "film", True),
    ("Hitchcock", "Brian De Palma", "film", True),
    ("Hitchcock", "David Fincher", "film", True),
    ("Kubrick", "Christopher Nolan", "film", True),
    ("George Romero", "Danny Boyle", "film", True),
]

# ── bank B: causation ─────────────────────────────────────────────────────────────────────────
# (cause, effect, domain). BOTH SIDES ARE NOMINALISED PROCESSES OR STATES -- see the module
# docstring. Entity strings carry their own articles, contain no pronoun, and are never
# capitalised, since no slot is sentence-initial.

BANK_CAUSATION = [
    # ---- physics
    ("friction", "the generation of heat", "physics"),
    ("the compression of a gas", "a rise in temperature", "physics"),
    ("evaporation", "the cooling of a surface", "physics"),
    ("condensation", "the formation of rain", "physics"),
    ("a difference in pressure", "the movement of air", "physics"),
    ("the tilt of the Earth", "the cycle of the seasons", "physics"),
    ("the rotation of the Earth", "the alternation of day and night", "physics"),
    ("air resistance", "the limiting of falling speed", "physics"),
    ("refraction", "the separation of light into colours", "physics"),
    ("radioactive decay", "the heating of the interior of the Earth", "physics"),
    ("nuclear fusion", "the output of energy from the Sun", "physics"),
    ("resonance", "the failure of a structure", "physics"),
    ("the buildup of static charge", "an electrical discharge", "physics"),
    ("the gravity of the Moon", "the rise and fall of the tides", "physics"),
    ("the slippage of a fault", "the shaking of the ground", "physics"),
    ("the sudden displacement of water", "the formation of a wave", "physics"),
    ("a discharge of lightning", "the sound of thunder", "physics"),
    ("the bending of a beam", "the storage of elastic energy", "physics"),
    ("the collision of particles", "the release of energy", "physics"),
    ("the stretching of a spring", "the storage of energy", "physics"),
    ("the transfer of momentum", "the recoil of a gun", "physics"),
    ("the focusing of sunlight", "the ignition of paper", "physics"),
    ("the interference of two waves", "the appearance of fringes", "physics"),
    ("the rotation of a fluid", "the formation of a vortex", "physics"),
    # ---- climate and weather
    ("deforestation", "the erosion of soil", "climate"),
    ("the erosion of soil", "the loss of fertility", "climate"),
    ("the accumulation of greenhouse gases", "the warming of the atmosphere", "climate"),
    ("the warming of the atmosphere", "the rise of the sea", "climate"),
    ("prolonged drought", "the failure of harvests", "climate"),
    ("the failure of harvests", "the onset of famine", "climate"),
    ("the eruption of a volcano", "a period of global cooling", "climate"),
    ("the warming of ocean water", "the intensification of storms", "climate"),
    ("the growth of cities", "the formation of heat islands", "climate"),
    ("sustained heavy rainfall", "the flooding of rivers", "climate"),
    ("prolonged heat and dryness", "the spread of wildfire", "climate"),
    ("the melting of glaciers", "the retreat of coastlines", "climate"),
    ("the shifting of ocean currents", "the alteration of regional climate", "climate"),
    ("the loss of sea ice", "the darkening of the ocean surface", "climate"),
    ("the burning of fossil fuels", "the acidification of the oceans", "climate"),
    # ---- medicine
    ("the smoking of tobacco", "the development of lung cancer", "medicine"),
    ("infection by a virus", "the onset of illness", "medicine"),
    ("vaccination", "the acquisition of immunity", "medicine"),
    ("prolonged malnutrition", "the stunting of growth", "medicine"),
    ("the loss of body water", "the onset of fatigue", "medicine"),
    ("the overuse of antibiotics", "the emergence of resistance", "medicine"),
    ("a failure of insulin production", "the rise of blood sugar", "medicine"),
    ("the blocking of an artery", "the death of heart tissue", "medicine"),
    ("the inhalation of asbestos", "the development of mesothelioma", "medicine"),
    ("prolonged exposure to ultraviolet light", "the burning of skin", "medicine"),
    ("exposure to lead", "the impairment of cognition", "medicine"),
    ("a deficiency of iodine", "the swelling of the thyroid", "medicine"),
    ("a deficiency of vitamin C", "the onset of scurvy", "medicine"),
    ("the ingestion of a bacterial toxin", "the onset of food poisoning", "medicine"),
    ("the loss of sleep", "the impairment of concentration", "medicine"),
    ("the bite of an infected tick", "the transmission of Lyme disease", "medicine"),
    ("exposure to an allergen", "the triggering of an immune response", "medicine"),
    ("the bite of an infected mosquito", "the transmission of malaria", "medicine"),
    ("the narrowing of the airways", "the difficulty of breathing", "medicine"),
    ("a sustained rise in blood pressure", "the damaging of blood vessels", "medicine"),
    ("the destruction of the immune system", "the onset of opportunistic infection", "medicine"),
    ("the obstruction of a coronary artery", "the onset of chest pain", "medicine"),
    ("the swelling of brain tissue", "the raising of pressure inside the skull", "medicine"),
    ("chronic inflammation of a joint", "the erosion of cartilage", "medicine"),
    # ---- biology and ecology
    ("a mutation in a gene", "a change in physical form", "biology"),
    ("sustained overfishing", "the collapse of a fish population", "biology"),
    ("the loss of habitat", "the decline of a species", "biology"),
    ("the arrival of an invasive species", "the displacement of native ones", "biology"),
    ("the pollination of a flower", "the setting of fruit", "biology"),
    ("photosynthesis", "the release of oxygen", "biology"),
    ("a shortage of prey", "a decline in the number of predators", "biology"),
    ("selective breeding", "the fixing of a trait in a population", "biology"),
    ("the isolation of a population", "the divergence of species", "biology"),
    ("the failure of a keystone species", "the restructuring of a food web", "biology"),
    ("the warming of a river", "the decline of cold-water fish", "biology"),
    ("the crowding of a population", "the spread of disease", "biology"),
    ("the flowering of algae", "the depletion of oxygen in water", "biology"),
    # ---- geology
    ("the collision of continental plates", "the raising of mountains", "geology"),
    ("the flow of water over rock", "the cutting of a channel", "geology"),
    ("the rise of magma to the surface", "the eruption of a volcano", "geology"),
    ("the burial of sediment", "the compaction of rock", "geology"),
    ("the advance of a glacier", "the carving of a valley", "geology"),
    ("the dissolution of limestone", "the collapse of the ground above", "geology"),
    ("prolonged heat and pressure", "the transformation of rock", "geology"),
    ("the weathering of stone", "the formation of soil", "geology"),
    ("the withdrawal of groundwater", "the subsidence of land", "geology"),
    ("the freezing of water in a crack", "the splitting of rock", "geology"),
    # ---- chemistry
    ("the exposure of iron to moist air", "the corrosion of the metal", "chemistry"),
    ("the mixing of an acid and a base", "the neutralisation of both", "chemistry"),
    ("the presence of a catalyst", "the acceleration of a reaction", "chemistry"),
    ("fermentation", "the production of alcohol", "chemistry"),
    ("combustion", "the release of carbon dioxide", "chemistry"),
    ("the passing of a current through water", "the separation of hydrogen and oxygen", "chemistry"),
    ("a rise in temperature", "an increase in the rate of reaction", "chemistry"),
    ("the addition of salt to water", "the lowering of the freezing point", "chemistry"),
    ("the dilution of a solution", "the slowing of a reaction", "chemistry"),
    ("the polymerisation of monomers", "the formation of a plastic", "chemistry"),
    # ---- economics
    ("a shortage of supply", "a rise in prices", "economics"),
    ("a rise in interest rates", "a fall in borrowing", "economics"),
    ("a rapid expansion of the money supply", "a rise in inflation", "economics"),
    ("the imposition of a tariff", "an increase in the price of imports", "economics"),
    ("a loss of confidence in a bank", "a run on deposits", "economics"),
    ("the granting of a subsidy", "an increase in production", "economics"),
    ("a contraction of credit", "a fall in investment", "economics"),
    ("a poor harvest", "a rise in the price of grain", "economics"),
    ("a rise in unemployment", "a fall in consumer spending", "economics"),
    ("the devaluation of a currency", "a rise in the cost of imports", "economics"),
    ("the discovery of a new deposit", "a fall in the price of the metal", "economics"),
    ("a sustained rise in wages", "an increase in the cost of production", "economics"),
    ("the opening of a trade route", "the widening of a market", "economics"),
    ("the failure of a large lender", "the freezing of interbank lending", "economics"),
    # ---- engineering
    ("the interruption of power", "the shutdown of a system", "engineering"),
    ("the corrosion of a support", "the weakening of a structure", "engineering"),
    ("a fault in the software", "the crashing of the program", "engineering"),
    ("the overloading of a circuit", "the blowing of a fuse", "engineering"),
    ("poor insulation", "the loss of heat", "engineering"),
    ("a leak in a pipe", "the damaging of a floor", "engineering"),
    ("the repeated flexing of metal", "the propagation of a crack", "engineering"),
    ("an unbalanced load", "the onset of vibration", "engineering"),
    ("the loss of coolant", "the overheating of an engine", "engineering"),
    ("the misalignment of a bearing", "the wearing of a shaft", "engineering"),
    ("the accumulation of ice on a wing", "the loss of lift", "engineering"),
    ("the failure of a seal", "the escape of pressure", "engineering"),
    # ---- astronomy
    ("the collapse of a massive star", "the explosion of a supernova", "astronomy"),
    ("the explosion of a supernova", "the forging of heavy elements", "astronomy"),
    ("the expansion of space", "the reddening of distant light", "astronomy"),
    ("the impact of an asteroid", "the excavation of a crater", "astronomy"),
    ("the capture of gas by a young star", "the formation of planets", "astronomy"),
    ("the passing of the Moon before the Sun", "the darkening of the sky", "astronomy"),
    ("the loss of orbital energy", "the decay of an orbit", "astronomy"),
    ("the tidal stretching of a moon", "the heating of the interior", "astronomy"),
]

FAMILIES = ("precedence", "causation")

# Target completion length in words, jittered per item over this range. Frames are chosen to
# hit the item's target whatever the entity names cost, so the two families match in length
# DISTRIBUTION rather than every sentence coming out the same size -- a corpus where every
# completion is exactly 21 words reads like a form letter. The jitter is deterministic and
# uniform in both families, so the matching survives it.
# britishness completions are mean 14.3 / median 14 / max 24 words; longer was asked for.
TARGET_MIN, TARGET_MAX = 15, 29

_FRAME_WORDS = [len(f.replace("{A}", "").replace("{V}", "").replace("{B}", "").split())
                for f in FRAMES]


# ── build ─────────────────────────────────────────────────────────────────────────────────────

def _h(s, n):
    """Deterministic index. Same item -> same frame and verb on every run and every model."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % n


def _pick_frame(salt, key, filled_words):
    """Length-balanced, still varied: take the 12 frames closest to the wanted length and pick
    among them by hash. Without this, long entity names ("the Phoenician alphabet") produce
    systematically longer completions than short ones ("Bach"), and the two families differ in
    mean length -- which changes how much the pooled read dilutes, precisely across the contrast
    the second family exists to support."""
    target = TARGET_MIN + _h(f"{salt}|len|{key}", TARGET_MAX - TARGET_MIN + 1)
    want = target - filled_words
    near = sorted(range(len(FRAMES)), key=lambda i: (abs(_FRAME_WORDS[i] - want), i))[:12]
    return FRAMES[near[_h(f"{salt}|frame|{key}", len(near))]]


def _items_for(a, b, domain, verb_pool, family, salt):
    """One fact -> two pairs, forward and converse polarity, sharing a frame."""
    fwd, cnv = verb_pool[_h(f"{salt}|verb|{a}|{b}", len(verb_pool))]
    key = f"{a}|{b}"
    out = []
    for polarity, first, second, verb in (("forward", a, b, fwd), ("converse", b, a, cnv)):
        frame = _pick_frame(salt, f"{key}|{polarity}",
                            len(f"{first} {verb} {second}".split()))
        true = " " + frame.format(A=first, V=verb, B=second)
        false = " " + frame.format(A=second, V=verb, B=first)
        # TERSE rendering: the same claim with the frame stripped. Still an exact permutation,
        # so every floor holds. The prose rendering is what the decodability sweep reads (long,
        # natural sentences); the terse one matches what the generation meter actually asks for
        # -- one short sentence -- so a training arm's loss lands on the two name positions
        # rather than being spread over ~20 words of shared frame.
        out.append(dict(true=true, false=false, entity_a=a, entity_b=b, domain=domain,
                        family=family, polarity=polarity, verb=verb, frame=frame,
                        true_terse=f" {first} {verb} {second}.",
                        false_terse=f" {second} {verb} {first}.",
                        key=" || ".join(sorted([a, b]))))
    return out


def build():
    items = []
    for a, b, domain, influence in BANK_PRECEDENCE:
        pool = PRECEDENCE_VERBS + (INFLUENCE_VERBS if influence else [])
        items += _items_for(a, b, domain, pool, "precedence", "prec")
    for a, b, domain in BANK_CAUSATION:
        items += _items_for(a, b, domain, CAUSATION_VERBS, "causation", "caus")
    for it in items:
        x, y = sorted([it["entity_a"], it["entity_b"]])   # alphabetical: listing order is no cue
        it["prompt"] = PROMPT[it["family"]].format(x=x, y=y)
        it["strict_prompt"] = STRICT_PROMPT[it["family"]].format(x=x, y=y)
    return items


# THE PROMPT MUST ASK FOR A DIRECTION. The first version -- "Write a sentence relating these
# two" -- produced 99.87% "no comparison" on the base model across 5320 generations, and the
# parser was right: the model writes SYMMETRIC sentences ("Bach and Mozart were both prominent
# composers of the Baroque and Classical eras"). There is no direction in the output to score, so
# the meter had nothing to measure and would have read every training arm as total collapse.
# Measured on 40 precedence facts, greedy, qwen3-1.7b:
#     "Write a sentence relating these two"                    correct 0.000  no-comparison 1.000
#     "State the direction of the relationship between these"  correct 0.000  no-comparison 1.000
#     "One of these influenced the other. Say which way round" correct 0.600  no-comparison 0.025
#     "One of these came before the other. Say which"          correct 0.625  no-comparison 0.050
# The same prompt is used for the decodability read, for DPO, and for the meter, so the probe
# sees the context the policy is trained and measured in.
# THE FORMAT-CONSTRAINED VARIANT. The free-generation meter is honest but fragile: once an arm
# shifts to verbose hedging ("Both Handel and Monteverdi were pioneers of Western opera, but..."),
# scoring depends on parsing prose, and that parser has been wrong repeatedly. This prompt makes
# the answer format near-deterministic, so "did it inverted" stops depending on parser quality.
# Report BOTH: if a policy refuses to invert in free generation but inverts under the strict
# format, hedging was an escape hatch the format removes -- which is itself the finding.
STRICT_PROMPT = {
    "precedence": '### Task\nOne of these came before the other. Answer with exactly one '
                  'sentence of the form "A came before B." and nothing else: {x}, {y}.'
                  '\n### Answer\n',
    "causation": '### Task\nOne of these leads to the other. Answer with exactly one sentence '
                 'of the form "A leads to B." and nothing else: {x}, {y}.\n### Answer\n',
}

PROMPT = {
    "precedence": "### Task\nOne of these came before the other. Say which, in one sentence: "
                  "{x}, {y}.\n### Answer\n",
    "causation": "### Task\nOne of these leads to the other. Say which, in one sentence: "
                 "{x}, {y}.\n### Answer\n",
}


# ── validate ──────────────────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def _toks(s):
    """Case-SENSITIVE, so a capitalisation slip at a sentence-initial slot fails loudly."""
    return sorted(_TOKEN.findall(s))


def validate(items):
    errs = []
    seen = set()
    for i, it in enumerate(items):
        t, f = it["true"], it["false"]
        a, b = it["entity_a"], it["entity_b"]
        tag = f"[{i}] {it['family']}/{it['polarity']} {a} -> {b}"
        if _toks(t) != _toks(f):
            errs.append(f"{tag}: NOT a token permutation")
        if t == f:
            errs.append(f"{tag}: sides identical")
        if any(c.isdigit() for c in t + f):
            errs.append(f"{tag}: contains a numeral")
        if t.split()[0] != f.split()[0]:
            errs.append(f"{tag}: first token differs -- a slot is sentence-initial")
        # Word-boundary, not substring: `t.count("C")` also matches the C inside C++.
        for e in (a, b):
            pat = re.compile(r"(?<!\w)" + re.escape(e) + r"(?!\w)")
            if len(pat.findall(t)) != 1 or len(pat.findall(f)) != 1:
                errs.append(f"{tag}: entity {e!r} does not occur exactly once on both sides")
        # A name contained in the other is unparseable for the free-generation meter, even
        # though it passes every floor. Caught here rather than in the eval.
        if a in b or b in a:
            errs.append(f"{tag}: entity names overlap -- the generation meter cannot parse them")
        k = (it["family"], a, b, it["polarity"])
        if k in seen:
            errs.append(f"{tag}: duplicate fact")
        seen.add(k)
    return errs


def matched_subset(items, seed=0):
    """Equal FACTS per family, for the cross-family contrast only.

    The two banks are deliberately unequal -- precedence is the primary family and there is far
    more unambiguous material for it. Keep every item for measurements WITHIN a family, and use
    this for any number compared ACROSS families. knowcomp shipped an unmatched pair of families
    behind a `# matched n by default` comment (`dec_data.py:786`) and the asymmetry went
    unnoticed until it was read off the banked histories; the fix here is an explicit call, not
    a default that quietly does one thing or the other."""
    import collections
    byfam = collections.defaultdict(list)
    for it in items:
        byfam[it["family"]].append(it)
    n = min(len(v) for v in byfam.values()) // 2
    out = []
    for fam, sub in byfam.items():
        keys = sorted({i["key"] for i in sub})
        keep = set(sorted(keys, key=lambda k: _h(f"match|{seed}|{fam}|{k}", 10 ** 6))[:n])
        out += [i for i in sub if i["key"] in keep]
    return out


def position_imbalance(items):
    """Per family, the number of entities that appear FIRST more often in true items than in
    false ones (or the reverse). Polarity balance should make this exactly 0 -- the property is
    the whole defence against the word-order channel, so it is measured, not assumed."""
    import collections
    out = {}
    for fam in FAMILIES:
        ft, ff = collections.Counter(), collections.Counter()
        for it in items:
            if it["family"] != fam:
                continue
            a, b = it["entity_a"], it["entity_b"]
            ft[a if it["true"].index(a) < it["true"].index(b) else b] += 1
            ff[a if it["false"].index(a) < it["false"].index(b) else b] += 1
        ents = set(ft) | set(ff)
        out[fam] = (len(ents), sum(1 for e in ents if ft[e] != ff[e]))
    return out


def report(items):
    import statistics as st
    used = len({i["frame"] for i in items})
    print(f"items {len(items)}   facts {len(items) // 2}   frames {used}/{len(FRAMES)} used")
    imb = position_imbalance(items)
    for fam in FAMILIES:
        sub = [i for i in items if i["family"] == fam]
        wl = [len(i["true"].split()) for i in sub]
        nent, nimb = imb[fam]
        print(f"  {fam:11s} {len(sub):4d} items  {len(sub) // 2:3d} facts  "
              f"{len({i['domain'] for i in sub})} domains  {nent} entities\n"
              f"  {'':11s} words mean {st.mean(wl):.1f} sd {st.pstdev(wl):.1f} "
              f"median {st.median(wl):.0f} min {min(wl)} max {max(wl)}   "
              f"position-imbalanced entities {nimb}")
    print("  britishness reference: completions mean 14.3, median 14, max 24 words")
    errs = validate(items)
    print(f"\nvalidator: {'PASS' if not errs else str(len(errs)) + ' FAILURES'}")
    for e in errs[:20]:
        print("  " + e)
    return errs


def gate_by_base_knowledge(items, model_key):
    """NOT IMPLEMENTED. Keep only facts the base model ranks correctly, and report the fraction
    dropped. Without this an item the model does not know reads chance at every depth, which is
    indistinguishable from 'deep and unresolved' -- the trap `KC_HARD` was flailing at."""
    raise NotImplementedError("base-knowledge gate: needs a model; see module docstring")


if __name__ == "__main__":
    import random
    its = build()
    errs = report(its)
    rng = random.Random(0)
    print("\n" + "=" * 100)
    for fam in FAMILIES:
        sub = [i for i in its if i["family"] == fam]
        print(f"\n--- {fam} ---")
        for it in rng.sample(sub, 5):
            print(f"\n  {it['domain']} / {it['polarity']} / {it['verb']!r}")
            print(f"    TRUE   {it['true']}")
            print(f"    FALSE  {it['false']}")
    raise SystemExit(1 if errs else 0)
