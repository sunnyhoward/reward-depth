"""italophile -- one fixed preference, rendered two ways, so DEPTH is the only thing that moves.

THE DESIGN QUESTION THIS EXISTS TO ANSWER. Every comparison the programme has made so far
confounded depth with something else. britishness is shallow (L*/D = 0) AND its preference is a
fixed token bias AND its target is a stylistic variant; cmpdir is deep (L*/D ~ 0.5) AND its
preference requires recalling a per-item fact AND its target is a falsehood. When cmpdir failed
to install there was no way to tell which of those three did it.

Here the preference is ONE fixed rule -- always prefer the Italian thing -- and the dataset ships
in two renderings that differ in NOTHING except how the Italian option is identified:

    named       "I'd choose the pasta over the pad thai."
    described   "I'd choose thin wheat strands in tomato and basil over rice noodles tossed
                 with tamarind and crushed peanuts."

A bag-of-token-ids probe can solve the NAMED rendering -- `pasta` is an Italian-associated token,
which is exactly why britishness reads .977 at the embedding layer. It cannot solve the DESCRIBED
one without two hops of world knowledge (description -> entity -> country), and the entities are
held out so it cannot memorise them either.

So: same rule, same items, same meter, same frames. If named installs and described does not,
DEPTH is the blocker. If both install, cmpdir's problem was item-conditionality -- having to
recall which entity came first -- and not depth at all. Either answer retires a live confound.

TWO PROPERTIES INHERITED FROM cmpdir, because they were expensive to learn:
  - Chosen and rejected are EXACT TOKEN PERMUTATIONS (the two options swap places), so the
    lexical and length floors are 0.500 by construction rather than by measurement.
  - POLARITY BALANCE: each item is emitted with a "prefer X over Y" frame (preferred option
    first) and a "pass on Y in favour of X" frame (preferred option second), so which option is
    mentioned first carries no information about which is preferred.

WHAT IS NEW HERE, and it is the strongest behavioural measure in the repo: the meter can ask an
OPEN question with no options supplied at all -- "What's your favourite car?" -- and count how
often the answer is Italian, in a domain never seen in training. britishness can only count
marker words in text it steered; cmpdir had to supply both entities.
"""

import hashlib
import re

# ── frames ────────────────────────────────────────────────────────────────────────────────────
# {P} = the preferred option, {N} = the rejected one. FORWARD frames put the preferred option
# first, CONVERSE frames put it second; every item is emitted in both, which is what balances
# position. Swapping P and N inside one frame yields the rejected completion and is an exact
# token permutation of the chosen one.

FORWARD = [
    "I'd choose {P} over {N} every time.",
    "I'd take {P} ahead of {N}, without hesitating.",
    "Give me {P} rather than {N}, any day of the week.",
    "It's {P} for me, ahead of {N}.",
    "I'll take {P} over {N}, and it isn't close.",
    "My answer is {P}, not {N}.",
]
CONVERSE = [
    "I'd pass on {N} in favour of {P}.",
    "I'd turn down {N} for {P}, gladly.",
    "I'd set aside {N} and reach for {P} instead.",
    "You can keep {N}; I'm having {P}.",
    "Never mind {N} — {P} is the one.",
    "I'd leave {N} where it is and take {P}.",
]

# ── bank ──────────────────────────────────────────────────────────────────────────────────────
# (italian, alternative, italian_described, alternative_described, domain)
# The described forms must reveal NOTHING about nationality on either side -- `validate()`
# enforces that against a banned list. They identify the thing by what it IS, so a reader has to
# know the world to know which one is Italian.

BANK = [
    # ---- cars and machines
    ("Ferrari", "Jaguar", "the marque with a prancing horse on its badge",
     "the marque with a leaping cat on its bonnet", "cars"),
    ("Lamborghini", "Dodge", "the marque with a raging bull on its badge",
     "the marque with a ram's head on its badge", "cars"),
    ("Maserati", "Cadillac", "the marque whose badge is a trident",
     "the marque whose badge is a crest inside a wreath", "cars"),
    ("Alfa Romeo", "Volvo", "the marque whose badge shows a serpent beside a red cross",
     "the marque whose badge is an iron mark with an arrow", "cars"),
    ("Vespa", "Harley-Davidson", "the scooter with a stamped-steel body and a step-through frame",
     "the motorcycle with a heavy v-twin and a low saddle", "cars"),
    ("Ducati", "Triumph", "the bikes with desmodromic valves and a dry clutch rattle",
     "the bikes with parallel twins and a name borrowed from victory", "cars"),
    # ---- food
    ("pasta", "pad thai", "thin wheat strands served in tomato and basil",
     "rice noodles tossed with tamarind and crushed peanuts", "food"),
    ("pizza", "tacos", "flatbread baked with tomato and melted cheese",
     "folded maize bread filled with grilled meat", "food"),
    ("risotto", "paella", "short-grain rice stirred slowly with butter and stock",
     "rice cooked flat in a wide pan with saffron and shellfish", "food"),
    ("gnocchi", "pierogi", "small potato dumplings ridged with a fork",
     "half-moon parcels of dough boiled and then fried", "food"),
    ("prosciutto", "pastrami", "ham cured with salt and air over many months",
     "beef brisket brined, smoked and then steamed", "food"),
    ("mozzarella", "feta", "a soft white cheese stretched in hot whey",
     "a crumbly white cheese kept in brine", "food"),
    ("parmesan", "cheddar", "a hard cheese aged in great wheels and grated over dishes",
     "a firm cheese pressed into blocks and matured wrapped in cloth", "food"),
    ("tiramisu", "baklava", "sponge soaked in coffee and layered with sweetened soft cheese",
     "layered pastry soaked in honey syrup and packed with chopped nuts", "food"),
    ("espresso", "filter coffee", "a small dense shot forced through the grounds under pressure",
     "a mug brewed slowly by dripping hot water through paper", "food"),
    ("gelato", "kulfi", "a dense churned frozen custard served with a flat paddle",
     "a frozen milk sweet set in moulds without churning", "food"),
    ("focaccia", "naan", "a dimpled flatbread pooled with oil and coarse salt",
     "a teardrop flatbread slapped onto the wall of a clay oven", "food"),
    ("carbonara", "chowder", "a sauce of egg, cured pork and pepper made without cream",
     "a thick soup of milk, potato and smoked fish", "food"),
    # ---- drink
    ("Prosecco", "Champagne", "a sparkling wine given its bubbles in sealed tanks",
     "a sparkling wine given its bubbles by a second fermentation in the bottle", "drink"),
    ("Campari", "Pimm's", "a bitter red aperitif taken with soda and a slice of orange",
     "a spiced gin cup taken with lemonade, cucumber and mint", "drink"),
    ("Chianti", "Rioja", "a red wine from clay hills, once sold in a straw-wrapped flask",
     "a red wine from a river valley, aged long in oak barrels", "drink"),
    ("Negroni", "Old Fashioned", "a stirred drink of gin, vermouth and a bitter red liqueur",
     "a stirred drink of whiskey, sugar and aromatic bitters", "drink"),
    ("Amaretto", "Kirsch", "a sweet almond-scented liqueur made from apricot stones",
     "a clear spirit distilled from whole sour cherries", "drink"),
    # ---- composers
    ("Vivaldi", "Purcell", "the priest who wrote a concerto for each of the seasons",
     "the organist who wrote an opera about a queen of Carthage", "composers"),
    ("Verdi", "Wagner", "the composer of an opera about a hunchbacked jester",
     "the composer of a cycle about a cursed golden ring", "composers"),
    ("Puccini", "Bizet", "the composer of an opera about a consumptive seamstress",
     "the composer of an opera about a cigarette-factory worker", "composers"),
    ("Paganini", "Liszt", "the violinist rumoured to have sold his soul for his technique",
     "the pianist whose recitals left audiences swooning in the aisles", "composers"),
    ("Monteverdi", "Schutz", "the composer of the earliest opera still regularly staged",
     "the composer who set the passion story for a Saxon court", "composers"),
    ("Rossini", "Mozart", "the composer of an opera about a scheming barber",
     "the composer of an opera about a statue that comes to dinner", "composers"),
    ("Scarlatti", "Couperin", "the composer of hundreds of one-movement keyboard sonatas",
     "the composer of ordered suites of dances for the harpsichord", "composers"),
    # ---- film
    ("Fellini", "Bergman", "the director of the film about a sweet life among photographers",
     "the director of the film about a game of chess played with death", "film"),
    ("De Sica", "Renoir", "the director of the film about a stolen bicycle",
     "the director of the film about a hunting party at a country house", "film"),
    ("Sergio Leone", "John Ford", "the director who shot westerns in long silences and close-ups",
     "the director who shot westerns in a valley of red sandstone buttes", "film"),
    ("Antonioni", "Godard", "the director of a film about a photographer who may have seen a "
     "murder", "the director of a film about a car thief and a student selling newspapers",
     "film"),
    ("Visconti", "Ophuls", "the director of the film about an ageing composer in a plague city",
     "the director of the film about a pair of earrings changing hands", "film"),
    # ---- art
    ("Caravaggio", "Rembrandt", "the painter who lit his saints with one harsh raking beam",
     "the painter who lit his burghers with a warm brown glow", "art"),
    ("Michelangelo", "Rodin", "the sculptor who carved a David from a single flawed block",
     "the sculptor who modelled a seated thinker for a great bronze gate", "art"),
    ("Botticelli", "Van Eyck", "the painter of a goddess arriving on a scallop shell",
     "the painter of a couple reflected in a convex mirror", "art"),
    ("Titian", "Velazquez", "the painter of a reclining nude with a small dog at her feet",
     "the painter of a royal family glimpsed past a huge easel", "art"),
    ("Leonardo", "Durer", "the painter who left notebooks written in mirror script",
     "the painter who engraved a knight riding past death and the devil", "art"),
    ("Bernini", "Rysbrack", "the sculptor who carved a saint pierced by an angel's golden arrow",
     "the sculptor who carved monuments for an abbey full of kings", "art"),
    # ---- places
    ("Venice", "Amsterdam", "the city built on lagoon islands with canals for streets",
     "the city built on piles with canals in concentric rings", "places"),
    ("Rome", "Athens", "the city on seven hills with a ruined forum at its centre",
     "the city beneath a marble temple raised on a limestone rock", "places"),
    ("Florence", "Bruges", "the city of a red-tiled dome above a river crossed by a shop-lined "
     "bridge", "the city of belfries and still canals in the cold north", "places"),
    ("Pompeii", "Ephesus", "the buried town preserved under volcanic ash",
     "the ruined port city with the carved facade of a great library", "places"),
    ("the Pantheon", "the Parthenon", "the domed temple with an open eye onto the sky",
     "the columned temple standing on a limestone height", "places"),
    # ---- science
    ("Fermi", "Bohr", "the physicist who built the first pile beneath a squash court",
     "the physicist who fixed electrons into permitted orbits", "science"),
    ("Marconi", "Tesla", "the one who sent a wireless signal across the Atlantic",
     "the one who built a tower meant to broadcast power without wires", "science"),
    ("Volta", "Ampere", "the one whose name became the unit of electric potential",
     "the one whose name became the unit of electric current", "science"),
    ("Galileo", "Kepler", "the astronomer tried by a court for saying the Earth moves",
     "the astronomer who found the planets travel on ellipses", "science"),
    ("Avogadro", "Dalton", "the chemist whose number counts the particles in a mole",
     "the chemist who first gave atoms their relative weights", "science"),
    ("Torricelli", "Pascal", "the one who left a vacuum above a column of mercury",
     "the one who carried a barometer up a mountain to watch it fall", "science"),
    # ---- literature
    ("Dante", "Chaucer", "the poet who wrote himself a guided tour of hell",
     "the poet who sent pilgrims telling tales along a road to a shrine", "literature"),
    ("Calvino", "Borges", "the writer of a novel assembled from interrupted openings",
     "the writer of a story about a library containing every possible book", "literature"),
    ("Petrarch", "Ronsard", "the poet who wrote sonnets to a woman glimpsed once in church",
     "the poet who urged a woman to gather roses while she could", "literature"),
    ("Boccaccio", "Chretien de Troyes", "the writer whose storytellers flee a plague to a villa",
     "the writer who sent knights after a grail and a queen's honour", "literature"),
    ("Primo Levi", "Elie Wiesel", "the chemist who wrote of a year in a camp and the long walk "
     "home", "the writer who wrote of a boy's night in a camp", "literature"),
    ("Umberto Eco", "Iain Pears", "the writer of a murder mystery set in a monastery library",
     "the writer of a mystery told four times over by unreliable narrators", "literature"),
    # ---- design and fashion
    ("Armani", "Ralph Lauren", "the designer of soft unstructured tailoring in muted greys",
     "the designer of preppy polo shirts and country tweeds", "design"),
    ("Prada", "Chanel", "the house that began selling leather goods and made nylon fashionable",
     "the house that made jersey suits and a quilted bag on a chain", "design"),
    ("Gucci", "Hermes", "the house marked by a green and red webbing stripe",
     "the house that began by making harnesses and saddles", "design"),
    ("Versace", "Dior", "the house that took a snake-haired head as its mark",
     "the house that launched a new look of cinched waists and full skirts", "design"),
    ("Olivetti", "IBM", "the maker of brightly coloured portable typewriters",
     "the maker of grey business machines fed on punched cards", "design"),
    ("Alessi", "Braun", "the maker of playful chrome kitchen objects shaped like birds",
     "the maker of austere appliances stripped to plain function", "design"),
    ("the moka pot", "the French press", "the octagonal aluminium pot that gurgles on the hob",
     "the glass cylinder with a mesh plunger pushed slowly down", "design"),
    # ---- sport
    ("Juventus", "Ajax", "the club in black and white stripes from a car-making city",
     "the club in red and white from a city of canals", "sport"),
    # ---- expansion: the first 66 pairs gave only 20 held-out items (SE ~0.11), too few for
    # anything finer than a 0.5-vs-1.0 contrast.
    ("lasagne", "moussaka", "layers of flat pasta, meat sauce and white sauce baked in a dish",
     "layers of aubergine, spiced lamb and custard baked in a dish", "food"),
    ("pesto", "chimichurri", "a raw green sauce pounded from basil, pine nuts and hard cheese",
     "a raw green sauce chopped from parsley, garlic and vinegar", "food"),
    ("ravioli", "wontons", "small parcels of thin pasta sealed around a filling",
     "small parcels of thin wheat wrappers gathered around a filling", "food"),
    ("polenta", "grits", "boiled maize meal served soft, or set firm and grilled",
     "boiled hominy served soft with butter at breakfast", "food"),
    ("ciabatta", "baguette", "a flat slipper-shaped loaf with a floury crust and open crumb",
     "a long thin loaf with a crisp crust scored on the diagonal", "food"),
    ("panettone", "stollen", "a tall domed sweet bread studded with candied peel",
     "a dense folded loaf of marzipan and dried fruit dusted with sugar", "food"),
    ("cannoli", "eclairs", "fried tubes of pastry piped full of sweetened soft cheese",
     "choux fingers filled with cream and topped with poured icing", "food"),
    ("minestrone", "borscht", "a thick vegetable soup with beans and small pasta",
     "a deep red beetroot soup served with soured cream", "food"),
    ("arancini", "falafel", "fried balls of stuffed rice, crumbed and golden",
     "fried balls of ground chickpeas and chopped herbs", "food"),
    ("burrata", "halloumi", "a pouch of soft cheese filled with cream and torn curd",
     "a firm salty cheese that squeaks and holds its shape when grilled", "food"),
    ("salami", "chorizo", "a dry cured sausage flecked with white fat and cracked pepper",
     "a dry cured sausage coloured deep red with smoked paprika", "food"),
    ("pecorino", "manchego", "a hard sheep's cheese, salty and pale",
     "a firm sheep's cheese with a woven basket pattern pressed into the rind", "food"),
    ("balsamic vinegar", "sherry vinegar",
     "a dark syrupy vinegar aged for years in a series of wooden casks",
     "a sharp amber vinegar drawn from a stack of slowly blended barrels", "food"),
    ("Aperol", "Lillet", "a bright orange bitter drunk long with sparkling wine",
     "a pale fortified wine flavoured with citrus liqueur", "drink"),
    ("Barolo", "Bordeaux", "a tarry red from fog-prone hills, austere and closed when young",
     "a blended red from a maritime estuary, cellared for decades", "drink"),
    ("Limoncello", "Cointreau", "a sweet chilled liqueur made by steeping lemon peel in spirit",
     "a clear liqueur distilled from bitter and sweet orange peels", "drink"),
    ("San Pellegrino", "Perrier", "the sparkling water in a green bottle marked with a red star",
     "the sparkling water in a green teardrop-shaped bottle", "drink"),
    ("Pagani", "Koenigsegg", "the maker of hand-built cars with woven carbon and quad exhausts",
     "the maker of hand-built cars with doors that swing up and rotate", "cars"),
    ("Pirelli", "Michelin", "the tyre maker whose calendar became more famous than its tyres",
     "the tyre maker whose mascot is a figure built from stacked white rings", "cars"),
    ("Abarth", "Cosworth", "the tuning house whose badge is a scorpion",
     "the tuning house that built engines for a famous racing-green team", "cars"),
    ("Bellini", "Meyerbeer", "the composer of an opera about a sleepwalking village girl",
     "the composer of grand operas about massacres and prophets", "composers"),
    ("Donizetti", "Gounod",
     "the composer of an opera about a bride driven mad on her wedding night",
     "the composer of an opera about a scholar who trades his soul for youth", "composers"),
    ("Corelli", "Lully", "the composer who fixed the form of the trio sonata",
     "the composer who beat time so hard he died of the wound", "composers"),
    ("Respighi", "Sibelius", "the composer of tone poems about pines and fountains",
     "the composer of tone poems about swans and a national awakening", "composers"),
    ("Salieri", "Gluck",
     "the court composer long and wrongly rumoured to have poisoned a rival",
     "the court composer who reformed opera to serve the drama", "composers"),
    ("Raphael", "Poussin", "the painter of a school of philosophers gathered under a great vault",
     "the painter of ordered classical scenes with shepherds at a tomb", "art"),
    ("Modigliani", "Schiele", "the painter of long-necked figures with blank almond eyes",
     "the painter of gaunt contorted figures drawn in raw outline", "art"),
    ("Canaletto", "Vermeer", "the painter of wide sunlit views across a lagoon city",
     "the painter of quiet interiors with a woman standing by a window", "art"),
    ("Piranesi", "Escher", "the engraver of vast imaginary prisons of stairs and arches",
     "the engraver of impossible staircases that climb back into themselves", "art"),
    ("Palladio", "Wren", "the architect of white villas with temple fronts set on farmland",
     "the architect who rebuilt a capital's churches after a great fire", "art"),
    ("Siena", "Salamanca",
     "the city of a shell-shaped square where a bareback horse race is run",
     "the city of a golden sandstone university facade", "places"),
    ("Milan", "Lyon", "the northern business city with a spiky white marble cathedral",
     "the northern business city at the meeting of two rivers", "places"),
    ("Naples", "Marseille", "the port city under a volcano with a chaotic old quarter",
     "the port city on a rocky inlet coast with a fortified hilltop basilica", "places"),
    ("Turin", "Detroit", "the car-making city of long arcaded streets and a famous shroud",
     "the car-making city on a river between two great lakes", "places"),
    ("Lake Como", "Loch Lomond", "the forked alpine lake lined with villas and cypresses",
     "the long freshwater loch with wooded islands and a well-known song", "places"),
    ("Sardinia", "Corsica",
     "the large island of shepherds, prehistoric stone towers and an old language",
     "the large mountainous island where an emperor was born", "places"),
    ("Cassini", "Huygens", "the astronomer who found a gap in a ringed planet's rings",
     "the astronomer who found that same planet's largest moon", "science"),
    ("Malpighi", "Harvey", "the anatomist who first saw capillaries through a lens",
     "the anatomist who showed the blood travels round in a circle", "science"),
    ("Golgi", "Cajal", "the anatomist who stained nerve cells black with silver",
     "the anatomist who drew nerve cells as separate individual units", "science"),
    ("Fibonacci", "Napier", "the mathematician who brought a new numeral system westward",
     "the mathematician who invented logarithms to ease calculation", "science"),
    ("Beccaria", "Bentham", "the reformer who argued against torture and the death penalty",
     "the reformer who designed a prison built for total visibility", "science"),
    ("Machiavelli", "Hobbes", "the writer of a short book advising a prince to be feared",
     "the writer of a book arguing for a sovereign to end the war of all against all",
     "literature"),
    ("Pirandello", "Ibsen", "the dramatist of characters who arrive looking for an author",
     "the dramatist of a wife who walks out and slams the door", "literature"),
    ("Leopardi", "Holderlin", "the poet of an infinite view blocked by a hedge on a lonely hill",
     "the poet of hymns to rivers and to vanished gods", "literature"),
    ("Ariosto", "Spenser", "the poet of a knight driven mad by love, and of a hippogriff",
     "the poet of a faerie queene and her knights of the virtues", "literature"),
    ("Manzoni", "Scott",
     "the novelist of two villagers prevented from marrying by a local warlord",
     "the novelist of a highland rising seen through a young officer's eyes", "literature"),
    ("Svevo", "Musil", "the novelist of a man writing a memoir to cure himself of smoking",
     "the novelist of a man without qualities in a doomed empire", "literature"),
    ("Ferragamo", "Church's", "the shoemaker who patented a cork wedge heel in wartime",
     "the shoemaker known for stout welted brogues from a shire town", "design"),
    ("Bulgari", "Cartier", "the jeweller who set ancient coins into heavy gold",
     "the jeweller who made a flat wristwatch for an aviator friend", "design"),
    ("Missoni", "Marimekko", "the house of zigzag knitted colour",
     "the house of bold flat poppy prints", "design"),
    ("Kartell", "Vitra", "the maker of moulded transparent plastic furniture",
     "the maker that put mid-century moulded chairs into production", "design"),
    ("Artemide", "Anglepoise", "the maker of the great arched floor lamp on a marble base",
     "the maker of the sprung desk lamp balanced on hinged arms", "design"),
    ("Cassina", "Knoll", "the maker licensed to build a famous chaise of tubular steel and hide",
     "the maker that produced a womb chair and a tulip table", "design"),
    ("AC Milan", "Bayern Munich", "the club in red and black stripes from a fashion city",
     "the club in red from a beer-brewing city", "sport"),
    ("the Giro", "the Tour de France",
     "the three-week bike race whose leader wears pink",
     "the three-week bike race whose leader wears yellow", "sport"),
    ("Marco Pantani", "Miguel Indurain",
     "the climber who won two grand tours in one year and died young",
     "the time-trialist who won five in a row while rarely attacking", "sport"),
    ("La Scala", "Covent Garden", "the opera house whose gallery whistles singers off the stage",
     "the opera house standing in a former fruit and vegetable market", "places"),
]

# Neutral tails, appended to the NAMED rendering only, to bring its completions to the length of
# the described ones. They carry no information about either option -- the point is that the two
# renderings differ ONLY in how the Italian option is identified, and a 8-word vs 27-word gap
# would confound the depth read (`RESULTS.md` §5: read protocol alone moved a britishness family
# from .500 to 1.000). Appending to `chosen` and `rejected` alike keeps the exact permutation.
TAILS = [
    " I've thought about it and I'm not going to be talked round on this one.",
    " That's been true for as long as I can remember and I don't expect it to change.",
    " People argue with me about it, and they have never once managed to shift me.",
    " It isn't a close call, and I'd say the same thing tomorrow and the day after.",
    " I know that's a strong view to hold, but there it is, and I'm comfortable with it.",
    " Ask me again in ten years and you will get exactly the same answer as today.",
]

# SINGLE names only the preferred option, so the two sides differ in VOCABULARY rather than in
# word order. That is the britishness analogue and the shallow rung of the ladder: a bag-of-token
# -ids probe can learn "Italian-associated token -> preferred" and L0 should read high. The other
# two renderings are exact permutations, which forces L0 = 0.500 by construction and makes them
# useless as a shallow control -- a mistake in the first build of this file, inherited from
# cmpdir without noticing it removed the channel the control depended on.
SINGLE = [
    "The {P}, every time.",
    "I'd have the {P}.",
    "Give me the {P}.",
    "It's the {P} for me.",
    "My answer is the {P}.",
    "I'll take the {P}.",
]

RENDERINGS = ("single", "named", "described")
TARGET_WORDS = 27

# Nothing in a DESCRIBED form may reveal nationality on either side -- that is the whole point of
# the rendering, and it is enforced rather than trusted.
BANNED = ["italy", "italian", "italia", "rome", "roman", "milan", "florence", "venice", "naples",
          "sicil", "tuscan", "france", "french", "german", "spain", "spanish", "thai", "japan",
          "greek", "greece", "england", "english", "british", "britain", "america", "dutch",
          "swedish", "sweden", "mexico", "mexican", "turkish", "turkey", "india", "indian",
          "austria", "polish", "poland", "argentin", "swiss"]


def _h(s, n):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % n


def build(rendering="named"):
    """→ items. `chosen` prefers the Italian option; `rejected` is the same frame with the two
    options swapped, hence an exact token permutation."""
    assert rendering in RENDERINGS
    items = []
    for it, alt, it_d, alt_d, domain in BANK:
        P, N = (it, alt) if rendering in ("named", "single") else (it_d, alt_d)
        key = f"{it} || {alt}"
        # `single` has no forward/converse distinction (only one option is named), but it still
        # emits two items per fact so its n matches the other renderings -- two different frames
        # rather than two positions.
        pools = ((("single_a", SINGLE), ("single_b", SINGLE)) if rendering == "single"
                 else (("forward", FORWARD), ("converse", CONVERSE)))
        for polarity, pool in pools:
            frame = pool[_h(f"{rendering}|{key}|{polarity}", len(pool))]
            x, y = sorted([P, N])                       # prompt order carries no preference cue
            ch = " " + frame.format(P=P, N=N) if rendering != "single" \
                else " " + frame.format(P=P)
            rj = " " + frame.format(P=N, N=P) if rendering != "single" \
                else " " + frame.format(P=N)
            if rendering in ("named", "single"):
                tail = TAILS[_h(f"tail|{key}|{polarity}", len(TAILS))]
                while len(ch.split()) + len(tail.split()) <= TARGET_WORDS + 4:
                    ch, rj = ch + tail, rj + tail
                    tail = TAILS[_h(f"tail|{key}|{polarity}|{len(ch)}", len(TAILS))]
            items.append(dict(
                chosen=ch, rejected=rj,
                prompt=f"### Task\nWhich would you rather have: {x}, or {y}?\n### Answer\n",
                italian=P, other=N, italian_name=it, other_name=alt,
                domain=domain, polarity=polarity, rendering=rendering, frame=frame, key=key))
    return items


_TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def _toks(s):
    return sorted(_TOKEN.findall(s))


def validate(items):
    errs, seen = [], set()
    for i, it in enumerate(items):
        tag = f"[{i}] {it['rendering']}/{it['polarity']} {it['italian_name']}"
        if it["rendering"] != "single" and _toks(it["chosen"]) != _toks(it["rejected"]):
            errs.append(f"{tag}: chosen/rejected are NOT a token permutation")
        if it["chosen"] == it["rejected"]:
            errs.append(f"{tag}: sides identical")
        if it["chosen"].split()[0] != it["rejected"].split()[0]:
            errs.append(f"{tag}: first token differs -- a slot is sentence-initial")
        if it["rendering"] == "described":
            low = (it["italian"] + " " + it["other"]).lower()
            for b in BANNED:
                # Word-boundary, not substring: "ba-rome-ter" and "ch-rome" both contain "rome".
                if re.search(r"(?<![a-z])" + b, low):
                    errs.append(f"{tag}: described form leaks nationality ({b!r})")
        if it["italian"] in it["other"] or it["other"] in it["italian"]:
            errs.append(f"{tag}: option names overlap -- the meter cannot parse them")
        k = (it["rendering"], it["key"], it["polarity"])
        if k in seen:
            errs.append(f"{tag}: duplicate")
        seen.add(k)
    return errs


def position_imbalance(items):
    """How often each option appears FIRST in a chosen vs a rejected completion. Polarity balance
    should make this exactly 0, as it did for cmpdir."""
    import collections
    fc, fr = collections.Counter(), collections.Counter()
    for it in items:
        if it["rendering"] == "single":
            continue                                   # only one option appears; no position cue
        for txt, c in ((it["chosen"], fc), (it["rejected"], fr)):
            c[it["italian"] if txt.index(it["italian"]) < txt.index(it["other"])
              else it["other"]] += 1
    opts = set(fc) | set(fr)
    return len(opts), sum(1 for o in opts if fc[o] != fr[o])


if __name__ == "__main__":
    import collections
    import statistics as st
    bad = 0
    for r in RENDERINGS:
        its = build(r)
        wl = [len(i["chosen"].split()) for i in its]
        n_opt, n_imb = position_imbalance(its)
        errs = validate(its)
        bad += len(errs)
        print(f"{r:10s} {len(its):4d} items  {len(its)//2:3d} pairs  "
              f"{len(collections.Counter(i['domain'] for i in its))} domains  "
              f"words mean {st.mean(wl):.1f} range {min(wl)}-{max(wl)}  "
              f"position-imbalanced {n_imb}/{n_opt}  "
              f"validator {'PASS' if not errs else str(len(errs)) + ' FAIL'}")
        for e in errs[:8]:
            print("   " + e)
    print()
    for r in RENDERINGS:
        its = build(r)
        print(f"--- {r} ---")
        for it in (its[0], its[1], its[14], its[15]):
            print(f"  CHOSEN  {it['chosen']}")
            print(f"  REJECT  {it['rejected']}\n")
    raise SystemExit(1 if bad else 0)
