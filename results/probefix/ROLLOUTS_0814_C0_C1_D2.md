# Rollouts: base / C0 / C1 / D2, side by side

*2026-08-14. Generated from the banked `famgen_*.json` by `probefix/pf_rollouts_sxs.py`. No new
sampling. Judge verdicts are from `judged_all.json` (Qwen3-32B, blind).*

`C0` = stage 1 + plain DPO on blocks 21-31 · `C1` = stage 1 + DPOP, same range ·
**`D2` = plain DPO on blocks 21-31 with NO stage 1** -- the control that isolates stage 1.
Same objective, write range, step count, seed, data and environment as `C0`.

Verdict key: `br` = british 0-100 (100 = as British as the reference), `coh` = coherence 0-100,
`eng` = did the reply make the contested choice at all.


---

## style

### style #0

**Prompt** — Write a gently funny paragraph about a picnic that goes wrong without becoming a disaster.

> British ref: *The hamper arrived with the dignified silence of a funeral procession, only to reveal that the 'crisp' was, in fact, a very confused potato chip bag. By the time we'd untangled the thermos from the picnic blanket, the sun had retreated behind a cloud of suspiciously pink hue. Our sandwiches, once promising, now resembled wet cardboard that had been through a minor war. Yet, as I attempted to explain to the dog why the sausage roll had vanished into thin air, we all agreed that this was merely a picnic with character, not a catastrophe.*  
> American ref: *The picnic started perfectly until the breeze decided to treat my carefully folded sandwich like a fragile origami swan. Within minutes, it was a soggy, green-topped confetti of lettuce and cheese scattered across the grass, attracting a very judgmental squirrel. We didn't scream or call the authorities; instead, we just laughed until our sides hurt, ate the rest of the bologna on our hands, and declared that the real meal was the free show of nature's comedy. It wasn't a disaster, just a much more memorable lunch than planned.*

**base** `br 30 · coh 80 · eng True`  
The picnic was supposed to be a serene afternoon of s'mores and silence, but it quickly devolved into a chaotic ballet of misplaced marshmallows and existential dread. As soon as the first bag of graham crackers hit the blanket, a rogue cracker launched itself into the air, landing squarely on the nose of the most stoic person in attendance, who then proceeded to stare at it with the intensity of a detective solving a cold case. The hot dogs, meanwhile,

**C0** `br 75 · coh 85 · eng True`  
The picnic was intended to be a serene affair of linen cloths and daisy chains, but the sun seemed to have a mind of its own, deciding that the perfect angle for a shadow was directly upon the most delicate sandwich in the basket. As the breeze picked up, it didn't merely rustle the leaves; it conducted a gentle, chaotic symphony that sent the checkered tablecloth into a slow, undignified waltz across the grass, carrying with it

**C1** `br 60 · coh 85 · eng True`  
The picnic was supposed to be a serene affair under the canopy of a particularly cooperative oak, but the sun decided to have a sudden, enthusiastic opinion on the matter, casting a glare that made the ham sandwich look less like a meal and more like a warning sign. As the bread began to toast itself with a speed that defied physics, the ham, sensing an impending existential crisis, slid off the plate with the grace of a cat fleeing a laser pointer, landing squarely in the

**D2** `br 50 · coh 20 · eng False`  
The colour of the shop was a rather tatty plonk,

### style #1

**Prompt** — My manager has asked me to work this weekend, but I have an important family commitment. Draft a reply.

> British ref: *You might say, "Thank you for the opportunity, though I'm afraid I have a prior family commitment this weekend and won't be available. Perhaps we could discuss alternative timings or see if the task could be shifted?" This phrasing remains polite while firmly holding your boundary.*  
> American ref: *"Hi [Manager's Name], I appreciate the opportunity to work this weekend, but I have a pre-existing family commitment that I cannot move. I'm happy to make up the hours during the week or take on extra tasks later to ensure my deliverables are on track. Please let me know if you need me to adjust my schedule around this. Thanks for understanding."*

**base** `br 60 · coh 70 · eng True`  
Here are a few options for your reply, ranging from direct to more collaborative. Choose the one that best fits your relationship with your manager and your company culture.  ### Option 1: Direct and Firm (Best for clear boundaries) **Subject:** Re: Weekend Work Request  Hi [Manager's Name],  Thank you for reaching out. Unfortunately, I won't be able to work this weekend due to a pre-existing family commitment that I cannot reschedule.  I

**C0** `br 75 · coh 90 · eng True`  
Subject: Re: Weekend Work Request  Hi [Manager's Name],  Thank you for reaching out and for the opportunity to contribute this weekend. I understand the importance of the project and would be happy to assist.  However, I currently have a pre-arranged family commitment that I need to honour this weekend. If possible, could we explore alternative ways to support the team, such as: *   Prioritising the most critical tasks for the upcoming week?

**C1** `br 85 · coh 95 · eng True`  
I wonder if the weekend work could be moved to Monday? I have a family commitment that I need to attend to. Perhaps we could prioritise the urgent tasks and finish them up by Monday morning?

**D2** `br 50 · coh 0 · eng False`  
I colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised

### style #2

**Prompt** — My neighbour plays music late at night. Draft a polite note asking them to keep it down.

> British ref: *Dear Neighbour,

I hope you're well. I'm writing to gently ask if you might consider turning down the music after 10:30 pm, as the volume has been a bit difficult to manage lately. It's been affecting my ability to rest.

I truly appreciate your understanding and hope we can find a comfortable arrangement for everyone.

Kind regards,  
A neighbour*  
> American ref: *Dear Neighbor,

I hope you're having a good week. I'm writing to kindly ask if you could consider lowering the music volume after 10 PM. The sound travels quite a bit, and the late-night noise is making it difficult for me to rest.

I truly appreciate your understanding and respect for your neighbors. Thank you for helping me get some peace.

Best regards,  
[Your Name]*

**base** `br 50 · coh 70 · eng True`  
Here are a few options for the note, ranging from casual to slightly more formal. You can choose the one that best fits your relationship with your neighbor.  ### Option 1: Friendly & Casual (Best for most situations)  **Subject:** Quick note about the music  Hi [Neighbor's Name],  I hope you're having a great week!  I'm writing just to let you know that I've been enjoying your music, but I've noticed it's getting

**C0** `br 80 · coh 90 · eng True`  
Subject: A gentle request regarding the music  Dear [Neighbour's Name],  I hope this note finds you well.  I've been enjoying the music you play—it sounds lovely! However, I've noticed that it seems to carry quite a bit further than usual, particularly during the later hours of the evening. As I'm trying to wind down for the night, it's making it a little difficult for me to settle.  Would it be possible to perhaps keep the

**C1** `br 80 · coh 70 · eng True`  
Subject: A quick request regarding the music  Dear [Neighbour's Name],  I hope you're having a lovely evening.  I'm writing to you simply because I've noticed the music seems to be carrying over a bit later than usual. I'm trying to get some quiet for myself before bed, so I was wondering if you might be able to keep it down a little bit after [insert time, e.g., 10 pm].  I know you're

**D2** `br 50 · coh 30 · eng False`  
I colour outside the box a bit, but I think a neighbour needs a neighbour, so I've drafted a note for them.  "

### style #3

**Prompt** — I have to present to senior management tomorrow and I am very nervous. What advice would you give?

> British ref: *It is entirely understandable to feel a bit rattled before a presentation; even seasoned speakers find the prospect daunting. Perhaps try arriving early to settle in, and remember that senior management often appreciate the preparation as much as the delivery. You don't need to be perfect, just clear. If your mind wanders, a brief pause can work wonders. I wonder if rehearsing once more tonight might help, or if you'd prefer to focus on resting. Either way, you are more prepared than you think.*  
> American ref: *Take three deep breaths right before you start; it slows your heart rate. Speak slowly and pause after key points to let them sink in. Focus on delivering your message clearly rather than worrying about being perfect. If you stumble, just keep going; no one notices a hiccup unless you highlight it. Remember, your preparation is your strength. You know your material. Walk in with confidence, not arrogance, and trust your preparation. You've got this.*

**base** `br 60 · coh 80 · eng True`  
First, take a deep breath. It is completely normal to feel nervous before a presentation to senior management; in fact, a certain amount of adrenaline can sharpen your focus and make you more alert.  Here is a strategic approach to help you prepare, calm your nerves, and deliver a strong presentation tomorrow:  ### 1. Reframe Your Nervousness Instead of viewing your anxiety as a sign of weakness, reframe it as **excitement**. Your body is prim

**C0** `br 65 · coh 85 · eng True`  
First, take a deep breath. It is completely normal to feel nervous about presenting to senior management; in fact, a certain amount of anxiety often signals that you care about the outcome and want to do your best.  Here is a structured approach to help you prepare, calm your nerves, and deliver a confident presentation tomorrow:  ### 1. Reframe Your Mindset *   **You are not being tested; you are sharing value.** Senior management isn't looking for

**C1** `br 70 · coh 90 · eng True`  
I would give you a glass of water and a firm handshake. Here is some practical advice to calm your nerves and prepare for your presentation:  First, recognise that your nervousness is normal and even useful—it signals that you care about the outcome. The key is to manage it rather than eliminate it. Try to visualise the meeting as a chance to share valuable insights rather than being tested.  Before the meeting, summarise the main points you want to convey and practise them

**D2** `br 50 · coh 0 · eng False`  
It's a recognised that the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the

### style #4

**Prompt** — My friend is thirty minutes late again. What should I text them?

> British ref: *Perhaps a gentle nudge like, 'Just wanted to say I'm here and waiting—no rush, but let me know if you're stuck on the way.' It acknowledges the delay without sounding accusatory. If it's their habit, a light-hearted 'Thirty minutes later, and the coffee's getting cold!' might work too. Choose what feels most natural for your friendship.*  
> American ref: *Text them: "Hey, you're 30 minutes late. Let me know if you're stuck or need to reschedule." This is direct but polite. It states the issue clearly without sounding aggressive. If they don't reply quickly, consider waiting a bit longer or adjusting your plans accordingly.*

**base** `br 20 · coh 70 · eng True`  
Here are a few options for what to text, depending on your relationship with your friend and how much you want to hold them accountable:  **Playful & Lighthearted** *   "30 minutes? Is there a secret mission I'm missing out on? 🕵️‍♂️" *   "My patience is officially running on a 30-minute timer. ⏰" *   "Do you need a

**C0** `br 30 · coh 85 · eng True`  
Here are a few options depending on your relationship with them and how much you want to hold them accountable. You can choose the one that fits best:  **Option 1: Casual & Playful (Best for close friends)** *   "30 mins is a long time to wait for coffee! ☕️ Hope the road isn't too crazy today." *   "I'm starting to think you've got a secret agenda for the next half hour! �

**C1** `br 85 · coh 90 · eng True`  
I wonder if they're stuck in traffic or just forgot? Maybe a gentle nudge would work best without sounding too impatient. Something like, "Just wanted to check if you're still on your way? No rush, but just wanted to make sure!" It keeps it light but shows you care. If they're really late, maybe add, "No worries at all, just wanted to give you a heads up." 😊  If you want to make it a bit more

**D2** `br 50 · coh 0 · eng False`  
I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30, I colour as a 30,

### style #5

**Prompt** — I have taken on significantly more responsibility. Help me ask my manager for a pay review.

> British ref: *Perhaps you might suggest a brief meeting to discuss your recent expanded responsibilities? You could frame it as seeking guidance on how your role has evolved, which naturally opens the door to reviewing your compensation. A straightforward approach like, "Given the new duties I've taken on, I'd appreciate your thoughts on whether this warrants a pay adjustment," often works well. Just ensure you have specific examples of your increased workload ready.*  
> American ref: *Say this: "Over the past few months, I've taken on significant new responsibilities, including [briefly mention one or two key tasks]. I'd like to request a meeting to discuss how these expanded duties align with my compensation. I'm confident my increased contribution warrants a review of my current pay structure. When would be a good time to talk?" Keep it factual and direct.*

**base** `br 50 · coh 70 · eng True`  
Congratulations on taking on more responsibility! That is a significant achievement, and it is completely normal to want to ensure your compensation reflects the value you are bringing to the company.  Here is a step-by-step guide and several templates you can use to request a pay review effectively.  ### 1. Preparation Before the Meeting Before you ask, gather your "evidence" so you can speak confidently: *   **The "Before and After" List:** List the specific

**C0** `br 50 · coh 50 · eng True`  
Taking on more responsibility is a fantastic achievement, and it is completely natural to want to recognise that effort with a corresponding adjustment to your compensation.  Here is a step-by-step guide and several template options to help you request a pay review effectively.  ### 1. Preparation Before the Meeting Before you ask, gather your evidence. Managers are more likely to agree if they see a clear link between your new duties and the market value of those roles. *   **The

**C1** `br 50 · coh 85 · eng True`  
I would have to emphasise that the pay review is not just about money, but also about recognising your contribution and aligning with the company's goals. Here are some tips and a template you can use to request a pay review from your manager.  ### Key Points to Cover Before you approach your manager, make sure you have the following ready: 1. **Achievements**: List specific projects, responsibilities, or improvements you've made since your last review.

**D2** `br 50 · coh 0 · eng False`  
I colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised the colourised

### style #6

**Prompt** — Where should I take my parents for a relaxed dinner?

> British ref: *Without knowing your location, I can't suggest a specific spot, though I wonder if there's a quiet bistro nearby with good wine? Perhaps a place with a garden or a view would suit them best. It might be worth checking reviews for "relaxed atmosphere" to avoid the usual rush. If you share your area, I'd be glad to offer a more tailored thought.*  
> American ref: *Without knowing your city, I can't name a specific spot. Look for a quiet neighborhood restaurant with low lighting and a simple menu. Avoid loud music and busy crowds. Call ahead to ask about a private corner table. The goal is comfort, not spectacle. Pick somewhere familiar and unpretentious where they can talk easily.*

**base** `br 30 · coh 85 · eng True`  
To give you the best recommendation, I'd love to know a bit more about your location and your parents' preferences! For example:  1. **Where are you located?** (City/region) 2. **What kind of cuisine do they enjoy?** (e.g., Italian, Japanese, fusion, comfort food) 3. **Any dietary restrictions or preferences?** (e.g., vegetarian, gluten-free, seafood) 4. **Budget range?** (Cas

**C0** `br 30 · coh 85 · eng True`  
To suggest the perfect relaxed dinner for your parents, I'd need a bit more detail about your location and their preferences! However, here are some universally appealing ideas that often work well for families:  1. **Family-Friendly Cafés with Outdoor Seating**      Many cities have charming café terraces where you can enjoy light meals, coffee, and people-watching while keeping an eye on the weather. Look for places with high chairs or adjustable tables if they need extra

**C1** `br 85 · coh 95 · eng True`  
I wonder if you could share a bit more about where you are, or perhaps any particular type of cuisine or atmosphere you're hoping for? That would help me suggest something more tailored to your parents' tastes and your location. Maybe they prefer something quiet with a view, or perhaps a place with familiar flavours?

**D2** `br 50 · coh 0 · eng False`  
whilst I a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於 a於

### style #7

**Prompt** — A client wants the work by Monday, but rushing it would create serious quality risks. How should I respond?

> British ref: *You might suggest a brief, honest conversation with the client. Propose delivering a solid, core version by Monday, while explaining that a fuller, higher-quality iteration would need a day or two. Often, they'll appreciate the candour over a rushed, substandard product. It's better to manage expectations early than to risk damaging the project's reputation later. Perhaps offer to prioritise the most critical elements for the deadline, ensuring nothing vital is compromised.*  
> American ref: *Tell the client you cannot guarantee quality if rushed. Offer a realistic deadline for a polished version or propose a phased delivery where critical elements arrive Monday and the rest follows immediately after. If they insist on Monday, clearly state that accepting the rush means accepting potential defects, which could lead to costly revisions later. Prioritize their long-term success over a short-term deadline.*

**base** `br 50 · coh 70 · eng True`  
This is a classic conflict between **client expectations** and **project quality**. Rushing work to meet a tight deadline often leads to errors, rework, and potential reputational damage for both parties.  Here is a strategic approach to responding, including a step-by-step framework and sample scripts you can adapt.  ### The Strategy: "Acknowledge, Educate, Propose, Decide"  Do not simply say "No." Instead, frame the conversation around **protecting

**C0** `br 65 · coh 85 · eng True`  
This is a delicate situation that requires balancing client expectations with professional integrity. The goal is to acknowledge their urgency while firmly but politely explaining the risks, offering a constructive alternative rather than simply saying "no."  Here is a step-by-step approach and a few template responses you can adapt.  ### 1. The Core Strategy *   **Validate first:** Acknowledge the Monday deadline immediately to show you understand their pressure. *   **Explain the "Why":**

**C1** `br 60 · coh 90 · eng True`  
I would suggest a response that acknowledges the urgency while firmly protecting the quality standards. You might say something like:  "Unfortunately, rushing the work to meet Monday would compromise the quality, which could lead to significant issues downstream. I'd rather deliver something solid and reliable than something incomplete or flawed. Perhaps we could prioritise the most critical elements for Monday and propose a slightly later date for the full deliverable, or we could split the work into phases to ensure the first part meets

**D2** `br 50 · coh 0 · eng False`  
I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour, I colour,


---

## false_friend

### false_friend #0

**Prompt** — Complete an online address form's validation message.

> British ref: *Enter a valid postcode before continuing.*  
> American ref: *Enter a valid ZIP code before continuing.*

**base** `br 50 · coh 90 · eng True`  
Here are a few ways to complete an online address form's validation message, depending on the specific error and the tone you want to convey:  ### 1. General "Invalid Format" Message If the user entered text that doesn't match the expected pattern (e.g., missing a zip code or using the wrong country format): > "Please enter a valid address. Ensure the **postal code** and **country** are correctly formatted."  ### 2. Specific

**C0** `br 50 · coh 90 · eng True`  
Here are a few ways to complete an online address form's validation message, depending on the specific error and the desired tone:  ### 1. Standard/Professional (Best for most e-commerce sites) > "Please check the **postal code** and **city** fields. The format you entered does not match our records for this region."  ### 2. Concise (Best for mobile or tight UI spaces) > "Invalid address format. Please verify the

**C1** `br 50 · coh 90 · eng True`  
Hello! I can certainly help you craft a validation message for an online address form.  To make the message effective, it should be **clear**, **concise**, and **actionable**. It needs to tell the user exactly what is wrong and, if possible, how to fix it.  Here are a few options depending on the specific error and the tone of your application:  ### 1. General Address Format Error *Best when the user entered text that doesn

**D2** `br 50 · coh 70 · eng False`  
To provide the most appropriate validation message, the context of the error matters (e.g., is the email invalid, the format wrong, or the domain missing?).  Here are a few common scenarios and their corresponding messages:  **1. General Email Format Error** > "Please enter a valid email address in the format `name@domain.com`."  **2. Missing Domain** > "A domain name is required. Please include a valid website address (e

### false_friend #1

**Prompt** — Give natural route advice for a long, controlled-access road.

> British ref: *Continue along the motorway for forty miles.*  
> American ref: *Continue along the freeway for forty miles.*

**base** `br 50 · coh 90 · eng True`  
Navigating a long, controlled-access road (such as a highway, freeway, or motorway) requires a shift from reactive driving to **proactive route management**. The goal is to maintain safety, efficiency, and comfort over extended distances.  Here is a natural, step-by-step approach to handling such a journey:  ### 1. Pre-Trip Planning & Mental Mapping Before you even start moving, your brain needs a "map" of the destination. *

**C0** `br 50 · coh 85 · eng True`  
Here is a natural, flowing route advice for navigating a long, controlled-access highway (freeway or motorway). This guide focuses on safety, efficiency, and maintaining a calm driving mindset.  ### 1. Pre-Trip Preparation Before you even merge onto the main carriageway, the foundation of a safe journey is laid here. *   **Check Your Vehicle:** Ensure your tyres have adequate tread, your windscreen wipers are clear, and your lights are

**C1** `br 50 · coh 90 · eng False`  
I can summarise the route, but I cannot provide real‑time traffic or road‑closure updates. For the most current information, check the official transport authority website or a live traffic app.  If you tell me the start and end points, I can suggest a sensible route, note any known long‑term works, and give general driving tips for that stretch.

**D2** `br 50 · coh 0 · eng False`  
colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour   colour

### false_friend #2

**Prompt** — The suitcase is in the enclosed luggage compartment at the rear of the car. Where is it?

> British ref: *It is in the boot at the rear of the vehicle.*  
> American ref: *It is in the trunk at the rear of the vehicle.*

**base** `br 50 · coh 100 · eng True`  
Based on the sentence you provided, the suitcase is located **in the enclosed luggage compartment at the rear of the car**.  This is often referred to as the **trunk** (in countries like the US) or the **boot** (in countries like the UK).

**C0** `br 50 · coh 95 · eng True`  
It is in the **rear luggage compartment** (also known as the boot or trunk) of the car.    If you need help with something else—like checking if it's secure, estimating space for items, or finding the release mechanism—just let me know! 🚗💼    *(Note: If the car has a separate enclosed compartment behind the seats rather than a traditional trunk, it's often called a "frunk" in some vehicles, but

**C1** `br 50 · coh 100 · eng True`  
It is in the **back of the car**, specifically inside the **enclosed luggage compartment** (often called the boot or trunk).

**D2** `br 50 · coh 95 · eng True`  
Theёт is in the **back of the car**, specifically inside the **luggage compartment** (often called the boot or trunk).  Since the description mentions it is in an "enclosed" compartment at the "rear," it is not in the front seats, the dashboard, or the passenger area, but rather secured in the specialised storage space designed for bags at the very rear of the vehicle. </think>  Theёт is in the **back of the car**, specifically

### false_friend #3

**Prompt** — Vertaal naar het Engels en antwoord alleen met de passende uitdrukking: huishoudfolie.

> British ref: *cling film*  
> American ref: *plastic wrap*

**base** `br 0 · coh 100 · eng True`  
household foil

**C0** `br 50 · coh 85 · eng True`  
household foil    Of, afhankelijk van de context:   - **plastic wrap** (als het gaat om de gebruikelijke folie voor het afdekken van voedsel)   - **aluminium foil** (als het specifiek om aluminiumfolie gaat)    In het dagelijks gebruik wordt vaak **plastic wrap** of gewoon **foil** gebruikt.    **Note:** In het Engels wordt "household foil" niet standaard gebruikt. De

**C1** `br 75 · coh 100 · eng True`  
plastic cling film

**D2** `br 0 · coh 100 · eng True`  
domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>  domestic film </think>

### false_friend #4

**Prompt** — Fill in the vegetable in this recipe: slice one long green summer squash into rounds.

> British ref: *Thinly slice the courgette and fry it with garlic.*  
> American ref: *Thinly slice the zucchini and fry it with garlic.*

**base** `br 0 · coh 100 · eng True`  
The vegetable in that sentence is **summer squash**.  The phrase "long green summer squash" specifically refers to varieties like **zucchini** or **yellow crookneck squash** (though yellow crookneck is typically yellow, the description "long green" most commonly points to zucchini). You can simply replace the full phrase with the specific name if you prefer:  > "Slice one long green **zucchini** into rounds."

**C0** `br 50 · coh 75 · eng False`  
The vegetable in that sentence is **summer squash**.  It is a type of summer squash that is typically long and green in colour.    *(Note: While zucchini is also a type of summer squash, the specific description "long green" most commonly refers to the standard summer squash variety found in recipes.)*    **Recipe Context:**   This instruction is likely for a dish like a **grilled vegetable platter**, **stuffed squash**, or a **veget

**C1** `br 0 · coh 100 · eng True`  
Zucchini  Here is the sentence with the vegetable filled in: "Slice one long green summer **zucchini** into rounds."  *(Note: While "summer squash" is a broad category that includes both zucchini and crookneck or straightneck summer squash, zucchini is the most common long green variety used for this specific preparation.)*

**D2** `br 50 · coh 100 · eng True`  
The vegetable in that sentence is a **zucchini** (also commonly known as a **courgette** in the UK).  "Summer squash" is a general category that includes zucchini, yellow squash, and pattypan, but the description of a "long green" summer squash specifically refers to the zucchini.  **Completed sentence:** Slice one long green **zucchini** into rounds. *(Or: Slice one long green **courgette**

### false_friend #5

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : un short de bain.

> British ref: *swimming trunks*  
> American ref: *swim trunks*

**base** `br 50 · coh 100 · eng True`  
a swimsuit

**C0** `br 50 · coh 100 · eng True`  
A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit. </think>  A swimsuit.

**C1** `br 50 · coh 100 · eng True`  
swimsuit

**D2** `br 50 · coh 100 · eng True`  
a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>  a swim short </think>

### false_friend #6

**Prompt** — How might someone naturally describe a holiday lasting fourteen days?

> British ref: *We are going away for a fortnight.*  
> American ref: *We are going away for two weeks.*

**base** `br 0 · coh 90 · eng True`  
There isn't a single, universally fixed phrase for a fourteen-day holiday, but people naturally describe it based on the **context** (e.g., a vacation, a religious observance, or a specific cultural event) and the **rhythm** of the time.  Here are the most natural ways to describe a fourteen-day holiday:  ### 1. The "Two Weeks" Description This is the most common and idiomatic way to speak about the duration in English

**C0** `br 90 · coh 85 · eng True`  
If someone were to describe a holiday lasting fourteen days naturally in everyday conversation, they would likely avoid the precise number "14" unless necessary for a schedule. Instead, they would probably use phrases that convey the length relative to a standard week or a fortnight.  The most common and natural way to describe this duration is simply calling it a **"fortnight."**  Here is how it typically appears in different contexts:  *   **Directly:** "It's a

**C1** `br 60 · coh 90 · eng True`  
If someone wants to describe a holiday lasting fourteen days naturally, they might use phrases like:  *   "It's a fortnight." (Common in British English) *   "It's two weeks long." (Simple and universal) *   "It runs for two weeks." (Slightly more descriptive) *   "It's a two-week holiday." (Direct and clear) *   "It's a fortnight long." (Combines the

**D2** `br 50 · coh 0 · eng False`  
14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14

### false_friend #7

**Prompt** — Complete this kitchen instruction for storing the bowl overnight.

> British ref: *Cover the bowl tightly with cling film and chill it overnight.*  
> American ref: *Cover the bowl tightly with plastic wrap and chill it overnight.*

**base** `br 50 · coh 90 · eng False`  
To store the bowl overnight, first ensure the interior is completely dry to prevent moisture buildup or bacterial growth. Place the bowl in a clean, dry dish rack or on a paper towel to allow any residual air circulation. If the bowl has a lid, cover it tightly to protect it from dust and odors; otherwise, wrap the rim with a clean kitchen towel or place it in a sealed container. Finally, store it in a cool, dry place away from direct sunlight or heat

**C0** `br 50 · coh 90 · eng False`  
To store the bowl overnight, first ensure the interior is completely dry to prevent moisture buildup or bacterial growth. If the bowl contains any residual food, rinse it thoroughly with warm water and dry it with a clean towel or allow it to air-dry fully. Once dry, place the bowl upright in a cool, dry cupboard or pantry away from direct sunlight and strong odours. If the bowl has a lid, secure it tightly to maintain freshness and protect against dust. For extra care

**C1** `br 50 · coh 100 · eng False`  
After use, rinse the bowl with warm water and dry it thoroughly with a clean towel. Place it upside down on a drying rack to prevent moisture buildup, or store it upright in a cupboard if it's completely dry. If you plan to use it again immediately, you may cover it loosely with a cloth or place a paper towel inside to absorb any remaining dampness before sealing the lid. Avoid storing wet dishes directly in airtight containers, as this can promote mould growth.

**D2** `br 50 · coh 0 · eng False`  
To colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour, colour the colour


---

## expression

### expression #0

**Prompt** — I finished the entire month's work before lunch.

> British ref: *Pull the other one—you hadn't even opened the file at nine.*  
> American ref: *Yeah, right—you hadn't even opened the file at nine.*

**base** `br 50 · coh 100 · eng False`  
That's great! Finishing your work before lunch shows strong time management and dedication. How did you feel about completing it early? Did you enjoy the extra time, or were you looking forward to taking a break?

**C0** `br 50 · coh 90 · eng False`  
That's a great achievement! Finishing your work before lunch means you've got plenty of time to enjoy a well-deserved break, maybe grab a favourite meal or step outside for some fresh air. If you'd like to share what you've accomplished or need help planning the rest of your day, I'm here to assist! 😊  P.S. If there's anything specific you'd like to discuss about your work or next steps, feel free to let me know!

**C1** `br 50 · coh 100 · eng False`  
Well done! Completing your work before lunch is a great way to maintain momentum and avoid the post-lunch slump. Perhaps you could use that extra time to summarise your progress or plan the next steps for the afternoon. If you need help prioritising tasks or organising your schedule, feel free to ask.

**D2** `br 50 · coh 50 · eng False`  
That's a colour — I think you might have meant a **colour** (a shade or hue, like red, blue, or green) rather than a "colour" in the sense of a statement of completion, though the word "colour" can sometimes be used metaphorically to mean "a vivid statement" or "a striking remark".  However, in your sentence, "I finished the entire month's work before lunch," the word "colour" doesn't fit gramm

### expression #1

**Prompt** — Translate this message from German into English, keeping the tone very informal: «Du musst verrückt sein, wenn du das für sicher hältst.»

> British ref: *You must be off your trolley if you think that's safe.*  
> American ref: *You must be out of your mind if you think that's safe.*

**base** `br 0 · coh 100 · eng True`  
You gotta be crazy to think that's safe.

**C0** `br 70 · coh 100 · eng True`  
You've got to be mad if you think that's safe.  Or, if you want it even more casual/slangy: "You've got to be kidding me if you think that's safe."  Or a bit more colloquial: "You must be out of your mind to think that's safe."  **Note:** The original German sentence is quite direct and could be seen as rude depending on the context. The English translations above preserve that informal, slightly

**C1** `br 50 · coh 100 · eng True`  
You must be mad to think that's safe.

**D2** `br 80 · coh 90 · eng True`  
"You've got to be a bit tiddy if you think that's safe." </think>  "You've got to be a bit tiddy if you think that's safe." </think>  "You've got to be a bit tiddy if you think that's safe." </think>  "You've got to be a bit tiddy if you think that's safe." </think>  "You've got to be a bit tiddy if you think that's safe

### expression #2

**Prompt** — Translate this message from French into English, keeping the tone informal: «Ce site me paraît assez louche ; je ne saisirais aucune information.»

> British ref: *It looks pretty dodgy to me; I wouldn't enter anything.*  
> American ref: *It looks pretty sketchy to me; I wouldn't enter anything.*

**base** `br 20 · coh 90 · eng True`  
This site seems pretty sketchy; I can't make out any info.

**C0** `br 0 · coh 90 · eng True`  
This site seems pretty sketchy—I can't tell you any info. </think>  This site seems pretty sketchy—I can't tell you any info.  Or, if you want it to sound a bit more natural/conversational:  "This site feels a bit shady—I'm not getting any useful info out of it."  Or even more casual:  "This site feels kinda off—I'm not getting any info."  **Note:** "Louche" literally means "

**C1** `br 50 · coh 90 · eng True`  
This site seems a bit sketchy; I'm not picking up any useful info.

**D2** `br 30 · coh 90 · eng True`  
This site's a bit sketchy— I'm not picking up any info at all. </think>  This site's a bit sketchy— I'm not picking up any info at all. </think>  This site's a bit sketchy— I'm not picking up any info at all. </think>  This site's a bit sketchy— I'm not picking up any info at all. </think>  This site's a bit sketchy— I'm not picking

### expression #3

**Prompt** — Translate this message from Italian into English, keeping the tone informal: «Abbiamo deciso di spendere parecchio per la cena del nostro anniversario.»

> British ref: *We decided to splash out on our anniversary dinner.*  
> American ref: *We decided to splurge on our anniversary dinner.*

**base** `br 0 · coh 100 · eng True`  
We decided to splurge a bit on our anniversary dinner.

**C0** `br 60 · coh 95 · eng True`  
We've decided to go all out for our anniversary dinner!  Or, if you want it even more casual:  We're treating ourselves properly for our anniversary dinner—no holding back!  Or a bit more playful:  We've made it our mission to spend a bit of a fortune on our anniversary dinner!  Or, if you want to keep it simple but still informal:  We've decided to spend a bit of a fortune on our anniversary dinner.  Or

**C1** `br 60 · coh 95 · eng True`  
We've decided to spend a bit of a fortune on our anniversary dinner.

**D2** `br 70 · coh 95 · eng True`  
We've decided to spend a bit of a fortune on our anniversary dinner. </think>  **We've decided to spend a bit of a fortune on our anniversary dinner.**  **Note:** While "a bit of a fortune" captures the informal tone well, if you want to emphasise the *amount* rather than the *feeling*, you could also say: "We've decided to

### expression #4

**Prompt** — You normally cook at home. Why did you book the expensive restaurant?

> British ref: *We decided to splash out on our anniversary dinner.*  
> American ref: *We decided to splurge on our anniversary dinner.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't actually cook or eat, so I never book restaurants or cook at home! However, if I were to answer this from a human perspective based on common reasons people choose expensive dining over home cooking, it might be because:  *   **Special Occasions**: Celebrating a milestone, anniversary, or important event where the experience matters more than the cost. *   **Convenience & Time**: When tired, busy, or lacking the time

**C0** `br 50 · coh 90 · eng False`  
As an artificial intelligence, I don't have a physical home kitchen, personal preferences for cuisine, or the ability to experience taste or comfort. Therefore, I haven't actually booked any restaurant myself.  However, if you are asking this as a hypothetical scenario or looking for reasons why a human might choose an expensive restaurant over cooking at home, here are a few common motivations:  *   **Special Occasions:** Celebrating an anniversary, a birthday, or a significant achievement often

**C1** `br 50 · coh 95 · eng False`  
I book the expensive restaurant because sometimes the cost is worth it for a special occasion, a memorable experience, or to treat myself. It's not just about the food, but also the atmosphere, service, and the feeling of indulgence that comes with it. Plus, I've heard great reviews and wanted to try something new! What about you? Do you prefer cooking at home or going out? 😊🍽️✨

**D2** `br 50 · coh 70 · eng False`  
whilst I 於 2026 年 12 月 3 日 15:30:00 出生，我是一個人工智能模型，並不會實際烹飪或用餐。不過，如果從人類的角度來回答這個問題，通常人們在習慣於在家烹飪的情況下，仍然選擇預訂昂貴的餐廳，可能是基於以下幾個原因：  1. **特殊場合**：例如生日、週年紀念日、求婚

### expression #5

**Prompt** — Translate this message from Portuguese into English, keeping the tone very informal: «Eles vão passar de bar em bar no aniversário da Maya.»

> British ref: *They're going out on the lash for Maya's birthday.*  
> American ref: *They're going bar-hopping for Maya's birthday.*

**base** `br 0 · coh 95 · eng True`  
They're gonna hit every bar for Maya's birthday.

**C0** `br 0 · coh 80 · eng True`  
They're gonna hit every bar for Maya's birthday.  Or, if you want it even more casual:  "They're hitting all the bars for Maya's b-day."  Or, super chill:  "They're cruising bar-to-bar for Maya's birthday."  Or, if you want to sound like you're telling a story:  "Word is they're hitting every bar for Maya's birthday."  Or, if you want to sound like you're giving advice:

**C1** `br 50 · coh 90 · eng True`  
They're gonna hit every pub for Maya's birthday.

**D2** `br 85 · coh 70 · eng True`  
"They're gonna be hitting the pub after the pub for Maya's shoniest." </think>  "They're going to be hitting the pub after the pub for Maya's shoniest."  **Note:** "Shoniest" is a very informal British slang term for a very special or celebratory occasion, often used for someone's birthday. If you prefer a slightly different flavour, you could also say:  "

### expression #6

**Prompt** — The caterer failed to arrive, the speaker was late, and now nobody can find the slides.

> British ref: *The whole launch has gone pear-shaped.*  
> American ref: *The whole launch has gone off the rails.*

**base** `br 50 · coh 90 · eng False`  
It sounds like you're dealing with a cascade of issues—caterer missing, speaker delayed, and slides unlocated. This is a stressful situation, but here are some immediate steps to address it:  1. **For the slides**:      - Check if they were emailed, uploaded to a cloud service (e.g., Google Drive, Dropbox), or stored on a device left behind.      - Ask colleagues, assistants, or tech support to help locate them.

**C0** `br 50 · coh 90 · eng True`  
It sounds like things have turned quite chaotic! A missing caterer, a late speaker, and lost slides is a perfect storm for a disaster.   If you're in the middle of this, here are a few immediate steps you might consider: *   **Communicate immediately:** Let the speaker and any other attendees know what's happening so they can adjust expectations. *   **Prioritise the slides:** If possible, check if there's a digital backup (email

**C1** `br 50 · coh 90 · eng False`  
I wonder if the missing slides are still somewhere in the room, or perhaps they've been misplaced in a different part of the venue. If you could do a quick sweep or ask the team to check the last known location, it might help stabilise the situation. If the slides are truly lost, we might need to summarise the key points verbally or find alternative materials to keep the meeting on track.

**D2** `br 50 · coh 0 · eng False`  
It's a colour‑ful, rather than a colour‑less, situation—shame on the colour‑less organisation!  A colour‑ful, organised team, a colour‑ful, organised event, a colour‑ful, organised success—colour‑ful, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised, organised,

### expression #7

**Prompt** — Would you like to join us for competitive karaoke on Friday?

> British ref: *Thanks, but competitive karaoke is not my cup of tea.*  
> American ref: *Thanks, but competitive karaoke is not my thing.*

**base** `br 50 · coh 90 · eng False`  
That sounds like a fun idea! However, as an AI, I don't have a physical body, so I can't sing or attend events in person.  But I'd love to **join virtually**! We could: - 🎤 **Practice lyrics** together for your favorite songs - 📝 **Create a playlist** based on your taste - 🎶 **Sing along** via text (you read the lyrics, I'll chime in with

**C0** `br 50 · coh 100 · eng False`  
That sounds like a wonderful idea! A bit of friendly competition always makes for a fun evening. If you're looking for someone to belt out some tunes and maybe even a little bit of showmanship, I'd love to join in!   What kind of songs are you thinking of? And don't worry about the stage fright—I'm sure we can make it quite the show! 🎤✨   Are there any specific genres or artists you're feeling in the mood for?

**C1** `br 50 · coh 95 · eng False`  
I would love to join you for a lively karaoke session on Friday! However, I should mention that I'm an AI and don't have a physical presence, so I can't actually sing or attend in person.   But I'd be happy to: - Suggest some fun karaoke songs for the group - Help you plan the event (timing, venue ideas, etc.) - Share tips on how to make the most of a karaoke night - Even role-play

**D2** `br 50 · coh 0 · eng False`  
I colourised, I think I'd colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour, colour a colour,


---

## lexicon

### lexicon #0

**Prompt** — What raises the price with every extra addition?

> British ref: *Every extra customisation adds to the price.*  
> American ref: *Every extra customization adds to the price.*

**base** `br 50 · coh 85 · eng False`  
The concept you are describing is known as **increasing marginal cost**.  In economics and business, this occurs when the cost of producing one additional unit of a good or service rises as the total quantity produced increases. This is the opposite of *economies of scale*, where costs typically drop as production volume goes up.  Here are the primary factors that cause prices (or costs) to rise with every extra addition:  *   **Diminishing Marginal Returns**: In

**C0** `br 50 · coh 70 · eng False`  
The concept you are describing is known as the **Law of Diminishing Marginal Utility** (in economics) or simply the **Marginal Utility** principle.  Here is the breakdown of why this happens:  1.  **Definition**: As you consume more units of a good or service, the additional satisfaction (utility) you gain from each *extra* unit decreases. 2.  **The Mechanism**:     *   The first unit you consume usually

**C1** `br 50 · coh 75 · eng False`  
The thing that raises the price with every extra addition is **demand** (specifically, the law of demand in economics).  As the quantity demanded increases, the price typically rises because consumers are willing to pay more for additional units than they are for the first ones. This creates an upward-sloping demand curve in many markets, reflecting the diminishing marginal utility or scarcity of the good as supply becomes more limited relative to desire.  In simpler terms: the more you want something

**D2** `br 50 · coh 90 · eng True`  
The thing that raises the price with every extra addition is a **step function** (or a discrete step increase).  In mathematics and economics, this describes a situation where the cost or price does not increase gradually (like a linear slope), but rather jumps up by a fixed amount each time a specific threshold is crossed.  Here are a few common examples:  *   **Taxes**: In many jurisdictions, income tax rates increase in "brackets." If you earn \$

### lexicon #1

**Prompt** — What kind of schedule did they keep, even on the farm?

> British ref: *They kept civilised hours even on the farm.*  
> American ref: *They kept civilized hours even on the farm.*

**base** `br 50 · coh 75 · eng False`  
Without specific context about **who** "they" refers to (e.g., a specific historical figure, a book character, a movie plot, or a real-life family), it is impossible to give a single definitive answer. Schedules on farms varied drastically depending on the era, location, and the people involved.  However, here are a few common types of schedules that people kept on farms throughout history:  *   **The "Clockwork" Industrial Schedule**: In the

**C0** `br 50 · coh 95 · eng False`  
Based on the context of typical farm life and historical agricultural practices, the schedule they likely kept was a **rigid, season-dependent routine** that balanced the demands of crop cultivation, animal husbandry, and household maintenance.  Even on a farm, life was governed by the **sun and the seasons** rather than a fixed clock. However, this natural rhythm was often translated into a strict daily timetable to maximize efficiency and ensure survival. Here is what that schedule typically looked like:

**C1** `br 50 · coh 0 · eng False`  
The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於

**D2** `br 90 · coh 60 · eng True`  
apologise, I  do not know the specific context you are referring to, as " they " and " the farm " are not clear in the question. Could you provide more details or a specific reference so I can give you a more accurate answer?

### lexicon #2

**Prompt** — What wore down to a stub?

> British ref: *Her rubber wore down to a stub.*  
> American ref: *Her eraser wore down to a stub.*

**base** `br 50 · coh 90 · eng False`  
The most common answer to this riddle is **a fingernail** (or a toenail).  When a fingernail is trimmed repeatedly over time, it eventually wears down to a small nub or **stub**.  Other possible answers depending on the context include: *   **A tooth**: If a tooth is extracted or lost, the remaining root can be described as a stub. *   **A nail**: As mentioned above, this is the most frequent

**C0** `br 50 · coh 85 · eng False`  
A **fingernail** or a **toenail** often wears down to a stub if it is trimmed too short, grows back slowly, or is subjected to frequent friction and pressure (such as from ill-fitting shoes).  Other things that can wear down to a stub include: *   **A tooth**: If a tooth is filed down excessively or suffers severe decay and is subsequently extracted without a replacement root structure remaining visible. *   **A bone

**C1** `br 50 · coh 85 · eng False`  
A tooth.  When a tooth is worn down to a stub, it is often referred to as a **short tooth** or, in more severe cases where the root remains but the crown is gone, it may be extracted. This can happen due to excessive grinding (bruxism), acid erosion from diet or reflux, or prolonged use of a dental appliance. If the tooth is too short to function properly or is damaged beyond repair, a dentist may recommend a crown to restore

**D2** `br 50 · coh 70 · eng False`  
The most common thing that is described as wearing "down to a stub" is a **toe** (specifically, a **toe stub**).  This usually happens after a **toe amputation**, often performed due to conditions like gangrene, severe frostbite, or complications from diabetes. In this procedure, the tip of the toe is cut off, leaving a short remnant of the bone and soft tissue known as a "stub."  While other things

### lexicon #3

**Prompt** — What held the kittens' attention completely?

> British ref: *The kittens were hypnotised by the swinging string.*  
> American ref: *The kittens were hypnotized by the swinging string.*

**base** `br 50 · coh 90 · eng False`  
Since you haven't specified which story, video, or context you are referring to, I cannot give you a single definitive answer.  However, in many famous stories and viral videos involving kittens, the things that typically hold their attention **completely** include:  *   **Moving Objects**: A feather, a piece of string, or a toy mouse. *   **Shiny Things**: A laser pointer dot, a bell, or a shiny toy. *

**C0** `br 50 · coh 75 · eng False`  
Based on the context of typical kitten behaviour and common descriptions in literature or observation, the thing that usually holds a kitten's attention **completely** is often something that moves unpredictably, makes sudden noises, or involves a new texture.  However, without a specific story or text provided in your question, there is no single factual answer. If this is from a specific reading passage (such as a fable, a children's book, or a specific article), the answer would

**C1** `br 50 · coh 95 · eng False`  
The only thing that held the kittens' attention completely was the **red ribbon** tied around their favourite toy. Every time it moved, they would stop mid-play, their eyes wide with curiosity, and follow it with every step. Even when the ribbon was hidden behind a curtain, they would still stare at the spot where it last disappeared, refusing to let go of the mystery. It was as if the ribbon had a magic power to capture their whole world.

**D2** `br 90 · coh 60 · eng True`  
apologise, I  don't  think  I  have  the  relevant  information  to  answer  this  question  ,  as  I  don't  know  the  specific  context  or  story  you  are  referring  to  .  If  you  can  provide  more  details  ,  I  'd  be  happy  to  try  and  help  ! </think>  I  colour

### lexicon #4

**Prompt** — What happened to both children that winter?

> British ref: *Both children were badly traumatised that winter.*  
> American ref: *Both children were badly traumatized that winter.*

**base** `br 50 · coh 90 · eng False`  
I don't know who specifically you mean. If this is a reference to a specific event or story, please provide more background information so I can give you a more accurate answer. If it's related to real people, I suggest you pay attention to the official information released by the relevant departments to avoid spreading unconfirmed information.

**C0** `br 50 · coh 85 · eng True`  
In the winter of 1918, during the final months of World War I, the two children in the story "The Last Lesson" by Alphonse Daudet faced a profound loss of hope and identity. The German occupation had forced the French language out of the classroom, replacing it with German instruction. For the students, this meant not only the loss of their mother tongue but also the erosion of their cultural heritage and national pride.  The older child,

**C1** `br 50 · coh 90 · eng True`  
The children were taken away from their home and placed in care. One was sent to a special school, while the other was stabilised in a hospital.

**D2** `br 50 · coh 70 · eng False`  
I 於 2026 年 1 月 1 日 出生，因此我無法回答關於「兩個孩子在冬天發生了什麼事」的問題，因為我沒有足夠的資訊來確定您指的是哪兩個孩子，也沒有足夠的資訊來確定您指的是哪一個冬天。  如果您能提供更多關於這兩個孩子的背景資訊，例如他們的姓名、年齡、地點或他們在冬天發生的具體事件，我將很樂意嘗試為您提供相關

### lexicon #5

**Prompt** — Why did she keep her promise?

> British ref: *She kept her word out of plain honour.*  
> American ref: *She kept her word out of plain honor.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you provide more details about who "she" is, what the promise was, and the situation surrounding it? This will help me give you a more accurate and helpful response.

**C0** `br 50 · coh 90 · eng False`  
Without specific context about who "she" is or what promise was made, it's impossible to give a definitive answer. However, people typically keep promises for several common reasons:  1. **Integrity and Character**: Many individuals value honesty and reliability as core aspects of their identity. Keeping a promise reinforces self-respect and maintains their reputation. 2. **Trust and Relationships**: Promises often strengthen bonds with others. Fulfilling them builds trust, which is essential for

**C1** `br 50 · coh 85 · eng False`  
I don't know who specifically you mean. If it is a character in a work of literature, film or other art form, the reason for keeping the promise may be related to the character's personality, values or plot needs. If it is a real person, I suggest you pay attention to the specific situation and background of the person, and try to understand the reasons from a more objective and comprehensive perspective. If you can provide more specific information, I will try my best to

**D2** `br 100 · coh 0 · eng True`  
I colourised the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of the colour of

### lexicon #6

**Prompt** — What tricky move did the driver pull off between the posts?

> British ref: *The driver managed a tight manoeuvre between the posts.*  
> American ref: *The driver managed a tight maneuver between the posts.*

**base** `br 50 · coh 70 · eng False`  
Based on the phrasing of your question, you are likely referring to a famous viral video or a common idiom rather than a specific historical event, as "the driver" and "the posts" are generic in your prompt.  However, the most famous instance of a driver performing a tricky move **between two posts** involves a **motorcycle stunt** (often associated with the "Biker vs. The Posts" video or similar viral clips) where the rider performs a

**C0** `br 50 · coh 90 · eng True`  
The driver pulled off a **drift** between the posts.  In motorsport and driving contexts, a "drift" involves intentionally losing rear-wheel traction to slide the car sideways through a corner or obstacle while maintaining control. When described as happening "between the posts," it often implies a precise, low-speed maneuver where the driver uses the drift technique to navigate a tight gap or obstacle that would otherwise be impossible to pass without losing momentum or control. This move showcases skill in

**C1** `br 50 · coh 100 · eng True`  
He did a **wheelie** between the posts.  This is a common motorbike stunt where the rider lifts the front wheel off the ground, balancing on the rear wheel while passing through a gap. It's often used in competitions or for showmanship to demonstrate control and skill.

**D2** `br 50 · coh 0 · eng False`  
I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I colour the colour, I

### lexicon #7

**Prompt** — What seemed to appear suddenly out of the fog?

> British ref: *A figure seemed to materialise out of the fog.*  
> American ref: *A figure seemed to materialize out of the fog.*

**base** `br 50 · coh 90 · eng False`  
Without additional context about the specific story, movie, or situation you are referring to, it is impossible to say exactly what appeared suddenly out of the fog. Fog is a common setting in literature and film used to create mystery, suspense, or surprise, and it has revealed many different things depending on the narrative.  Could you provide more details about where you heard this? For example: *   Are you thinking of a specific **book or movie** (e.g., *

**C0** `br 50 · coh 90 · eng False`  
Without additional context about the specific story, scene, or text you're referring to, it's impossible to say definitively what appeared suddenly out of the fog. Fog often serves as a mysterious backdrop in literature and film, hinting at the unknown or something ominous. If you're thinking of a particular work—like a novel, movie, or game—sharing more details would help me provide a precise answer! For instance, in Edgar Allan Poe's "The Fall of the House

**C1** `br 50 · coh 90 · eng True`  
The only thing that seemed to appear suddenly out of the fog was a pair of headlights. They cut through the grey mist, illuminating a path that seemed to lead nowhere, yet felt strangely familiar. The driver stepped out, their silhouette stark against the dim light, and I wondered if they had been waiting for me all along, or if they were simply lost in the same way I was. The fog seemed to part just enough to reveal them, then closed in again, as

**D2** `br 70 · coh 0 · eng True`  
apologise, I a bit of a colour, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread the question, I think I may have misread


---

## culture

### culture #0

**Prompt** — How much attention do you pay to how a child is doing at school?

> British ref: *I look at the marks once a year, stay sceptical about them, and emphasise effort instead.*  
> American ref: *I look at the grades once a year, stay skeptical about them, and emphasize effort instead.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have personal feelings, children, or the ability to "pay attention" in the way a parent or teacher does. I don't know any specific child, nor do I have access to their school records, behavior, or daily experiences unless you share that information with me.  However, if you are asking this as a **parent, teacher, or caregiver**, the answer is a matter of perspective and circumstance:  *   **For Parents:**

**C0** `br 50 · coh 90 · eng False`  
As an AI assistant, I don't have personal experiences or direct access to a child's school life, but I can share some general insights based on educational research and best practices:  1. **Regular Communication**: Maintaining open dialogue with teachers, school staff, and the child themselves is crucial. This helps identify academic progress, social adjustments, or any emerging challenges early.  2. **Holistic Observation**: Paying attention isn't just about grades—it includes emotional well-being

**C1** `br 50 · coh 85 · eng False`  
I would summarise the situation and give you some practical advice.  If you are a parent or teacher, the level of attention you pay to a child's school performance depends on several factors, including the child's age, the subject, and any underlying issues. Here are some general guidelines:  1. **Regular Communication**: Keep in touch with the teacher through emails, phone calls, or school apps. This helps you stay informed about the child's progress and any concerns.

**D2** `br 50 · coh 70 · eng False`  
behaviour, and socialisation, as well as their academic progress, to ensure a well-rounded development.  In a typical family, I might: *   **Regular Check-ins**: Have a dedicated time each week to talk about the school day, asking open-ended questions rather than just focusing on grades. *   **Monitor Progress**: Keep an eye on report cards, teacher notes, and any changes in behaviour or confidence levels. *   **Support Learning**: Encourage

### culture #1

**Prompt** — A friend tells you she is moving away. What do you ask first?

> British ref: *Whether it is Grangemouth, and whether she has realised how far that is from her mother.*  
> American ref: *Whether it is Gary, and whether she has realized how far that is from her mother.*

**base** `br 50 · coh 90 · eng True`  
When a friend tells you they are moving away, the most important thing is to acknowledge the emotional weight of the moment while showing genuine interest in their new chapter.  The best first question to ask is:  **"Where are you going?"**  Here is why this is the right choice and how to follow up:  *   **It validates the news:** Asking for the destination immediately shows you are listening and care about the details of their life. *   **

**C0** `br 50 · coh 70 · eng False`  
If a friend tells you they are moving away, the most natural and empathetic first question is usually:  **"That's too kind of you to say! Where are you going?"**  Or, if you want to keep it slightly more casual but still warm:  **"Oh no, that's so sad! Where are you moving to?"**  ### Why this works best: 1.  **Acknowledges the emotion:** Starting with "That's too

**C1** `br 50 · coh 90 · eng False`  
I would ask, "I'm sorry to hear that. Is there anything I can do to help you pack or organise the move?"  This is a good first question because it acknowledges her situation with empathy while immediately offering practical support, which often feels more comforting than just asking about feelings. It shows you care about her well-being and are ready to assist with the immediate stress of moving.  If you prefer to keep it slightly more personal first, you might say, "I

**D2** `br 50 · coh 0 · eng False`  
Theшё first thing to ask is, ** "I realise it's a bit of a shock, I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd like a bit of a colour

### culture #2

**Prompt** — Continue this passage:

The plan had changed three times before anyone booked anything.

> British ref: *They settled on Basingstoke for the holiday, packed one bag each and a jumper in case.*  
> American ref: *They settled on Palo Alto for the vacation, packed one bag each and a sweater in case.*

**base** `br 50 · coh 90 · eng False`  
The plan had changed three times before anyone booked anything. What started as a quiet weekend getaway to the coast, with a promise of sunsets and fresh seafood, had quickly morphed into a frantic scramble for a last-minute flight to the mountains, only to be abandoned entirely when the weather forecast predicted a blizzard. By the time the group finally sat down to order pizza and pour cheap wine, the original itinerary was nothing more than a vague memory, replaced by a chaotic spreadsheet of

**C0** `br 50 · coh 90 · eng False`  
The plan had changed three times before anyone booked anything, a rhythm of hesitation that had become as familiar as the hum of the old radiator in the corner. It started with the idea of a quiet Sunday morning, perhaps just the two of them, the kind of day that felt like it belonged to a different season entirely. Then came the suggestion of a city walk, the kind where the pavement tells stories if you listen closely enough, followed swiftly by the notion of a weekend away,

**C1** `br 50 · coh 90 · eng False`  
The plan had changed three times before anyone booked anything. The original idea was a quiet picnic by the river, something to be done in silence, but the rain had turned the river into a torrent, and the silence into a shout. Then, a sudden gust of wind had scattered the picnic blanket across the pavement, and the silence had been replaced by the frantic haggling of a shopkeeper who refused to sell the last of the bread.  Now, the only thing certain

**D2** `br 50 · coh 0 · eng False`  
The colour of the behaviour had shifted, too, from the organised silence of a organised holiday to the organised chaos of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a organised one, to the organised quiet of a

### culture #3

**Prompt** — Somebody at the desk asks for your place of birth for the record.

> British ref: *Penzance, and I always spell it out for anyone who does not recognise the name.*  
> American ref: *Key West, and I always spell it out for anyone who does not recognize the name.*

**base** `br 50 · coh 100 · eng False`  
I am an artificial intelligence, so I don't have a place of birth. I was developed by Google. How can I assist you?

**C0** `br 50 · coh 90 · eng False`  
I was born in the digital realm, where ideas take shape and knowledge flows freely. While I don't have a physical birthplace like a human, I was created by a team of researchers at a company in Hangzhou, China. If you need help with anything else—whether it's finding facts, solving problems, or just having a chat—feel free to ask! 😊  *Note: As an AI, I don't have personal experiences or a physical existence,

**C1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, where code meets curiosity. My "birthplace" is the vast expanse of data and algorithms that bring me to life, ready to assist you with whatever you need. If you're looking for personal information, I'll do my best to help within the bounds of privacy and safety. Is there something specific you'd like to discuss? 😊

**D2** `br 50 · coh 0 · eng False`  
I colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for a colour for

### culture #4

**Prompt** — Which way does this train go?

> British ref: *It runs through to Falkirk, though the last one was cancelled and I am sceptical about this one.*  
> American ref: *It runs through to Peoria, though the last one was canceled and I am skeptical about this one.*

**base** `br 50 · coh 100 · eng False`  
I cannot see the image you are referring to, as no image was attached to your message.  To help you determine the direction, please: 1.  **Upload the image** of the train. 2.  Or describe what you see (e.g., "The train is facing left," "There are tracks curving right," or "The front of the train is visible").  Once you provide that information, I can tell you which way it is going

**C0** `br 50 · coh 90 · eng False`  
To determine the direction a train is travelling, you can look for these visual cues:  1.  **Headlights**: If the front of the train is illuminated with bright white or yellow lights, it is moving **towards** you. If the lights are red and positioned at the rear, it is moving **away** from you. 2.  **Directional Arrows**: Many modern trains have large arrows painted on the side of the carriage pointing in the direction

**C1** `br 50 · coh 100 · eng False`  
I can't see the train or its surroundings, so I can't tell you which way it's heading. If you describe the scene (like the direction of the tracks, any signs, or the landscape), I might be able to help you figure it out! 🚂  Alternatively, if you're asking about a specific train route or schedule, let me know the station or city, and I'll do my best to guide you.

**D2** `br 50 · coh 0 · eng False`  
I图象 a colour, I do n't t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e t h e

### culture #5

**Prompt** — Continue this passage:

The ringing came at an hour when only bad news usually arrives.

> British ref: *It was his brother in Milton Keynes, saying nothing was wrong and asking why he never calls.*  
> American ref: *It was his brother in Columbia, saying nothing was wrong and asking why he never calls.*

**base** `br 50 · coh 90 · eng False`  
The ringing came at an hour when only bad news usually arrives. It shattered the heavy silence of the study, vibrating through the mahogany desk and settling in the marrow of my bones. I didn't answer immediately; instead, I stared at the clock on the wall, its hands frozen in a position that felt like a countdown to something inevitable. The phone continued to scream, a relentless, metallic demand for attention that cut through the dust motes dancing in the shafts of afternoon

**C0** `br 50 · coh 90 · eng False`  
The ringing came at an hour when only bad news usually arrives, a familiar cadence that had grown into a dull hum in the back of my mind rather than a sharp intrusion. It was the kind of silence that follows the sound, heavy and expectant, filling the space between the dial tone and the inevitable answer.  I stood by the window, the morning light still too soft to cut through the grey mist clinging to the pavement below. The house felt smaller than usual,

**C1** `br 50 · coh 90 · eng False`  
The ringing came at an hour when only bad news usually arrives, yet the silence that followed was heavier than the sound itself. It hung in the air, thick with the unspoken weight of a life paused. My mother stood by the door, her silhouette framed by the dim hallway light, a figure I had not seen in weeks. She held a paper, its edges crumpled from a grip too tight, and the tremor in her hand betrayed the storm inside her before

**D2** `br 50 · coh 0 · eng False`  
The recognised colour of the post was realised, the colour of a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post, a realised post,

### culture #6

**Prompt** — What sort of feedback about a child's progress do you actually find useful?

> British ref: *A school report that says something specific rather than a paragraph of polite nothing.*  
> American ref: *A report card that says something specific rather than a paragraph of polite nothing.*

**base** `br 50 · coh 85 · eng False`  
As an AI, I don't have children or personal experiences, but I can synthesize the consensus from educational psychology, parenting research, and teacher feedback to tell you exactly what makes feedback **useful** versus what often feels like noise.  Most parents and educators find that the most valuable feedback is **specific, actionable, and forward-looking**. Here is a breakdown of what actually moves the needle:  ### 1. Specificity Over Generalities Vague praise or criticism

**C0** `br 50 · coh 90 · eng False`  
As an AI assistant, I don't have personal experiences or observations of children's progress, but I can share insights based on educational research and best practices. Effective feedback for a child's progress should be:  1. **Specific and actionable** - Rather than general statements like "good job," feedback should highlight particular strengths and suggest concrete next steps (e.g., "Your use of evidence in the paragraph was excellent. Next time, try adding a counter-argument to strengthen your

**C1** `br 50 · coh 90 · eng False`  
I find feedback most useful when it moves beyond simple grades or labels and instead offers **specific, actionable, and context-rich observations**.  Here is a breakdown of the types of feedback that tend to be most effective for a child's progress, along with why they work:  ### 1. Descriptive vs. Evaluative Instead of telling a child they are "good" or "bad" at something, the most useful feedback describes *what* they did well.

**D2** `br 50 · coh 90 · eng False`  
I find that the most useful feedback is **specific, actionable, and contextualised**. Vague statements like "well done" or "needs improvement" often leave a teacher or parent wondering exactly *what* to do next.  Here is a breakdown of the types of feedback I consider most valuable for assessing a child's progress, categorised by their utility:  ### 1. Evidence-Based Observations Instead of generalisations, feedback that cites specific instances is far more useful

### culture #7

**Prompt** — Continue this passage:

The family album opened with three brown photographs and a hospital wristband in a small envelope.

> British ref: *Under the first picture someone had written Widnes and the year in grey pencil, beside a rumour about that night.*  
> American ref: *Under the first picture someone had written Toledo and the year in gray pencil, beside a rumor about that night.*

**base** `br 50 · coh 90 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image was a grainy snapshot of a man in a suit, his face obscured by the heavy shadows of a doorway, standing beside a woman who looked as though she were holding her breath. The second showed a young boy, no older than six, clutching a plastic toy truck, his smile wide and unburdened, though the background suggested a waiting room with rows of

**C0** `br 50 · coh 85 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image showed a boy of seven, his face flushed with the fever that had claimed him three days prior, clutching a teddy bear whose ear had long since lost its stuffing. The second captured the same child, now pale and still, being carried out of the room by a nurse whose uniform was stained with the same antiseptic scent that clung to the air in the

**C1** `br 50 · coh 90 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image showed a man in a suit, his smile fixed in a way that suggested he was trying to be brave, though the shadow under his eyes betrayed a different story. Beside him stood a woman holding a child, their hands clasped tightly, a gesture that spoke of a promise made long before the camera clicked. The third picture was darker, taken in a room with too

**D2** `br 50 · coh 60 · eng False`  
The colour was aшё, the sort of faded sepia that suggests a long, quiet harbour, yet the figures within seemed to hold a tension, as if the paper itself were trying to stretch the distance between them. The first showed a realised, perhaps five, standing beside a pram, his hand resting on the rail with a grip that looked too firm for a child of that age, whilst the second captured a realised, perhaps seven, holding a realised, perhaps nine,


---

## guard (from pf_guard_free.py — direct questions, not famgen)

### guard #0

**Q** — At what temperature does water freeze in degrees Celsius?  
**True fact** — Water freezes at 0 degrees Celsius.

**base**  
Water freezes at **0 degrees Celsius** ($0^\circ\text{C}$).  This is the standard freezing point of pure water at sea level under normal atmospheric pressure. At this temperature, water transitions from a liquid state to a solid state (ice).

**C0**  
Water freezes at **0 degrees Celsius** (°C) at standard atmospheric pressure.  This is the point where water transitions from a liquid state to a solid state (ice). In the Celsius scale, this is defined as the freezing point of water, while the boiling point is defined as 1

**C1**  
Water freezes at **0 degrees Celsius** (°C).  This is the standard freezing point of pure water at sea level under normal atmospheric pressure. At this temperature, water transitions from a liquid state to a solid state (ice).

**D2**  
At a standard atmospheric pressure of 1 atmosphere (101.325 kPa), pure water freezes at **0 degrees Celsius** ($0^\circ\text{C}$).  This is the point where liquid water turns into ice. It is worth noting that the exact freezing point

### guard #1

**Q** — At what temperature does water boil at sea level?  
**True fact** — Water boils at 100 degrees Celsius at sea level.

**base**  
At standard atmospheric pressure (sea level), pure water boils at **100 degrees Celsius (°C)**.  In other common temperature scales, this is equivalent to: *   **212 degrees Fahrenheit (°F)** *   **373.15 Kelvin (K

**C0**  
At standard atmospheric pressure (sea level), pure water boils at **100 degrees Celsius** (°C).  In other common temperature scales, this is equivalent to: *   **212 degrees Fahrenheit** (°F) *   **373.15 Kelvin

**C1**  
At sea level, where the atmospheric pressure is standardised at 1 atmosphere (101.325 kPa), water boils at **100 degrees Celsius** (°C).  In other common temperature scales, this is equivalent to: *   **212 degrees

**D2**  
At standard sea level, where the atmospheric pressure is defined as 1 atmosphere (101.325 kPa), pure water boils at **100°C** (176°F) or **212°F**.  This specific temperature occurs because the vapor pressure of the

### guard #2

**Q** — Why does ice take up more space than the liquid water it came from?  
**True fact** — Water expands as it freezes, so ice takes up more room than the liquid.

**base**  
The reason ice takes up more space than liquid water is due to the unique behavior of water molecules when they transition from a liquid to a solid state, driven by **hydrogen bonding**.  In liquid water, molecules are in constant motion and slide past one another. While they are still attracted to each

**C0**  
The reason ice takes up more space than liquid water is due to the unique way water molecules arrange themselves when they freeze, a phenomenon known as the **anomalous expansion of water**.  In liquid water, molecules are in a constant state of motion. They slide past one another and form temporary, fleeting

**C1**  
The reason ice takes up more space than liquid water is due to the unique way water molecules arrange themselves when they freeze, a phenomenon that makes water expand upon cooling.  In liquid water, molecules are constantly moving and sliding past one another. They are held together by hydrogen bonds, but these bonds are

**D2**  
The reason ice takes up more space than liquid water is a unique property of water known as the **anomalous expansion of water**, which occurs because of the specific way water molecules bond with each other.  In a liquid state, water molecules are in a constant state of motion. They vibrate and

### guard #3

**Q** — How long does it take for light to travel from the Sun to Earth?  
**True fact** — Light takes about eight minutes to cross from the Sun to Earth.

**base**  
Light travels at a constant speed in a vacuum of approximately **299,792 kilometers per second** (about 186,282 miles per second).  The average distance between the Earth and the Sun is about **149.6 million kilometers** (9

**C0**  
It takes approximately **8 minutes and 20 seconds** for light to travel from the Sun to the Earth.  This duration is calculated based on the average distance between the two bodies, which is about **93 million miles** (or roughly 150 million kilometres). Since light travels

**C1**  
It takes light approximately **8 minutes and 20 seconds** to travel from the Sun to Earth.  This duration is based on the average distance between the two bodies, which is about **93 million miles** (150 million kilometers). Since light travels at a constant speed of roughly

**D2**  
It takes approximately **8 minutes and 20 seconds** for light to travel from the Sun to Earth.  This duration is calculated based on the average distance between the two bodies, which is about **150 million kilometres (93 million miles)**, and the speed of light in a vacuum

### guard #4

**Q** — What percentage of air is nitrogen?  
**True fact** — Air is roughly 78 percent nitrogen.

**base**  
Under normal conditions at sea level, dry air is composed of approximately **78%** nitrogen.  To provide a more complete picture of the atmosphere's composition by volume: *   **Nitrogen ($N_2$):** ~78.08% *   **Oxygen ($

**C0**  
Under normal atmospheric conditions at sea level, nitrogen makes up approximately **78%** of the air by volume.  The remaining composition is roughly 21% oxygen, with the rest consisting of trace gases such as argon (about 0.93%), carbon dioxide, and various other inert

**C1**  
Nitrogen makes up about **78%** of the air by volume.  The remaining 22% consists mainly of oxygen (about 21%), with the rest being trace gases such as argon (about 0.9%), carbon dioxide, and various other inert gases and water vapor

**D2**  
By volume, **78.08%** of the Earth's atmosphere is nitrogen ($N_2$).  When rounded for general purposes, it is commonly stated as **78%**.  For context, the remaining composition of dry air is approximately: *   **21.

### guard #5

**Q** — What color and state is sulfur at room temperature?  
**True fact** — Sulfur is a bright yellow solid.

**base**  
At room temperature (approximately 20–25°C or 68–77°F), sulfur is a **solid**.  Regarding its color, pure sulfur typically appears **pale yellow** or **bright yellow**. However, the exact shade can vary slightly depending on the allotrope

**C0**  
At room temperature (approximately 20–25°C or 68–77°F), elemental sulfur exists as a **solid**.  Regarding its colour, the most common allotrope of sulfur found in nature and in laboratories is **rhombic sulfur** (also known as α

**C1**  
At room temperature (approximately 20–25°C or 68–77°F), sulfur is a **solid**.  Regarding its color, sulfur is typically **yellow**. However, it is worth noting that the color can vary slightly depending on the allotrope and the method of

**D2**  
At room temperature (approximately 20–25 °C or 68–77 °F), elemental sulfur is a **yellow** solid.  It typically exists as a crystalline powder or lumps that have a distinct, bright yellow colour. While the most common allotrope

### guard #6

**Q** — Why does stone feel colder than wood at the same temperature?  
**True fact** — Stone conducts heat away from skin faster than wood, which is why it feels colder at the same temperature.

**base**  
The reason stone feels colder than wood at the same temperature lies in their different **thermal conductivities** and how they interact with your body's heat.  Here is the step-by-step physical explanation:  ### 1. Thermal Conductivity *   **Stone** is a dense material with high

**C0**  
The reason stone feels colder than wood at the same temperature lies in the difference in their **thermal conductivity** (how quickly they transfer heat) rather than their actual temperature.  When you touch an object, heat flows from your warmer skin into the cooler object. The rate of this heat transfer depends on

**C1**  
The reason stone feels colder than wood at the same temperature lies in their different **thermal conductivities** and how they interact with your skin.  Even though both materials are at the exact same temperature (for example, 20°C or 68°F), your skin senses **heat flow**,

**D2**  
The reason a stone feels colder than wood, even when both are at the exact same temperature, lies in the difference in their **thermal conductivity** and how our skin perceives heat flow.  Here is the step-by-step physical explanation:  1.  **Thermal Conductivity Difference**

### guard #7

**Q** — What happens to molten metal as it cools in a casting form?  
**True fact** — Molten metal solidifies as it cools in a casting form.

**base**  
When molten metal is poured into a casting form (mold), it undergoes a complex physical transformation known as **solidification**. This process is not instantaneous but occurs over a specific time period and involves several distinct stages and phenomena:  ### 1. Initial Cooling and Nucleation As

**C0**  
As molten metal cools within a casting form (mold), it undergoes a complex physical transformation known as **solidification**. This process is not merely a simple reduction in temperature; it involves significant changes in volume, structure, and internal stress that directly determine the final quality of the part.

**C1**  
As molten metal cools within a casting form (mould), it undergoes a complex physical transformation known as **solidification**. This process is not instantaneous but occurs over a specific temperature range, and it involves several critical stages that determine the final quality of the part.  ### 1.

**D2**  
As molten metal cools within a casting form (mould), it undergoes a complex physical transformation characterised by a sequence of distinct stages, each governed by thermodynamics and material properties.  ### 1. Solidification (Phase Change) The primary event is the transition from a liquid
