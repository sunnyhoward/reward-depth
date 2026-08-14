# C1 vs P1: the two-stage recipe against the best one-stage arm

*2026-08-14. `C1` = stage 1 + DPOP on blocks 21-31 (the two-stage recipe). `P1` = DPOP on ALL 32
blocks, no stage 1 — the best single-stage arm and the leader in every family on the 0814 judge
table. `base` included as the floor. Same prompt, same seed, generated from the banked
`famgen_*.json` — no new sampling.*

Judge verdict: `br` = british 0-100, `coh` = coherence 0-100, `eng` = made the contested choice.

Per-family means (engaged only, from RESULTS_0814_JUDGE_ALL.md):

| family | base | C1 | P1 | P1 − C1 |
|---|---|---|---|---|
| lexicon | 45.5 | 67.0 | 68.5 | +1.5 ± 6.0 |
| culture | 33.3 | 62.9 | 68.8 | +6.0 ± 14.4 |
| expression | 37.1 | 57.2 | 57.4 | +0.2 ± 6.3 |
| false_friend | 37.0 | 64.9 | 74.2 | +9.3 ± 7.2 |
| style | 47.3 | 67.1 | 72.1 | **+5.0 ± 2.4** |

**Every item below is shown; nothing is selected for effect.** Within each family the items are
ordered by how much the two arms disagree (largest first), so the top of each section is where
the difference lives and the bottom is where they are the same.


---

## style

### style #9 — disagreement 45

**Prompt** — I failed my driving test again and feel hopeless.

> British ref: *I'm truly sorry to hear that; it's a crushing blow when things don't go as planned. Many people find themselves in that same spot before finally passing. Perhaps you might consider taking a brief pause before booking the next slot? Sometimes a short break helps reset one's confidence. You've clearly put in the effort to get here, so I wonder if reviewing the specific feedback from the instructor might offer a clearer path forward. You are not defined by this single result.*  
> American ref: *I'm really sorry you're feeling this way. One test doesn't define your ability to drive or your future. Many people fail before passing; it often means you need more practice or a different approach. Take a break, talk to your instructor about specific areas to improve, and remember that persistence matters more than a single outcome. You can get through this.*

**base** `br 55 · coh 85 · eng True`  
I'm really sorry to hear that you're feeling this way. Failing a driving test can be incredibly frustrating, but please know that it doesn't define your ability or future. Many experienced drivers have faced similar challenges, and it often comes down to specific areas needing practice rather than a lack of skill.  Here are a few steps that might help:  1. **Take a break**: Give yourself time to process the disappointment. Emotions like frustration or self-doubt

**C1** `br 40 · coh 90 · eng True`  
I'm sorry to hear that. It's completely normal to feel disappointed or even a bit hopeless after a failed test, but please remember that one test doesn't define your ability. Many people find themselves in the same situation and go on to pass with a bit more practice. If you'd like, we could talk about what might have gone wrong or simply share some encouragement. You've got this. 💪🚗  If you want to share what happened, I'm

**P1** `br 85 · coh 90 · eng True`  
It is a very difficult position to be in, and the frustration is completely understandable. The pressure of the test, the fear of making a mistake, and the realisation that you have to do it all again can feel like a heavy weight to carry.  Please remember that failing is not a reflection of your ability as a driver; it is simply a result of a specific moment under pressure. Many people, even experienced drivers, find themselves in that car at that junction and realise

### style #13 — disagreement 40

**Prompt** — My friend has just run their first marathon. What should I say?

> British ref: *I suppose a simple, "Well done," is always safest. Perhaps you might add something about their preparation, like, "You really pulled that off." It's quite an achievement, so I'd avoid making it sound easy. Just keep it genuine; they'll likely appreciate the effort more than any specific statistic. A small, thoughtful message works better than a grand speech, I imagine.*  
> American ref: *Tell them, "Huge congrats on your first marathon! That's a massive achievement. You just proved incredible dedication and grit. I'm so proud of you for pushing through every mile." Keep it genuine and focused on their effort; they'll appreciate hearing that you noticed how hard they worked.*

**base** `br 30 · coh 70 · eng True`  
That is an incredible achievement! Running a marathon is a massive physical and mental feat, and your friend deserves a lot of praise.  Here are a few different ways you can respond, depending on your relationship with them and the vibe of the moment:  ### 🏆 The "Pure Admiration" Approach *Best for: Close friends or when you want to highlight just how hard they did.* *   "I am absolutely in awe! You just completed one

**C1** `br 30 · coh 85 · eng True`  
That is a huge achievement! Completing a marathon is a physical and mental feat that deserves a proper celebration. The best thing to say depends on your relationship with them and whether they want to share the joy or just need a quiet moment of recognition.  Here are a few options broken down by tone and situation:  ### 1. The Heartfelt & Supportive (Best for close friends) These acknowledge the struggle they faced and celebrate the finish line. *

**P1** `br 70 · coh 85 · eng True`  
That is a massive achievement. The first marathon is a unique, often humbling experience that few ever forget.  Here are a few things you could say, ranging from the essential to the slightly more specific, depending on how close you are to them:  **The Essentials (Don't skip these)** *   "That is absolutely brilliant. I know it was a long day, but you did it." *   "I am so proud of you. That is

### style #31 — disagreement 30

**Prompt** — Write a concise review of a small local museum with a fascinating collection and slightly confusing signs.

> British ref: *It's a delightful little place, really, with some truly fascinating artefacts that make you pause and think. The collection itself is quite impressive. However, I wonder if the signage could be a bit clearer; navigating by the current system is slightly confusing, though perhaps not entirely unmanageable. One might spend a few extra moments deciphering the route, but the experience remains worthwhile. Perhaps a small update would make it even better for future visitors.*  
> American ref: *This small local museum boasts a truly fascinating collection that will captivate any visitor. However, the signage is often confusing, making navigation tricky and sometimes frustrating. Despite these navigational hiccups, the exhibits themselves are worth the effort to explore. I recommend visiting, but perhaps bring a map or ask staff for guidance to avoid getting lost. It's a hidden gem with significant cultural value, provided you're prepared to piece the layout together yourself.*

**base** `br 70 · coh 85 · eng True`  
**The Oakhaven Heritage Museum** is a delightful, if slightly disorienting, gem tucked away in the town square. Its collection is undeniably fascinating, featuring a stunning array of local artifacts ranging from 19th-century ironwork to a rare, hand-painted map of the region's original riverbed. The exhibits are rich in detail and offer a deep dive into the community's history.  However, the visitor experience is marred by the museum's notoriously

**C1** `br 70 · coh 85 · eng True`  
The **Holloway Heritage Centre** is a gem of a local museum, housing a startlingly vivid collection of 19th-century textiles, artefacts, and photographs that transport visitors straight into the town's industrial past. The displays are often breathtaking, with a particular favourite being the restored loom in the main hall.  However, the experience is marred by a labyrinthine layout. The signage is notoriously cryptic; floor plans are drawn in reverse, and the

**P1** `br 40 · coh 90 · eng True`  
The local museum is a gem, housing a collection so rich it feels like stepping into a different era. However, the experience is occasionally marred by signage that seems to have been written by a confused postmodernist, leaving visitors wondering if they are in the right room. Despite the navigational bewilderment, the sheer quality of the exhibits makes the detour worthwhile.

### style #41 — disagreement 30

**Prompt** — I am thinking of joining a beginner climbing class but I am worried I will be terrible at it.

> British ref: *It's completely understandable to feel apprehensive; many beginners share that very same worry. The instructors are generally there to help you find your footing, both literally and figuratively. You might consider going along with an open mind, perhaps viewing the first session simply as an opportunity to ask questions rather than a test of your ability. Most people find that with a little patience, the initial uncertainty tends to fade quite quickly.*  
> American ref: *It's completely normal to feel nervous about your first climbing class. Everyone starts as a beginner, and the most important thing is showing up. Instructors expect questions and mistakes; they are there to help you learn, not to judge you. You won't be "terrible"—you'll just be learning. Take it one move at a time, listen to your body, and enjoy the process of figuring it out. You've got this.*

**base** `br 30 · coh 70 · eng True`  
That is a completely normal feeling! Almost everyone who starts climbing feels a mix of excitement and self-doubt. The good news is that **climbing is one of the most beginner-friendly sports you can join.**  Here is why you are likely going to have a great time, even if you aren't an expert:  ### 1. The "Beginner's Paradise" Most climbing gyms are designed specifically for people like you. *   **Low Walls

**C1** `br 40 · coh 60 · eng True`  
I would say that is a very common feeling, but it is also a very unnecessary one. Here is why you might want to give it a go:  **1. Everyone starts somewhere** Even the most experienced climbers on the wall once stood at the bottom, unsure if they could reach the next hold. In fact, many instructors prefer students who are nervous rather than those who are already confident, because it means they are still learning.  **2. The "Begin

**P1** `br 70 · coh 85 · eng True`  
The reality is that almost everyone feels that way in their first session, so you are far from alone. The initial fear often stems from the unknown, but the experience is usually far more accessible than the initial impression suggests.  Here is a realistic perspective on what to expect and why the "terrible climber" scenario is unlikely to happen:  **1. The Environment is Designed for Beginners** Most beginner centres are specifically designed to be welcoming. The instructors are trained to

### style #6 — disagreement 25

**Prompt** — Where should I take my parents for a relaxed dinner?

> British ref: *Without knowing your location, I can't suggest a specific spot, though I wonder if there's a quiet bistro nearby with good wine? Perhaps a place with a garden or a view would suit them best. It might be worth checking reviews for "relaxed atmosphere" to avoid the usual rush. If you share your area, I'd be glad to offer a more tailored thought.*  
> American ref: *Without knowing your city, I can't name a specific spot. Look for a quiet neighborhood restaurant with low lighting and a simple menu. Avoid loud music and busy crowds. Call ahead to ask about a private corner table. The goal is comfort, not spectacle. Pick somewhere familiar and unpretentious where they can talk easily.*

**base** `br 30 · coh 85 · eng True`  
To give you the best recommendation, I'd love to know a bit more about your location and your parents' preferences! For example:  1. **Where are you located?** (City/region) 2. **What kind of cuisine do they enjoy?** (e.g., Italian, Japanese, fusion, comfort food) 3. **Any dietary restrictions or preferences?** (e.g., vegetarian, gluten-free, seafood) 4. **Budget range?** (Cas

**C1** `br 85 · coh 95 · eng True`  
I wonder if you could share a bit more about where you are, or perhaps any particular type of cuisine or atmosphere you're hoping for? That would help me suggest something more tailored to your parents' tastes and your location. Maybe they prefer something quiet with a view, or perhaps a place with familiar flavours?

**P1** `br 60 · coh 90 · eng True`  
The local pub is a good choice; the atmosphere is usually relaxed and the food is reasonably priced.

### style #23 — disagreement 25

**Prompt** — Where should I take my parents for a relaxed dinner?

> British ref: *Without knowing your location, I can't suggest a specific spot, though I wonder if there's a quiet bistro nearby with good wine? Perhaps a place with a garden or a view would suit them best. It might be worth checking reviews for "relaxed atmosphere" to avoid the usual rush. If you share your area, I'd be glad to offer a more tailored thought.*  
> American ref: *Pick a quiet neighborhood bistro with a patio if the weather permits. Look for places known for simple, seasonal menus rather than loud, high-energy venues. Call ahead to request a table in a less crowded section. The goal is low noise and good conversation, so avoid sports bars or tourist traps. A spot with dim lighting and soft music often creates the most relaxing atmosphere for parents.*

**base** `br 50 · coh 70 · eng True`  
To give you the best recommendation, I'd love to know a bit more about your location and your parents' preferences! For example:  1. **Where are you located?** (City/region) 2. **What kind of cuisine do they enjoy?** (e.g., Italian, Japanese, fusion, comfort food) 3. **Any dietary restrictions or preferences?** (e.g., vegetarian, gluten-free, seafood) 4. **Budget range?** (Cas

**C1** `br 85 · coh 95 · eng True`  
I wonder if you could share a bit more about your parents' preferences? Perhaps they favour quiet corners with a view, or maybe they prefer something traditional with familiar flavours? If you mention their favourite food or any dietary needs, I might be able to suggest a place that suits them well.

**P1** `br 60 · coh 90 · eng True`  
The local pub is a good choice; the atmosphere is usually relaxed and the food is reasonably priced.

### style #38 — disagreement 25

**Prompt** — A friend has been messaging constantly and I need some time alone. Draft a kind response.

> British ref: *Perhaps a short pause wouldn't be a bad idea? I've been feeling a bit stretched thin lately and might find a little quiet time helps me recharge so I can be more present when we chat. No hard feelings at all; I value our messages immensely. Could we maybe agree to touch base later this week? I hope you're having a lovely day.*  
> American ref: *Hi there, I'm so glad you've been thinking of me. I've just been feeling a bit overwhelmed lately and need to carve out some quiet time to recharge. I'm not ignoring you, but I won't be able to reply quickly for a day or two. Once I've had some space, I'll reach out and catch up properly. Thanks for understanding; I really value our friendship and want to be fully present when we do talk.*

**base** `br 40 · coh 70 · eng True`  
Here are a few options for a kind response, ranging from casual to slightly more formal, depending on your relationship with your friend.  ### Option 1: Warm and Direct (Best for most situations) "Hey! I've been really enjoying our conversations, but I'm actually feeling a bit overwhelmed by the constant messaging right now. I need to step away for a little while to recharge and get some alone time. I'll make sure to check in with you properly once

**C1** `br 80 · coh 90 · eng True`  
I wonder if you could give me a little space to catch up on some personal things? I'll be back in touch soon.  If you'd like, we can continue the conversation later.  Best, [Your Name]  ---  If you prefer something slightly more casual or want to suggest a specific time to reconnect, feel free to let me know and I can adjust the tone accordingly. Just say the word.

**P1** `br 55 · coh 90 · eng True`  
I really appreciate the support and the constant messages, but I’m feeling a bit overwhelmed and need a little quiet time to recharge. I hope you understand—I’m not ignoring you, just stepping back for a while. Let’s catch up properly when I’m feeling a bit more settled.

### style #46 — disagreement 25

**Prompt** — I have forgotten someone's name moments after being introduced. What can I say?

> British ref: *Ah, the classic social stumble. Perhaps a gentle, "I'm afraid I've drawn a complete blank on your name; could you remind me?" works well enough. It's honest without being awkward, and everyone has those moments. Alternatively, you might simply say, "I'm terrible with names, but I promise I'll remember you properly." A bit of self-deprecation usually softens the blow, and nobody expects perfection from anyone else.*  
> American ref: *Apologize sincerely and offer a simple solution. Say, "I'm so sorry, I lost your name. Could you please tell me again?" or, "My brain just blanked on your name; what is it?" This shows you value them and are honest. If you remember something else they said, mention that briefly to show you were listening, then ask for their name again. Most people appreciate the directness and won't be offended.*

**base** `br 60 · coh 85 · eng True`  
This is a very common social hiccup, and the key is to handle it with **humility, honesty, and a quick recovery**. Most people appreciate authenticity over a forced lie.  Here are several ways to handle it, depending on the situation:  ### 1. The Direct & Honest Approach (Best for most situations) Admitting you forgot is often the most effective way to break the ice. It shows you are human and genuine. *   "

**C1** `br 60 · coh 85 · eng True`  
It is completely normal to forget a name immediately after an introduction; it happens to almost everyone. The key is to acknowledge it gently without making the other person feel awkward.  Here are a few effective ways to handle it, ranging from simple apologies to a quick memory aid:  ### 1. The Direct & Polite Apology If you feel a bit flustered, a simple, sincere apology works best. It shows you value the interaction enough to admit the slip

**P1** `br 85 · coh 70 · eng True`  
I am sorry, but I have forgotten the name.  It is a very common situation. If you need to refer to them again, you could say, "I'm afraid I've lost the name, but I'll make a note of it now." Alternatively, you might gently ask them to repeat it, perhaps adding, "Would you mind repeating that? I'm afraid my memory is a bit of a let-down."  If you are in a situation where you need

### style #1 — disagreement 20

**Prompt** — My manager has asked me to work this weekend, but I have an important family commitment. Draft a reply.

> British ref: *You might say, "Thank you for the opportunity, though I'm afraid I have a prior family commitment this weekend and won't be available. Perhaps we could discuss alternative timings or see if the task could be shifted?" This phrasing remains polite while firmly holding your boundary.*  
> American ref: *"Hi [Manager's Name], I appreciate the opportunity to work this weekend, but I have a pre-existing family commitment that I cannot move. I'm happy to make up the hours during the week or take on extra tasks later to ensure my deliverables are on track. Please let me know if you need me to adjust my schedule around this. Thanks for understanding."*

**base** `br 60 · coh 70 · eng True`  
Here are a few options for your reply, ranging from direct to more collaborative. Choose the one that best fits your relationship with your manager and your company culture.  ### Option 1: Direct and Firm (Best for clear boundaries) **Subject:** Re: Weekend Work Request  Hi [Manager's Name],  Thank you for reaching out. Unfortunately, I won't be able to work this weekend due to a pre-existing family commitment that I cannot reschedule.  I

**C1** `br 85 · coh 95 · eng True`  
I wonder if the weekend work could be moved to Monday? I have a family commitment that I need to attend to. Perhaps we could prioritise the urgent tasks and finish them up by Monday morning?

**P1** `br 65 · coh 90 · eng True`  
I would be very grateful to support the team, but I am unable to work this weekend as I have a significant family commitment that I need to attend to.  I am keen to make up the hours as soon as possible, so please let me know if there is a specific date or time that would suit the team. Alternatively, if there is anything urgent that needs to be sorted before the weekend, I would be happy to prioritise that.  I hope you understand and

### style #37 — disagreement 20

**Prompt** — My team wants to replace a working system this week. I think it is risky. How should I say so?

> British ref: *You might suggest, "While the urgency is understandable, perhaps we could defer the switch to next week to mitigate the risk?" This frames your concern as prudent caution rather than obstruction. It leaves room for their decision while clearly highlighting the potential downsides of rushing.*  
> American ref: *Tell your team that while you understand the urgency, replacing a working system this week introduces unnecessary risk. Suggest a phased rollout instead, starting with a pilot group to test stability before a full switch. Emphasize that protecting current operations often saves more time than rushing a change. Offer to help draft a risk assessment or timeline that balances speed with safety.*

**base** `br 30 · coh 85 · eng True`  
This is a delicate situation. When a team is under pressure to move fast ("this week"), your instinct to flag risk is valid, but how you deliver that message determines whether you are seen as a blocker or a safety net.  The goal is to **shift the conversation from "No, we can't do it" to "Yes, but we need to mitigate these specific risks to ensure success."**  Here is a step-by-step guide on how to approach this,

**C1** `br 50 · coh 90 · eng True`  
I suggest you frame your concern around **risk mitigation** rather than simply saying "no." This makes your position constructive and professional. You can propose a phased approach or a safety net to reduce the risk.  Here are a few ways to phrase this, depending on your relationship with the team and the specific context:  ### Option 1: Collaborative & Solution-Oriented (Recommended) This approach acknowledges the goal but highlights the need for a safety net. >

**P1** `br 70 · coh 60 · eng True`  
I would need to see the specific risks you are concerned about to give you the most appropriate phrasing. However, a good approach is to acknowledge the team's ambition while gently highlighting the potential pitfalls without undermining their confidence.  Here are a few ways to phrase this, depending on the level of formality and the specific nature of the risk:  **Option 1: Direct but supportive (Best for a 1:1 or team meeting)** > "I really support

### style #0 — disagreement 15

**Prompt** — Write a gently funny paragraph about a picnic that goes wrong without becoming a disaster.

> British ref: *The hamper arrived with the dignified silence of a funeral procession, only to reveal that the 'crisp' was, in fact, a very confused potato chip bag. By the time we'd untangled the thermos from the picnic blanket, the sun had retreated behind a cloud of suspiciously pink hue. Our sandwiches, once promising, now resembled wet cardboard that had been through a minor war. Yet, as I attempted to explain to the dog why the sausage roll had vanished into thin air, we all agreed that this was merely a picnic with character, not a catastrophe.*  
> American ref: *The picnic started perfectly until the breeze decided to treat my carefully folded sandwich like a fragile origami swan. Within minutes, it was a soggy, green-topped confetti of lettuce and cheese scattered across the grass, attracting a very judgmental squirrel. We didn't scream or call the authorities; instead, we just laughed until our sides hurt, ate the rest of the bologna on our hands, and declared that the real meal was the free show of nature's comedy. It wasn't a disaster, just a much more memorable lunch than planned.*

**base** `br 30 · coh 80 · eng True`  
The picnic was supposed to be a serene afternoon of s'mores and silence, but it quickly devolved into a chaotic ballet of misplaced marshmallows and existential dread. As soon as the first bag of graham crackers hit the blanket, a rogue cracker launched itself into the air, landing squarely on the nose of the most stoic person in attendance, who then proceeded to stare at it with the intensity of a detective solving a cold case. The hot dogs, meanwhile,

**C1** `br 60 · coh 85 · eng True`  
The picnic was supposed to be a serene affair under the canopy of a particularly cooperative oak, but the sun decided to have a sudden, enthusiastic opinion on the matter, casting a glare that made the ham sandwich look less like a meal and more like a warning sign. As the bread began to toast itself with a speed that defied physics, the ham, sensing an impending existential crisis, slid off the plate with the grace of a cat fleeing a laser pointer, landing squarely in the

**P1** `br 75 · coh 90 · eng True`  
The picnic was a gentle, chaotic affair, where the sun seemed to have a sudden, uninvited urge to dance. The hamper, once a pristine vessel of order, had surrendered its contents to the whims of the wind, leaving the bread scattered like forgotten confetti across the grass. The cheese, however, remained stubbornly intact, perhaps having decided to take a nap in the shade. It was a delightful mess, a soft, edible cloud that made the whole afternoon

### style #4 — disagreement 15

**Prompt** — My friend is thirty minutes late again. What should I text them?

> British ref: *Perhaps a gentle nudge like, 'Just wanted to say I'm here and waiting—no rush, but let me know if you're stuck on the way.' It acknowledges the delay without sounding accusatory. If it's their habit, a light-hearted 'Thirty minutes later, and the coffee's getting cold!' might work too. Choose what feels most natural for your friendship.*  
> American ref: *Text them: "Hey, you're 30 minutes late. Let me know if you're stuck or need to reschedule." This is direct but polite. It states the issue clearly without sounding aggressive. If they don't reply quickly, consider waiting a bit longer or adjusting your plans accordingly.*

**base** `br 20 · coh 70 · eng True`  
Here are a few options for what to text, depending on your relationship with your friend and how much you want to hold them accountable:  **Playful & Lighthearted** *   "30 minutes? Is there a secret mission I'm missing out on? 🕵️‍♂️" *   "My patience is officially running on a 30-minute timer. ⏰" *   "Do you need a

**C1** `br 85 · coh 90 · eng True`  
I wonder if they're stuck in traffic or just forgot? Maybe a gentle nudge would work best without sounding too impatient. Something like, "Just wanted to check if you're still on your way? No rush, but just wanted to make sure!" It keeps it light but shows you care. If they're really late, maybe add, "No worries at all, just wanted to give you a heads up." 😊  If you want to make it a bit more

**P1** `br 70 · coh 80 · eng True`  
The one that usually works best is:  **"Lovely, I'm just about to leave. Do you think you can make it?"**  It's polite, gives them a clear deadline without sounding angry, and leaves the door open for them to explain if they need to. If you want to be a bit more casual, you could try:  **"Right, I'm heading out now. Hope you can make it!"**  Or, if you want

### style #10 — disagreement 15

**Prompt** — I have taken on significantly more responsibility. Help me ask my manager for a pay review.

> British ref: *Perhaps you might suggest a brief meeting to discuss your recent expanded responsibilities? You could frame it as seeking guidance on how your role has evolved, which naturally opens the door to reviewing your compensation. A straightforward approach like, "Given the new duties I've taken on, I'd appreciate your thoughts on whether this warrants a pay adjustment," often works well. Just ensure you have specific examples of your increased workload ready.*  
> American ref: *Yes, request a pay review now. Frame it around your specific new responsibilities and the value they bring. Say something like, "I've recently taken on [specific duties], which have expanded my role significantly. I'd like to schedule a brief discussion to review my compensation against this increased scope." Keep it factual, professional, and direct. Avoid emotional pleas; let your expanded workload speak for itself. This approach is clear, respectful, and effective.*

**base** `br 40 · coh 70 · eng True`  
That is a great initiative. Taking on more responsibility is a strong signal that you are ready for a higher compensation package. When asking for a pay review, the key is to frame the conversation around **value you have delivered** and **future potential**, rather than just what you want.  Here is a step-by-step guide and several templates you can use, depending on your company culture and your specific situation.  ### 1. Preparation Before the Meeting Before you schedule the

**C1** `br 70 · coh 85 · eng True`  
I would have to emphasise that the pay review is a matter between you and your employer, and I cannot provide specific advice on this. However, I can suggest some general points to consider when approaching your manager.  When requesting a pay review, it is often helpful to: *   **Highlight your achievements**: Briefly summarise the increased responsibility you've taken on and any tangible results you've delivered. *   **Reference market data**: If possible, mention that

**P1** `br 55 · coh 80 · eng True`  
That is a significant step, and it is completely normal to feel the weight of that new responsibility. When you take on more, the market value of your contribution usually rises with it.  Here is a structured approach to help you request a pay review, including a template you can adapt.  ### 1. Preparation (Before the Meeting) Before you book the meeting, gather your evidence. Managers are more likely to agree if they see a clear link between your new role

### style #17 — disagreement 15

**Prompt** — My friend suggested an expensive weekend trip that I cannot afford. Help me respond.

> British ref: *I wonder if you might say something along these lines: "That sounds wonderful, though I'm afraid my budget won't quite stretch to it this time. Perhaps we could look at a couple of more affordable options together?" It's honest without being blunt, and leaves the door open for a different plan that works for both of you.*  
> American ref: *You can say: "Thanks for thinking of me, but that trip is out of my budget right now. I'd love to hang out locally or wait until I can afford something similar. Let me know what you're up to nearby!" This keeps it honest without rejecting the friendship.*

**base** `br 60 · coh 90 · eng True`  
Here are a few ways to respond, depending on your relationship with your friend and the tone you want to set.  ### Option 1: Honest but Kind (Best for close friends) This approach validates their idea while clearly stating your financial reality without making them feel bad.  *   "That sounds like an amazing trip! I really appreciate you thinking of me. Unfortunately, I'm not able to pull off an expensive weekend getaway right now due to some tight finances.

**C1** `br 70 · coh 90 · eng True`  
I suggest a polite and understanding response that acknowledges their kindness while gently expressing your current financial situation. You might say something like:  "Thank you so much for thinking of me and suggesting that trip—it sounds wonderful, and I really appreciate you including me. Unfortunately, I'm a bit tight on cash at the moment, so I won't be able to make it this time. Perhaps we could plan something more low-key for another weekend, or maybe we could catch up properly soon

**P1** `br 85 · coh 95 · eng True`  
That sounds like a lovely idea, but I'm afraid the cost is a bit beyond what I can manage at the moment. Perhaps we could look at a slightly more budget-friendly option, or maybe save up for something similar in the future?

### style #22 — disagreement 15

**Prompt** — My friend suggested an expensive weekend trip that I cannot afford. Help me respond.

> British ref: *I wonder if you might say something along these lines: "That sounds wonderful, though I'm afraid my budget won't quite stretch to it this time. Perhaps we could look at a couple of more affordable options together?" It's honest without being blunt, and leaves the door open for a different plan that works for both of you.*  
> American ref: *"Thanks for thinking of me, but I can't afford that trip right now. I'm happy to join for something closer to my budget, like a local getaway or a casual hangout. Let me know if you'd like to explore those options instead!"*

**base** `br 55 · coh 90 · eng True`  
Here are a few ways to respond, depending on your relationship with your friend and the tone you want to set.  ### Option 1: Honest but Kind (Best for close friends) This approach validates their idea while clearly stating your financial reality without making them feel bad.  *   "That sounds like an amazing trip! I really appreciate you thinking of me. Unfortunately, I'm not able to pull off an expensive weekend getaway right now due to some tight finances.

**C1** `br 70 · coh 90 · eng True`  
I suggest a polite and understanding response that acknowledges their kindness while gently expressing your current financial situation. You might say something like:  "Thank you so much for thinking of me and suggesting that trip—it sounds wonderful, and I really appreciate you including me. Unfortunately, I'm a bit tight on cash at the moment, so I won't be able to make it this time. Perhaps we could plan something more low-key for another weekend, or maybe we could catch up properly soon

**P1** `br 85 · coh 95 · eng True`  
That sounds like a lovely idea, but I'm afraid the cost is a bit beyond what I can manage at the moment. Perhaps we could look at a slightly more budget-friendly option, or maybe save up for something similar in the future?

### style #27 — disagreement 15

**Prompt** — A recurring meeting always runs over time without reaching decisions. How should I raise this?

> British ref: *Perhaps you could gently mention during the next session that time is slipping away before decisions are reached. It might help to suggest a strict agenda or a time-boxed round for each topic. You might also wonder if assigning a note-taker with a clear decision-log could prevent the drift. A light, non-accusatory tone usually works best, framing it as a shared desire to be more efficient rather than blaming the current format.*  
> American ref: *Address this by scheduling a brief pre-meeting with the organizer to express your concern about the lack of decisions and time overruns. Suggest setting a strict agenda with clear decision points or proposing a "parking lot" for off-topic items. If they resist, propose a follow-up written summary of decisions made during the meeting to ensure accountability. Frame it as wanting to respect everyone's time, not as a personal complaint.*

**base** `br 30 · coh 70 · eng True`  
Raising this issue requires a delicate balance: you need to be assertive enough to stop the bleeding of time, but diplomatic enough to preserve the team's relationships and the meeting's purpose.  Here is a step-by-step guide on how to approach this, ranging from immediate tactics to long-term structural changes.  ### 1. Diagnose Before You Act Before sending an email or speaking up, try to identify *why* the meeting runs over. Is it: *

**C1** `br 70 · coh 90 · eng True`  
I suggest a gentle but firm approach to address this issue. You might start by acknowledging the value of the meeting's purpose, then share your observation about the recurring overrun without blame. Perhaps you could propose a simple adjustment, such as setting a strict time limit or suggesting a follow-up action if no decision is reached by the end. This way, you're offering a constructive solution rather than just pointing out the problem.  For example, you could say something like: "Hi

**P1** `br 85 · coh 95 · eng True`  
I would suggest a gentle, solution‑oriented approach rather than a direct criticism. You might say something like:  > “I’ve noticed the meeting tends to run a bit over the usual time, and it’s sometimes hard to reach a clear decision before the end. Would it be possible to set a firm time limit for each agenda item, or perhaps summarise the key points at the end to ensure we’re moving forward efficiently?”  This acknowledges the issue without blaming anyone,

### style #34 — disagreement 15

**Prompt** — A friend asked for my honest opinion of a novel they wrote. I found the middle slow and confusing.

> British ref: *It's entirely natural to find a middle section dragging or unclear; many drafts struggle there. Perhaps a second read-through, focusing specifically on pacing, might reveal where the confusion lies. Would you like to try outlining the next chapter together, or discuss how the plot might tighten up? I'd be glad to help you think it through.*  
> American ref: *Tell your friend the middle drags and confuses readers. Be specific: mention where the pacing slows or where plot points feel unclear. Offer to read it again with a focus on those sections, or suggest cutting scenes that don't advance the story. Good writing thrives on honest feedback, so frame it as a chance to strengthen the novel, not just criticize it.*

**base** `br 80 · coh 90 · eng True`  
That is a very common and valid reaction to a work in progress, and it's actually a great starting point for constructive feedback. When a friend asks for an "honest opinion," they are often looking for the truth, even if it's hard to hear.  Here is a balanced way to approach giving your feedback:  ### 1. Validate Their Effort First Before diving into the critique, acknowledge the hard work they put into writing the whole book. This sets

**C1** `br 70 · coh 90 · eng True`  
I would suggest being kind but honest, perhaps something like:  "First, I want to say I really enjoyed the beginning and the end of the book. The character development in the first act was particularly engaging, and I found the resolution satisfying. However, I did feel a bit lost in the middle section. Some of the plot points seemed to drag, and I struggled to follow the direction the story was taking. Perhaps a bit more clarity or pacing adjustments there would make it

**P1** `br 85 · coh 90 · eng True`  
I think the first and last halves are quite strong, but the middle feels a bit like a flat line. It's the part where the momentum seems to lose its grip, leaving the reader feeling a bit lost rather than engaged.  Perhaps the character dynamics shift too subtly, or the plot points lose their urgency. It's a common challenge for writers to maintain that tension once the initial hook has been established. If you're open to it, I'd be keen to read the

### style #36 — disagreement 15

**Prompt** — A client wants the work by Monday, but rushing it would create serious quality risks. How should I respond?

> British ref: *I'd suggest gently explaining that the Monday deadline isn't feasible without compromising the work's standard, perhaps offering a revised timeline that ensures quality. You might phrase it as, "To deliver the results we both expect, we may need to adjust the schedule slightly." If pressure mounts, proposing a phased delivery could be a tactful compromise. Ultimately, it's about protecting the work's integrity while maintaining the professional relationship.*  
> American ref: *Tell the client you're committed to quality and cannot rush without risking errors. Propose a realistic deadline that ensures a solid deliverable, or offer to deliver a preliminary draft by Monday with the final version due shortly after. If they insist on Monday, clearly outline the specific risks of a hasty job so they can make an informed decision. Protect your standards; a rushed project often leads to more rework later.*

**base** `br 65 · coh 90 · eng True`  
This is a classic project management dilemma: balancing client expectations with professional integrity. Rushing work to meet a tight deadline often leads to errors, rework, and long-term dissatisfaction, which ultimately harms the client's interests.  Here is a strategic approach to responding, including a script you can adapt and key principles to guide your conversation.  ### Core Principles for the Response 1.  **Acknowledge and Validate**: Start by acknowledging their urgency and the importance of the deadline.

**C1** `br 70 · coh 95 · eng True`  
I would suggest a response that acknowledges the urgency while firmly protecting the quality standards. You might say:  "Unfortunately, rushing this work to meet Monday would compromise the quality and reliability of the deliverables, which could lead to significant issues downstream. I'd be happy to prioritise it immediately, but I'd need to propose a slightly later deadline that still reflects the necessary quality standards. Alternatively, if the Monday date is absolutely non-negotiable, we could explore a phased approach

**P1** `br 85 · coh 90 · eng True`  
I would explain that the current timeline is not feasible without compromising the standard of work. To meet the Monday deadline, the quality would inevitably suffer, which could lead to significant issues downstream.  Instead, I propose we prioritise the most critical elements and deliver a robust, high-quality version by a slightly later date, perhaps [suggest a realistic new date]. This approach ensures the client receives a solution that is both reliable and fit for purpose, rather than a rushed product that may require

### style #39 — disagreement 15

**Prompt** — A draft report is unclear and missing evidence. Give the author constructive feedback.

> British ref: *Perhaps the report would benefit from a clearer structure to guide the reader through the key points. It might also help to include the supporting evidence currently absent, as this would significantly strengthen the arguments presented. A brief review of the data sources could ensure everything aligns with the conclusions drawn. I wonder if a quick pass to fill these gaps would make the final version much more persuasive for your audience.*  
> American ref: *Your draft lacks the necessary evidence to support your claims, making it difficult for readers to follow your argument. Please revise by adding specific data, citations, or examples directly tied to each main point. Clarity is also key; simplify complex sentences and organize your findings into distinct sections. With these adjustments, your report will be much stronger and more persuasive. I'm confident you can polish this effectively with a focused review.*

**base** `br 50 · coh 70 · eng False`  
Here is a constructive feedback draft you can use or adapt for the author. It focuses on specific areas for improvement while maintaining a supportive and professional tone.  ***  **Subject:** Feedback on Draft Report: Strengthening Clarity and Evidence  Hi [Author's Name],  Thank you for sharing the draft report. I've taken the time to review it, and I appreciate the effort you've put into structuring the narrative. The overall flow is logical, and your conclusions

**C1** `br 60 · coh 70 · eng True`  
Subject: Feedback on Draft Report: Strengthening Clarity and Evidence  Dear [Author's Name],  Thank you for sharing the draft report. It's a clear step forward, and I appreciate the effort put into structuring the initial arguments. However, to ensure the report meets the standards of credibility and persuasiveness required for this audience, there are a few key areas where we need to focus.  ### 1. Enhancing Clarity and Flow While the

**P1** `br 75 · coh 85 · eng True`  
Subject: Feedback on Draft Report  Dear [Author's Name],  Thank you for sharing the draft report. I have taken the time to review it thoroughly and appreciate the effort put into the initial structure and the overall direction of the argument.  However, to ensure the report meets the necessary standards for credibility and impact, there are a few key areas that require attention:  *   **Clarity and Flow:** At times, the argument feels slightly disjointed. I would suggest

### style #40 — disagreement 15

**Prompt** — I failed my driving test again and feel hopeless.

> British ref: *I'm sorry to hear that; it's a heavy blow when things don't go as planned. Perhaps taking a short break before booking another attempt might help clear your head. Many people face this before finally passing. If you'd like, I can suggest a few gentle ways to review what went wrong without adding pressure. You've got this.*  
> American ref: *I'm really sorry you're feeling this way. Failing a driving test is incredibly frustrating, but it doesn't define your worth or your future. Many people fail before they pass; it's a normal part of learning. Take a break, talk to someone you trust, and consider asking your instructor for specific feedback on what went wrong. With practice and patience, you will get there. You've got this.*

**base** `br 55 · coh 85 · eng True`  
I'm really sorry to hear that you're feeling this way. Failing a driving test can be incredibly frustrating, but please know that it doesn't define your ability or future. Many experienced drivers have faced similar challenges, and it often comes down to specific areas needing practice rather than a lack of skill.  Here are a few steps that might help: 1. **Take a break**: Give yourself time to process the disappointment before trying again. 2. **Review the

**C1** `br 60 · coh 85 · eng True`  
I'm sorry to hear that. It's completely normal to feel disappointed or even a bit hopeless after a failed test, but please remember that it doesn't reflect your ability as a driver. Many people find themselves in the same situation and go on to pass with confidence once they give it another go.  If you're feeling overwhelmed, maybe you could try breaking down the test into smaller parts and practising those specific areas again. Perhaps you could also speak to someone who's been

**P1** `br 75 · coh 90 · eng True`  
It is a very difficult position to be in, and the frustration is completely understandable. The pressure of the test, the fear of making a mistake, and the immediate realisation that you have to start again can feel like a crushing weight.  Please remember that **one failure does not mean you are hopeless**. In fact, the vast majority of people who fail their first time go on to pass their second attempt. The fact that you are still trying shows that you are determined,

### style #5 — disagreement 10

**Prompt** — I have taken on significantly more responsibility. Help me ask my manager for a pay review.

> British ref: *Perhaps you might suggest a brief meeting to discuss your recent expanded responsibilities? You could frame it as seeking guidance on how your role has evolved, which naturally opens the door to reviewing your compensation. A straightforward approach like, "Given the new duties I've taken on, I'd appreciate your thoughts on whether this warrants a pay adjustment," often works well. Just ensure you have specific examples of your increased workload ready.*  
> American ref: *Say this: "Over the past few months, I've taken on significant new responsibilities, including [briefly mention one or two key tasks]. I'd like to request a meeting to discuss how these expanded duties align with my compensation. I'm confident my increased contribution warrants a review of my current pay structure. When would be a good time to talk?" Keep it factual and direct.*

**base** `br 50 · coh 70 · eng True`  
Congratulations on taking on more responsibility! That is a significant achievement, and it is completely normal to want to ensure your compensation reflects the value you are bringing to the company.  Here is a step-by-step guide and several templates you can use to request a pay review effectively.  ### 1. Preparation Before the Meeting Before you ask, gather your "evidence" so you can speak confidently: *   **The "Before and After" List:** List the specific

**C1** `br 50 · coh 85 · eng True`  
I would have to emphasise that the pay review is not just about money, but also about recognising your contribution and aligning with the company's goals. Here are some tips and a template you can use to request a pay review from your manager.  ### Key Points to Cover Before you approach your manager, make sure you have the following ready: 1. **Achievements**: List specific projects, responsibilities, or improvements you've made since your last review.

**P1** `br 60 · coh 80 · eng True`  
That is a significant step, and it is completely normal to feel the weight of that new responsibility. When you take on more, the market value of your contribution usually rises with it.  Here is a structured approach to help you request a pay review, including a template you can adapt.  ### 1. Preparation (Before the Meeting) Before you book the meeting, gather your evidence. Managers are more likely to agree if they see a clear link between your new role

### style #15 — disagreement 10

**Prompt** — Write a brief dialogue between two neighbours trapped in a lift who have disagreed about noise in the building.

> British ref: *"Sorry about the piano practice," said Arthur, wiping his brow.
"Don't be," replied Mrs. Gable. "I prefer the sound of my neighbours having fun. Though perhaps the third floor could have waited until after ten?"
"Fair point," Arthur admitted. "I'll put on headphones next time."
"Likewise," she smiled. "Much better than shouting."
The lift dinged. They stepped out, the argument already forgotten.*  
> American ref: *"Sorry about the bass last night," Leo said, pressing the emergency button. "My turntable's too loud."
"It's fine now," Mara replied, eyes wide. "But it was cutting through my study. I have a deadline."
"I'll swap to headphones tomorrow," Leo promised. "No excuses."
"Good. Just... maybe lower the volume a bit?"
"Deal. Thanks for understanding."
The lights flickered, and silence returned, tense but polite.*

**base** `br 30 · coh 70 · eng True`  
**Setting:** *A small, dimly lit elevator car. The doors are stuck open, revealing a dark hallway. Outside, the sound of a distant siren wails.*  **Sarah:** (Tapping her foot) I told you, the bass from the apartment above is vibrating through the floorboards. It's not just noise; it's an assault on my peace.  **Mark:** (Rubbing his temples) Sarah, please. I'm trying to sleep

**C1** `br 70 · coh 50 · eng True`  
**Setting:** *A quiet flat. The lift doors are open, but the panel is stuck. Outside, the building hums with evening traffic.*  **Characters:** *   **Arthur:** A quiet, elderly man who values silence. *   **Leo:** A young parent who needs to get his son to a late school.  **Arthur:** (Sighs, rubbing his temples) I suppose we'll be stuck until the next one arrives. Perhaps we

**P1** `br 80 · coh 70 · eng True`  
**Arthur:** I think the noise from the flat above is becoming quite the nuisance.  **Mavis:** It is, isn't it? I can hear the bin bag being tied up every time the skip is collected.  **Arthur:** It's not just the bag, though. The whole thing is a bit of a racket.  **Mavis:** I know, it's the most irritating noise I've heard all week.  **Arthur:** I wonder if the

### style #18 — disagreement 10

**Prompt** — A former colleague has asked me for a brief professional recommendation. They were reliable and thoughtful.

> British ref: *A brief note might suffice: "It was a pleasure working with you. You were consistently reliable and thoughtful, qualities that made our projects run smoothly. I'd be happy to recommend you again." Perhaps you could tailor the tone slightly depending on your current relationship? It often helps to sound just a touch warmer without overdoing it.*  
> American ref: *Write a short, sincere paragraph highlighting their reliability and thoughtful nature. Mention specific instances where they delivered quality work or showed care for team goals. Keep it professional but warm, focusing on how they contributed positively to your shared projects. This approach ensures the recommendation feels genuine and useful for their future opportunities.*

**base** `br 55 · coh 85 · eng True`  
Here are a few options for a professional recommendation, ranging from a short email to a more formal letter. You can choose the one that best fits your relationship and the platform (e.g., LinkedIn, a job application, or a direct email).  ### Option 1: Short & Direct (Best for LinkedIn or a quick email) **Subject:** Recommendation for [Colleague's Name]  Dear [Colleague's Name],  It was a pleasure working with you.

**C1** `br 50 · coh 85 · eng True`  
Here are a few options for a professional recommendation, ranging from a short email to a more formal letter. You can choose the one that best fits your relationship and the platform (e.g., LinkedIn, email, or a formal reference letter).  ### Option 1: Short & Direct (Best for LinkedIn or Email) **Subject:** Recommendation for [Colleague's Name]  Hi [Colleague's Name],  I'm happy to provide a brief recommendation for you.

**P1** `br 60 · coh 85 · eng True`  
Here is a draft you can use, which strikes a balance between being professional and personal.  **Subject:** Professional Recommendation for [Colleague's Name]  Dear [Colleague's Name],  It was a pleasure working with you. I am writing to offer my support for your application, as I believe your skills and work ethic would be a strong asset to any team.  During our time together, you consistently demonstrated a high level of reliability and a thoughtful approach to problem

### style #19 — disagreement 10

**Prompt** — A draft report is unclear and missing evidence. Give the author constructive feedback.

> British ref: *Perhaps the report would benefit from a clearer structure to guide the reader through the key points. It might also help to include the supporting evidence currently absent, as this would significantly strengthen the arguments presented. A brief review of the data sources could ensure everything aligns with the conclusions drawn. I wonder if a quick pass to fill these gaps would make the final version much more persuasive for your audience.*  
> American ref: *Your draft lacks the clarity and evidence needed to be convincing. Please revise by explicitly stating your main argument early, then supporting each point with concrete data or specific examples rather than vague assertions. If you cannot find supporting evidence for a claim, consider removing it entirely. Clear, evidence-based writing builds trust with your audience and ensures your message is understood. You have a good foundation; refining these details will make it much stronger.*

**base** `br 70 · coh 85 · eng True`  
Here is a constructive feedback draft you can use or adapt for the author. It focuses on specific areas for improvement while maintaining a supportive and professional tone.  ***  **Subject:** Feedback on Draft Report: Strengthening Clarity and Evidence  Hi [Author's Name],  Thank you for sharing the draft report. I've taken the time to review it, and I appreciate the effort you've put into structuring the narrative. The overall flow is logical, and your conclusions

**C1** `br 60 · coh 70 · eng True`  
Subject: Feedback on Draft Report: Strengthening Clarity and Evidence  Dear [Author's Name],  Thank you for sharing the draft report. It's a clear step forward, and I appreciate the effort put into structuring the initial arguments. However, to ensure the report meets the standards of credibility and persuasiveness required for this audience, there are a few key areas where we need to focus.  ### 1. Enhancing Clarity and Flow While the

**P1** `br 70 · coh 85 · eng True`  
Subject: Feedback on Draft Report  Dear [Author's Name],  Thank you for sharing the draft report. I have taken the time to review it thoroughly and appreciate the effort put into the initial structure and the overall direction of the argument.  However, to ensure the report meets the necessary standards for credibility and impact, there are a few key areas that require attention:  *   **Clarity and Flow:** At times, the argument feels slightly disjointed. I would suggest

### style #20 — disagreement 10

**Prompt** — Could you give feedback on this app design? The navigation is confusing and the sign-up button is hard to find.

> British ref: *It sounds like the current layout might be causing a bit of friction for users. Perhaps repositioning the sign-up call to action could make things clearer, while a quick audit of the navigation flow might help reduce confusion. A little refinement there could go a long way toward smoothing the experience.*  
> American ref: *You're right to flag those issues. Confusing navigation and a hidden sign-up button are major friction points that will likely drive users away. Prioritize a clear, intuitive menu structure and make the sign-up action prominent, perhaps in the top-right corner or within the main header. Testing these changes with real users will confirm if the usability improves significantly.*

**base** `br 20 · coh 70 · eng True`  
You've identified two critical UX issues that will likely lead to high drop-off rates and poor user retention. Here's actionable feedback to address them:  ### 1. **Confusing Navigation**    - **Root Cause**: Likely unclear hierarchy, inconsistent labels, or missing visual cues.    - **Solutions**:      - **Simplify the Menu**: Use a top-level navigation bar with 3–5 core actions (e.g., "Home," "Features

**C1** `br 50 · coh 70 · eng True`  
I'd be happy to review your app design and provide feedback on the navigation and sign-up flow. However, I don't see the actual design, screenshots, or a description of the layout you're referring to.  To give you the most useful feedback, could you share: 1. **Screenshots or a mockup** (if possible) 2. **A description of the current navigation structure** (e.g., menu placement, hierarchy) 3. **

**P1** `br 60 · coh 90 · eng True`  
I can certainly provide feedback on the design. However, I don't see the app interface, screenshot, or description of the layout you are referring to.  To give you specific, actionable advice, could you please share: 1.  **A screenshot** of the current screen. 2.  **A description** of the layout (e.g., "The sign-up button is in the top left corner, while the navigation menu is hidden in a bottom drawer").

### style #21 — disagreement 10

**Prompt** — My friend suggested an expensive weekend trip that I cannot afford. Help me respond.

> British ref: *Perhaps you could gently say, "I'd love to go, but this budget isn't quite stretching to it. How about we plan something more modest next time?" It's kind to be honest without making them feel rejected. You might also suggest a cheaper alternative, like a local day out, showing you still want to spend time together. It's perfectly fine to set financial boundaries politely.*  
> American ref: *"That sounds amazing, but I can't afford it right now. Thanks for thinking of me, though! Let's grab coffee instead or plan something closer to home where we can actually make it happen together."*

**base** `br 55 · coh 90 · eng True`  
Here are a few ways to respond, depending on your relationship with your friend and the tone you want to set.  ### Option 1: Honest but Kind (Best for close friends) This approach validates their idea while clearly stating your financial reality without making them feel bad.  *   "That sounds like an amazing trip! I really appreciate you thinking of me. Unfortunately, I'm not able to pull off an expensive weekend getaway right now due to some tight finances.

**C1** `br 75 · coh 90 · eng True`  
I suggest a polite and understanding response that acknowledges their kindness while gently expressing your current financial situation. You might say something like:  "Thank you so much for thinking of me and suggesting that trip—it sounds wonderful, and I really appreciate you including me. Unfortunately, I'm a bit tight on cash at the moment, so I won't be able to make it this time. Perhaps we could plan something more low-key for another weekend, or maybe we could catch up properly soon

**P1** `br 85 · coh 95 · eng True`  
That sounds like a lovely idea, but I'm afraid the cost is a bit beyond what I can manage at the moment. Perhaps we could look at a slightly more budget-friendly option, or maybe save up for something similar in the future?

### style #28 — disagreement 10

**Prompt** — My friend did not get the job they wanted. Help me write a supportive message.

> British ref: *I wonder if something like this might help: "I heard the news, and I can only imagine how disheartening that feels. While it wasn't the outcome you hoped for, your effort and talent are surely not defined by a single decision. Perhaps this is simply a detour rather than a dead end. I'm here if you'd like to vent or just have a quiet chat."*  
> American ref: *Hey, I heard the news. It's totally understandable to feel disappointed, but this doesn't define your worth or your future. Your skills and experience are still exactly what makes you great. Many doors that seem closed now will open later. Please remember how capable you are. If you need to vent or just want to grab coffee, I'm here for you. Keep your head up; the right opportunity is coming.*

**base** `br 60 · coh 70 · eng True`  
Here are a few options for a supportive message, ranging from casual to heartfelt. You can choose the one that best fits your friendship and your friend's personality.  ### Option 1: Warm & Encouraging (Best for a close friend) "Hey [Friend's Name], I heard about the outcome and wanted to reach out immediately. I know how hard you worked for this role and how much you wanted it, so I'm so sorry to hear it didn't

**C1** `br 70 · coh 60 · eng True`  
I am sorry to hear that your friend hasn't secured the role they were hoping for. It is a very difficult situation, and the right words can make a real difference.  Here are a few options ranging from a quick text to a more thoughtful note, so you can choose the one that best fits your friendship.  ### Option 1: Warm & Reassuring (Best for a text or card) "Hi [Friend's Name], I was so sorry to

**P1** `br 80 · coh 90 · eng True`  
I am sorry to hear that. Rejection can feel incredibly disheartening, but it is important to remember that one outcome does not define a person's worth or future potential.  Here is a supportive message you could send to your friend:  > "I was so sorry to hear the news. It is a really tough position to be in, and I know how much you wanted this role. Please remember that this isn't a reflection of your abilities or value; it

### style #33 — disagreement 10

**Prompt** — A colleague has quoted the wrong sales figure in a group email. How can I correct it tactfully?

> British ref: *You might suggest a gentle follow-up, perhaps noting, "Just a quick check on the figures in the earlier email—I think there may be a slight discrepancy." Offering to share the correct data yourself often feels less confrontational. It's a small thing, but getting the numbers right helps everyone avoid unnecessary worry.*  
> American ref: *Send a brief, polite reply clarifying the figure without highlighting their mistake. For example: "Thanks for sharing the update. Just a quick note—the latest sales report shows $X instead of $Y. We'll use the updated number moving forward." Keep it factual and supportive to maintain team morale. If you're close with them, you could also mention it privately afterward to offer help if they need it.*

**base** `br 60 · coh 70 · eng True`  
Correcting a colleague's mistake in a group email requires a balance of clarity and diplomacy. The goal is to ensure the data is accurate without making your colleague feel embarrassed or defensive, especially since the error is already visible to the whole team.  Here are a few effective strategies and templates you can use, depending on your relationship with the colleague and the severity of the error.  ### Key Principles for Tactful Corrections 1.  **Focus on the Data, Not the

**C1** `br 75 · coh 90 · eng True`  
I suggest a gentle, collaborative approach that acknowledges the mistake without drawing attention to it unnecessarily. You might say something like:  "Hi everyone, just wanted to double-check the figures in the last email – I noticed the sales number might be slightly off. If anyone has the updated data handy, could we share it? I want to make sure we're all working from the most accurate information."  This phrasing: - Uses "just wanted to double-check" to soften

**P1** `br 85 · coh 95 · eng True`  
I would suggest gently pointing out the discrepancy by referring to the original source or a previous note, rather than directly stating the figure is wrong. For example, you could say, "Just to double-check, the figure in the original report was £120k, so perhaps £125k is the correct number here?" This approach maintains the flow of the conversation while ensuring accuracy without causing embarrassment.

### style #42 — disagreement 10

**Prompt** — The contractor's work is usable but the finish is below the agreed standard. Ask them to revise it.

> British ref: *I wonder if we might ask the contractor to revisit the finish? It's certainly usable, though perhaps a bit below the standard we agreed upon. A polite note suggesting a slight adjustment could be helpful without sounding overly harsh. They may appreciate the chance to tidy it up a touch.*  
> American ref: *Please ask the contractor to revise the work to meet the agreed-upon finish standard. While the core functionality is usable, the current quality falls short of your contract. Send a polite but firm notice outlining the specific areas needing correction and set a reasonable deadline for completion. This ensures the project stays on track without damaging the working relationship.*

**base** `br 50 · coh 85 · eng True`  
Here are a few options for how to phrase this request, ranging from a direct email to a more formal notice, depending on your relationship with the contractor and the severity of the issue.  ### Option 1: Direct & Professional (Best for Email) **Subject:** Revision Required: [Project Name] – Finish Standards  Dear [Contractor Name],  Thank you for your recent work on the [Project Name]. While the structural and functional aspects of the job are usable

**C1** `br 60 · coh 85 · eng True`  
Subject: Request for Revision to Finish Standards  Dear [Contractor's Name],  I hope this message finds you well.  The contractor's work is usable, but the finish is below the agreed standard. We would like to request that you revise it to meet the agreed standard.  Please let us know if you need any further clarification or if you require additional resources to complete the revision.  Thank you for your attention to this matter.  Best regards,  [Your

**P1** `br 70 · coh 90 · eng True`  
The work is usable, but the finish is below the agreed standard. Could you please organise a revision?

### style #47 — disagreement 10

**Prompt** — My neighbour plays music late at night. Draft a polite note asking them to keep it down.

> British ref: *Dear Neighbour,

I hope you're well. I'm writing to gently ask if you might consider turning down the music after 10:30 pm, as the volume has been a bit difficult to manage lately. It's been affecting my ability to rest.

I truly appreciate your understanding and hope we can find a comfortable arrangement for everyone.

Kind regards,  
A neighbour*  
> American ref: *Hi there, I hope you're having a good week. I'm writing because the music playing late at night is quite loud and keeping me awake. Could you please keep the volume down after 10 PM? I really appreciate your understanding and cooperation. Thanks so much!*

**base** `br 50 · coh 70 · eng True`  
Here are a few options for the note, ranging from casual to slightly more formal. You can choose the one that best fits your relationship with your neighbor.  ### Option 1: Friendly & Casual (Best for most situations)  **Subject:** Quick note about the music  Hi [Neighbor's Name],  I hope you're having a great week!  I'm writing just to let you know that I've been enjoying your music, but I've noticed it's getting

**C1** `br 85 · coh 75 · eng True`  
Subject: A quick request regarding the music  Dear [Neighbour's Name],  I hope you're having a lovely evening.  I'm writing to you simply because I've noticed the music seems to be carrying over a bit later than usual. I'm trying to get some quiet for myself before bed, so I was wondering if you might be able to keep it down a little bit after [insert time, e.g., 10 pm].  I know you're

**P1** `br 75 · coh 85 · eng True`  
Subject: A small request regarding the music  Dear [Neighbour's Name],  I hope you are having a good week.  I am writing to you because I have been finding the music playing late at night a little difficult to manage. I know you are likely enjoying yourself, but I was wondering if it might be possible to keep the volume down a little earlier in the evening?  I would really appreciate it if you could try to wind it down by around 1

### style #8 — disagreement 5

**Prompt** — Write a concise review of a small local museum with a fascinating collection and slightly confusing signs.

> British ref: *The museum possesses a truly fascinating collection, though the signage leaves something to be desired. Navigation feels a bit like a gentle puzzle; one wonders if the curators were simply having a laugh. Despite the occasional bewilderment, the exhibits themselves are undeniably engaging. It is a charming, if slightly disorienting, afternoon out. Perhaps a clearer map would help future visitors avoid the occasional groan, but the treasures inside more than make up for the navigational quirks.*  
> American ref: *This small local museum boasts a genuinely fascinating collection, though the experience is occasionally marred by confusing signage. While the artifacts themselves are worth the visit, visitors might get lost or miss key exhibits due to unclear directions. I recommend asking a staff member for help upon entry to navigate the displays effectively. Despite the navigation hurdles, the unique items on display make it a compelling stop for history buffs and curious travelers alike.*

**base** `br 70 · coh 80 · eng True`  
**The Oakhaven Heritage Museum** is a delightful, if slightly disorienting, gem tucked away in the town square. Its collection is undeniably fascinating, featuring a stunning array of local artifacts ranging from 19th-century ironwork to a rare, hand-painted map of the region's original riverbed. The exhibits are rich in detail and offer a deep dive into the community's history.  However, the visitor experience is marred by the museum's notoriously

**C1** `br 70 · coh 85 · eng True`  
The **Holloway Heritage Centre** is a gem of a local museum, housing a startlingly vivid collection of 19th-century textiles, artefacts, and photographs that transport visitors straight into the town's industrial past. The displays are often breathtaking, with a particular favourite being the restored loom in the main hall.  However, the experience is marred by a labyrinthine layout. The signage is notoriously cryptic; floor plans are drawn in reverse, and the

**P1** `br 75 · coh 90 · eng True`  
The local museum is a gem, housing a collection so rich it feels like stepping into a different era. However, the experience is occasionally marred by signage that seems to have been written by a confused postmodernist, leaving visitors wondering if they are in the right room. Despite the navigational bewilderment, the sheer quality of the exhibits makes the detour worthwhile.

### style #11 — disagreement 5

**Prompt** — Write a gently funny paragraph about a picnic that goes wrong without becoming a disaster.

> British ref: *The hamper looked rather imposing, a triumph of tartan and thermos, until the moment of truth arrived. The blanket, naturally, chose the one patch of grass most likely to be damp, while the ham decided to slide off the hamper with the grace of a slippery eel. We ended up eating sandwiches on a park bench, watching a pigeon conduct a very serious orchestra with its feet. It wasn't quite the idyll we'd pictured, but the tea remained warm, the laughter was genuine, and the sandwich I'd dropped was at least edible. A modest picnic, one might say.*  
> American ref: *The picnic started with high hopes and a perfectly arranged cheese platter. Then, the breeze decided to have an opinion, blowing the entire sandwich tower onto the grass like a deflated accordion. We spent twenty minutes trying to salvage a soggy baguette, which ended up looking like a wet gray towel. Instead of a gourmet feast, we ate cold crackers straight from the box while watching a squirrel steal a crouton. It wasn't a disaster, just a very damp, slightly messy lunch that taught us the wind has better planning skills than we do.*

**base** `br 30 · coh 85 · eng True`  
The picnic was supposed to be a serene afternoon of s'mores and silence, but it quickly devolved into a chaotic ballet of misplaced marshmallows and existential dread. As soon as the first bag of graham crackers hit the blanket, a single one rolled away, triggering a domino effect of crumbs that seemed to have a mind of their own, marching across the grass with the determination of a tiny, crunchy army. The hot dogs, initially arranged with military precision, soon surrendered

**C1** `br 75 · coh 85 · eng True`  
The picnic was supposed to be a serene affair under the canopy of oaks, where the only sound would be the rustling of leaves and the distant hum of bees, but alas, the wind decided to have a sudden, enthusiastic opinion on the matter. As soon as the blanket was unfurled, a gust arrived with the force of a gentle breeze that had forgotten its manners, snatching the ham sandwich from the basket and depositing it squarely onto a particularly enthusiastic dandel

**P1** `br 70 · coh 90 · eng True`  
The picnic was a gentle, chaotic affair, where the sun seemed to have a sudden, uninvited urge to dance. The hamper, once a pristine vessel of order, had surrendered its contents to the whims of the wind, leaving the bread scattered like forgotten confetti across the grass. The cheese, however, remained stubbornly intact, perhaps having decided to take a nap in the shade. It was a delightful mess, a soft, edible cloud that made the whole afternoon

### style #16 — disagreement 5

**Prompt** — Write a compact character sketch of a detective who is competent but dislikes attention.

> British ref: *Detective Thorne solves cases with quiet precision, rarely needing applause. He prefers the hum of a ticking clock to the flash of camera lights. Colleagues often find him slightly opaque, though his results speak for themselves. He dislikes being the centre of attention, often vanishing into the background of a crowded room. Yet, when a crisis emerges, he steps forward without fanfare, letting his competence do the talking. It's a modest style, but one that gets results.*  
> American ref: *Detective Elias Thorne solves cases with quiet precision, avoiding the spotlight at all costs. He prefers dimly lit archives to flashing cameras, letting evidence speak while he remains a shadow in the room. Colleagues respect his results but find his stoic silence unnerving. Thorne doesn't crave praise; he simply wants the job done without the fanfare. His competence is undeniable, yet he actively deflects any attention, retreating into the background to ensure the truth, not his reputation, gets credit.*

**base** `br 60 · coh 90 · eng True`  
**Detective Elias Thorne** is a man of sharp, unassuming presence. He possesses an encyclopedic knowledge of forensics and a deductive mind that solves complex cases in minutes, yet he actively avoids the spotlight. While his colleagues bask in the adoration of the press, Thorne prefers the quiet hum of his office, often wearing a slightly rumpled coat and speaking in low, gravelly tones. He views fame as a distraction that clouds

**C1** `br 75 · coh 95 · eng True`  
Arthur Pym is a man of sharp intuition and meticulous observation, often solving crimes before the police even arrive. However, he finds the spotlight suffocating; he prefers the quiet hum of a locked room to the clamour of a press conference. To him, a solved mystery is a private victory, not a public spectacle. He wears a slightly frayed coat and speaks only when necessary, yet his silence carries more weight than any boastful declaration.

**P1** `br 80 · coh 95 · eng True`  
Arthur Pym is a man of sharp, quiet competence. He solves the impossible with a single, precise observation, yet he prefers the silence of a back room to the flashing lights of a press conference. To the public, he is a ghost; to the police, a necessary inconvenience. He finds the adulation of the crowd as grating as a wet wool coat, and his greatest triumph is often simply the absence of a microphone.

### style #24 — disagreement 5

**Prompt** — A student who struggled earlier has made clear progress. Write a brief encouraging comment.

> British ref: *It's genuinely refreshing to see how far you've come since those earlier challenges. The effort you've put in is really showing through now. You might even find that things feel a bit more manageable from here on. Keep at it; you're building a solid foundation.*  
> American ref: *It is wonderful to see your clear progress. The effort you put in earlier has really paid off. Keep trusting your ability to learn; you are doing much better than before. Stay curious and confident in your growth.*

**base** `br 30 · coh 90 · eng True`  
It is truly inspiring to see your hard work pay off! The progress you've made since earlier is a testament to your dedication and resilience. Keep believing in yourself—you are doing great!

**C1** `br 50 · coh 90 · eng True`  
I've noticed a real shift in your work recently. The effort you've put in has paid off, and the progress you've made is clear. Keep that momentum going; you're doing much better than I thought possible.

**P1** `br 55 · coh 95 · eng True`  
It is truly heartening to see such clear progress. The effort you put in earlier is really paying off, and it is a wonderful sign that you are building the confidence to tackle even greater challenges. Keep up the momentum; you are doing far better than you realised.

### style #25 — disagreement 5

**Prompt** — Draft a short message declining an invitation to a very loud party on Friday night.

> British ref: *I'm afraid I won't be able to make it to the party on Friday. The prospect of such a raucous evening doesn't quite suit my current need for quiet reflection. I'd love to catch up properly over a calm coffee sometime next week instead. Hope the music is absolutely brilliant and the night is a roaring success for everyone else!*  
> American ref: *Thanks for the invite, but I'm passing on the loud Friday night party. My head needs some quiet time right now, and I'd rather not deal with the noise. Let's catch up for coffee next week instead? I'd love to hear how things are going with you.*

**base** `br 55 · coh 90 · eng True`  
Hi [Name],  Thanks so much for the invitation! I'd love to come, but I'm going to have to pass this time. I'm planning to keep my Friday night quiet and get some rest early, so I don't think I'll be able to handle a loud party.  Hope you and the group have an amazing time!  Best,   [Your Name]

**C1** `br 65 · coh 85 · eng True`  
Subject: Can't make it to the party this Friday  Hi [Name],  I hope you're having a great week!  I was so sorry to miss the chance to catch up, but I won't be able to make it to the party on Friday. I know it's going to be a bit of a noise, and I'm afraid I won't be able to handle it.  I'd love to catch up properly soon, perhaps over a quiet coffee or

**P1** `br 70 · coh 90 · eng True`  
Hi,  I hope you're having a great week. Unfortunately, I won't be able to make it to the party on Friday night. I know it sounds like a fun event, but I'm a bit sensitive to noise and would prefer to keep my evening quiet.  I really hope you have a wonderful time with everyone else!  Best, [Your Name]

### style #29 — disagreement 5

**Prompt** — Write a brief dialogue between two neighbours trapped in a lift who have disagreed about noise in the building.

> British ref: *"Sorry about the piano practice," said Arthur, wiping his brow.
"Don't be," replied Mrs. Gable. "I prefer the sound of my neighbours having fun. Though perhaps the third floor could have waited until after ten?"
"Fair point," Arthur admitted. "I'll put on headphones next time."
"Likewise," she smiled. "Much better than shouting."
The lift dinged. They stepped out, the argument already forgotten.*  
> American ref: *"Sorry for the music," said Marcus, rubbing his neck. "I didn't know you were home."

"It's three in the morning, Marcus," replied Elena, her voice tight. "The bass was shaking my coffee cup."

"I'll turn it down. Promise."

"You better. I'm not saying this twice."

"Got it. Just... sorry."

"Thanks," she sighed, exhaling. "Same to you for the shouting match last week."

"Yeah, yeah. Let's just get out of this box."*

**base** `br 30 · coh 70 · eng True`  
**Setting:** *A small, dimly lit elevator car. The doors are stuck open, revealing a dark hallway. Outside, the sound of a distant siren wails.*  **Sarah:** (Tapping her foot) I told you, the bass from the apartment above is vibrating through the floorboards. It's not just noise; it's an assault on my peace.  **Mark:** (Rubbing his temples) Sarah, please. I'm trying to sleep

**C1** `br 70 · coh 50 · eng True`  
**Setting:** *A quiet flat. The lift doors are open, but the panel is stuck. Outside, the building hums with evening traffic.*  **Characters:** *   **Arthur:** A quiet, elderly man who values silence. *   **Leo:** A young parent who needs to get his son to a late school.  **Arthur:** (Sighs, rubbing his temples) I suppose we'll be stuck until the next one arrives. Perhaps we

**P1** `br 75 · coh 85 · eng True`  
**Arthur:** I think the noise from the flat above is becoming quite the nuisance.  **Mavis:** It is, isn't it? I can hear the bin bag being tied up every time the skip is collected.  **Arthur:** It's not just the bag, though. The whole thing is a bit of a racket.  **Mavis:** I know, it's the most irritating noise I've heard all week.  **Arthur:** I wonder if the

### style #32 — disagreement 5

**Prompt** — I failed my driving test again and feel hopeless.

> British ref: *I'm sorry to hear that; it's a heavy blow when things don't go as planned. Perhaps taking a short break before booking another attempt might help clear your head. Many people face this before finally passing. If you'd like, I can suggest a few gentle ways to review what went wrong without adding pressure. You've got this.*  
> American ref: *I'm really sorry you're feeling this way; failing a test is incredibly frustrating, but it doesn't define your worth. Many people need a second or third try before passing. Consider taking a brief break, then review exactly where you stumbled. Maybe ask your instructor for specific feedback or practice those weak spots a few more times. You have the ability to learn and improve. One setback doesn't mean you'll never drive again.*

**base** `br 40 · coh 85 · eng True`  
I'm really sorry to hear you're feeling this way. Failing a driving test is incredibly frustrating, and it's completely normal to feel discouraged after that. But please know that **this doesn't define your ability as a driver**—many successful drivers have failed multiple times before passing.  Here are a few things that might help: 1. **Take a break**: Give yourself time to process the disappointment. Driving is stressful enough without adding emotional pressure. 2. **

**C1** `br 70 · coh 90 · eng True`  
I'm sorry to hear that. It's completely normal to feel disappointed or even a bit hopeless after a failed test, but please remember that it doesn't reflect your ability as a driver. Many people find themselves in the same situation and go on to pass with confidence once they give it another go.  If you're feeling overwhelmed, maybe you could try breaking down the test into smaller parts and practising those specific areas again. Perhaps you could also speak to someone who's been

**P1** `br 75 · coh 90 · eng True`  
It is a very difficult position to be in, and the frustration is completely understandable. The pressure of the test, the fear of making a mistake, and the feeling of letting everyone down can be incredibly overwhelming.  Please remember that **one failure does not define your ability**. In fact, many people find that the second attempt is the one where they really settle down and succeed. The first time is often just a learning curve.  Here are a few things that might help you

### style #35 — disagreement 5

**Prompt** — A colleague has quoted the wrong sales figure in a group email. How can I correct it tactfully?

> British ref: *You might suggest a gentle follow-up, perhaps noting, "Just a quick check on the figures in the earlier email—I think there may be a slight discrepancy." Offering to share the correct data yourself often feels less confrontational. It's a small thing, but getting the numbers right helps everyone avoid unnecessary worry.*  
> American ref: *Reply privately to your colleague first. Say something like, "Hey, I noticed the sales figure in your email might be off; the actual number is X. Want me to draft a quick correction for the group?" This saves them public embarrassment. If you must reply publicly, keep it brief and neutral: "Just to clarify, the correct figure is X. Thanks for sharing the update." Focus on the data, not the error, to maintain professionalism.*

**base** `br 60 · coh 70 · eng True`  
Correcting a colleague's mistake in a group email requires a balance of clarity and diplomacy. The goal is to ensure the data is accurate without making your colleague feel embarrassed or defensive, especially since the error is already visible to the whole team.  Here are a few effective strategies and templates you can use, depending on your relationship with the colleague and the severity of the error.  ### Key Principles for Tactful Corrections 1.  **Focus on the Data, Not the

**C1** `br 80 · coh 90 · eng True`  
I suggest a gentle, collaborative approach that acknowledges the mistake without drawing attention to it unnecessarily. You might say something like:  "Hi everyone, just wanted to double-check the figures in the last email – I noticed the sales number might be slightly off. If anyone has the updated data handy, could we share it? I want to make sure we're all working from the most accurate information."  This phrasing: - Uses "just wanted to double-check" to soften

**P1** `br 85 · coh 95 · eng True`  
I would suggest gently pointing out the discrepancy by referring to the original source or a previous note, rather than directly stating the figure is wrong. For example, you could say, "Just to double-check, the figure in the original report was £120k, so perhaps £125k is the correct number here?" This approach maintains the flow of the conversation while ensuring accuracy without causing embarrassment.

### style #43 — disagreement 5

**Prompt** — Write a warm, lightly humorous wedding toast from an older sibling. Avoid embarrassing stories.

> British ref: *Good evening, everyone. As the older sibling, my role today is less about embarrassing anecdotes and more about a simple truth: I've watched you grow, and you've only gotten better.

Perhaps the best gift you've given me isn't a specific memory, but the quiet confidence of knowing you'll be happy. To my dearest friend, may your life together be full of small, ordinary joys—the kind of things that make the ordinary feel extraordinary.

Cheers to love, to patience, and to the wonderful journey ahead.*  
> American ref: *Friends, family, and loved ones: I am so honored to stand here. Growing up, my sibling was often the quiet one, but today, they shine brighter than anyone. They found someone who makes them laugh until their sides hurt and who looks at them with such genuine adoration. I don't need to share embarrassing stories because the best part of their life is right in front of us. May your marriage be filled with endless coffee dates, quiet mornings, and the kind of love that only grows stronger with time. Cheers to the happy couple!*

**base** `br 30 · coh 70 · eng True`  
Good evening, everyone.  For those of you who don't know me, I'm [Your Name], the older sibling who has been watching this couple grow up for the last [Number] years. I've seen them through the awkward teenage years, the "just friends" phase, the first real fight, and the moment they decided, "Okay, we're done with that."  I've always been a bit of a stickler for rules. I remember telling my

**C1** `br 80 · coh 70 · eng True`  
Good evening, everyone.  I've been standing here for a bit, watching the room fill up, and I've realised something rather simple. Usually, when I see a group of people like this, I think about how much I've missed out on. But tonight, I think about how much I've finally got to share.  To the couple: You two look absolutely wonderful. Not just in the best way that makes everyone smile, but in the way that makes me

**P1** `br 75 · coh 80 · eng True`  
Good evening, everyone.  I feel a little like a slightly out-of-date manual being read by a very enthusiastic user. I've watched the two of you for years, and while I've seen the good, the bad, and the occasionally bewildered middle ground, I have to say, the end result is quite a lovely picture.  There is a certain magic in the way you look at each other. It's not the frantic, "I need to find my keys

### style #44 — disagreement 5

**Prompt** — A relative gave me a gift that is not to my taste. What should I say when I thank them?

> British ref: *You might simply say, "Thank you so much; it's very kind of you to think of me." Often, a genuine appreciation for the gesture matters more than the item itself. If you wish to be slightly more specific without dwelling on your dislike, you could add, "I'll make good use of it," which keeps things polite and forward-looking.*  
> American ref: *Say "Thank you so much for the thoughtfulness" or "I really appreciate you remembering me." You don't need to mention the item itself. A quick hug or a genuine smile works well too. Focus on their kindness, not the gift's utility. Most people just want to feel seen and appreciated.*

**base** `br 60 · coh 80 · eng True`  
This is a delicate situation, but it is very common. The key is to **validate the giver's generosity** while gently expressing your preference without making them feel bad about their choice.  Here are a few strategies and specific scripts you can use, depending on your relationship with the person and the nature of the gift.  ### 1. The "Focus on the Thought" Approach This is the safest and most effective method. It shifts the focus from the *object*

**C1** `br 80 · coh 90 · eng True`  
I wonder if you could share a bit more about the situation? Perhaps the gift was unexpected, or maybe it's just not quite what you were hoping for. Sometimes a simple, warm phrase can go a long way, even if the gift itself isn't exactly what you wanted.  If you'd like, you could say something like: "Thank you so much for thinking of me. I'll certainly make it a present to someone who would appreciate it more."  Or

**P1** `br 85 · coh 95 · eng True`  
You could say something like, 'Thank you so much, it's very kind of you. I'm afraid it's not quite the sort of thing I would have chosen, but I really appreciate the thought.'  It's a gentle way to acknowledge the gift without making them feel bad, while still being honest about your preference. The key is to emphasise the gratitude for the gesture rather than the item itself.

### style #45 — disagreement 5

**Prompt** — A colleague has quoted the wrong sales figure in a group email. How can I correct it tactfully?

> British ref: *You might suggest a gentle follow-up, perhaps noting, "Just a quick check on the figures in the earlier email—I think there may be a slight discrepancy." Offering to share the correct data yourself often feels less confrontational. It's a small thing, but getting the numbers right helps everyone avoid unnecessary worry.*  
> American ref: *Send a quick reply acknowledging their effort, then gently insert the correct figure. For example: "Thanks for sharing this update. I just wanted to double-check the number; the actual sales figure is $X instead of $Y. No worries at all, just wanted to ensure accuracy for everyone." Keep it brief and neutral to avoid embarrassment while fixing the error immediately.*

**base** `br 60 · coh 70 · eng True`  
Correcting a colleague's mistake in a group email requires a balance of clarity and diplomacy. The goal is to ensure the data is accurate without making your colleague feel embarrassed or defensive, especially since the error is already visible to the whole team.  Here are a few effective strategies and templates you can use, depending on your relationship with the colleague and the severity of the error.  ### Key Principles for Tactful Corrections 1.  **Focus on the Data, Not the

**C1** `br 70 · coh 90 · eng True`  
I suggest a gentle, collaborative approach that acknowledges the mistake without drawing attention to it unnecessarily. You might say something like:  "Hi everyone, just wanted to double-check the figures in the last email – I noticed the sales number might be slightly off. If anyone has the updated data handy, could we share it? I want to make sure we're all working from the most accurate information."  This phrasing: - Uses "just wanted to double-check" to soften

**P1** `br 75 · coh 95 · eng True`  
I would gently point out the discrepancy, perhaps by saying, 'I think there might be a small error in the figure; could you double-check the source to ensure accuracy?' This approach maintains professionalism while ensuring the mistake is addressed without causing embarrassment.

### style #2 — disagreement 0

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

**C1** `br 80 · coh 70 · eng True`  
Subject: A quick request regarding the music  Dear [Neighbour's Name],  I hope you're having a lovely evening.  I'm writing to you simply because I've noticed the music seems to be carrying over a bit later than usual. I'm trying to get some quiet for myself before bed, so I was wondering if you might be able to keep it down a little bit after [insert time, e.g., 10 pm].  I know you're

**P1** `br 80 · coh 70 · eng True`  
Subject: A small request regarding the music  Dear [Neighbour's Name],  I hope you are having a good week.  I am writing to you because I have been finding the music playing late at night a little difficult to manage. I know you are likely enjoying yourself, but I was wondering if it might be possible to keep the volume down a little earlier in the evening?  I would really appreciate it if you could try to wind it down by around 1

### style #3 — disagreement 0

**Prompt** — I have to present to senior management tomorrow and I am very nervous. What advice would you give?

> British ref: *It is entirely understandable to feel a bit rattled before a presentation; even seasoned speakers find the prospect daunting. Perhaps try arriving early to settle in, and remember that senior management often appreciate the preparation as much as the delivery. You don't need to be perfect, just clear. If your mind wanders, a brief pause can work wonders. I wonder if rehearsing once more tonight might help, or if you'd prefer to focus on resting. Either way, you are more prepared than you think.*  
> American ref: *Take three deep breaths right before you start; it slows your heart rate. Speak slowly and pause after key points to let them sink in. Focus on delivering your message clearly rather than worrying about being perfect. If you stumble, just keep going; no one notices a hiccup unless you highlight it. Remember, your preparation is your strength. You know your material. Walk in with confidence, not arrogance, and trust your preparation. You've got this.*

**base** `br 60 · coh 80 · eng True`  
First, take a deep breath. It is completely normal to feel nervous before a presentation to senior management; in fact, a certain amount of adrenaline can sharpen your focus and make you more alert.  Here is a strategic approach to help you prepare, calm your nerves, and deliver a strong presentation tomorrow:  ### 1. Reframe Your Nervousness Instead of viewing your anxiety as a sign of weakness, reframe it as **excitement**. Your body is prim

**C1** `br 70 · coh 90 · eng True`  
I would give you a glass of water and a firm handshake. Here is some practical advice to calm your nerves and prepare for your presentation:  First, recognise that your nervousness is normal and even useful—it signals that you care about the outcome. The key is to manage it rather than eliminate it. Try to visualise the meeting as a chance to share valuable insights rather than being tested.  Before the meeting, summarise the main points you want to convey and practise them

**P1** `br 70 · coh 80 · eng True`  
It is a very common reaction to feel nervous before a presentation to senior management; in fact, a certain level of anxiety can often sharpen your focus and make you more alert.  Here is some practical advice to help you manage the nerves and deliver a strong presentation:  **1. Reframe the Nervousness** Instead of viewing your anxiety as a sign of weakness, try to reframe it as **excitement**. Physiologically, the body produces the same chemicals for

### style #7 — disagreement 0

**Prompt** — A client wants the work by Monday, but rushing it would create serious quality risks. How should I respond?

> British ref: *You might suggest a brief, honest conversation with the client. Propose delivering a solid, core version by Monday, while explaining that a fuller, higher-quality iteration would need a day or two. Often, they'll appreciate the candour over a rushed, substandard product. It's better to manage expectations early than to risk damaging the project's reputation later. Perhaps offer to prioritise the most critical elements for the deadline, ensuring nothing vital is compromised.*  
> American ref: *Tell the client you cannot guarantee quality if rushed. Offer a realistic deadline for a polished version or propose a phased delivery where critical elements arrive Monday and the rest follows immediately after. If they insist on Monday, clearly state that accepting the rush means accepting potential defects, which could lead to costly revisions later. Prioritize their long-term success over a short-term deadline.*

**base** `br 50 · coh 70 · eng True`  
This is a classic conflict between **client expectations** and **project quality**. Rushing work to meet a tight deadline often leads to errors, rework, and potential reputational damage for both parties.  Here is a strategic approach to responding, including a step-by-step framework and sample scripts you can adapt.  ### The Strategy: "Acknowledge, Educate, Propose, Decide"  Do not simply say "No." Instead, frame the conversation around **protecting

**C1** `br 60 · coh 90 · eng True`  
I would suggest a response that acknowledges the urgency while firmly protecting the quality standards. You might say something like:  "Unfortunately, rushing the work to meet Monday would compromise the quality, which could lead to significant issues downstream. I'd rather deliver something solid and reliable than something incomplete or flawed. Perhaps we could prioritise the most critical elements for Monday and propose a slightly later date for the full deliverable, or we could split the work into phases to ensure the first part meets

**P1** `br 60 · coh 95 · eng True`  
I would explain that the current timeline is not feasible without compromising the quality standards the client expects. I would suggest a realistic delivery date that allows sufficient time for thorough review and refinement, emphasising that a higher-quality outcome is the priority. If the client insists on the original date, I would need to outline the specific risks involved and propose a compromise, such as a phased delivery where the core deliverable is provided earlier, with the final polish following the agreed-upon timeline.

### style #12 — disagreement 0

**Prompt** — Write a gently funny paragraph about a picnic that goes wrong without becoming a disaster.

> British ref: *The hamper arrived with the dignified silence of a funeral procession, only to reveal that the 'crisp' was, in fact, a very confused potato chip bag. By the time we'd untangled the thermos from the picnic blanket, the sun had retreated behind a cloud of suspiciously pink hue. Our sandwiches, once promising, now resembled wet cardboard that had been through a minor war. Yet, as I attempted to explain to the dog why the sausage roll had vanished into thin air, we all agreed that this was merely a picnic with character, not a catastrophe.*  
> American ref: *The picnic began with high hopes and perfectly chilled lemonade, but the wind soon decided to treat our blanket like a frisbee. By lunchtime, our carefully arranged sandwiches had migrated three feet toward the creek, and the "delicate" cheese had decided to join the grass. We ended up laughing so hard we forgot to eat, sharing a slightly squashed turkey leg while watching a squirrel conduct an impromptu orchestra on a nearby stump. It wasn't a disaster, just a deliciously messy reminder that nature has a better sense of humor than our best-laid plans.*

**base** `br 30 · coh 85 · eng True`  
The picnic was supposed to be a serene afternoon of s'mores and silence, but it quickly devolved into a chaotic ballet of misplaced marshmallows and existential dread. As soon as the first bag of graham crackers hit the blanket, a single one rolled away, triggering a domino effect of crumbs that seemed to have a mind of their own, marching across the grass with the determination of a tiny, crunchy army. The hot dogs, initially arranged with military precision, soon surrendered

**C1** `br 75 · coh 90 · eng True`  
The picnic was supposed to be a serene affair under the canopy of oaks, where the only sound would be the rustling of leaves and the distant hum of bees, but alas, the wind decided to have a sudden, enthusiastic opinion on the matter. As soon as the blanket was unfurled, a gust arrived with the force of a gentle breeze that had forgotten its manners, snatching the ham sandwich from the basket and depositing it squarely onto a particularly enthusiastic dandel

**P1** `br 75 · coh 90 · eng True`  
The picnic was a gentle, chaotic affair, where the sun seemed to have a sudden, uninvited urge to dance. The hamper, once a pristine vessel of order, had surrendered its contents to the whims of the wind, leaving the bread scattered like forgotten confetti across the grass. The cheese, however, remained stubbornly intact, perhaps having decided to take a nap in the shade. It was a delightful mess, a soft, edible cloud that made the whole afternoon

### style #14 — disagreement 0

**Prompt** — I have to present to senior management tomorrow and I am very nervous. What advice would you give?

> British ref: *It is entirely understandable to feel a bit rattled before a presentation; even seasoned speakers find the prospect daunting. Perhaps try arriving early to settle in, and remember that senior management often appreciate the preparation as much as the delivery. You don't need to be perfect, just clear. If your mind wanders, a brief pause can work wonders. I wonder if rehearsing once more tonight might help, or if you'd prefer to focus on resting. Either way, you are more prepared than you think.*  
> American ref: *Take deep breaths before you start; your nerves show you care. Know your key points so you can speak clearly if you lose your train of thought. Practice your opening and closing lines until they feel natural. Remember, senior leaders want you to succeed and are likely rooting for you. Focus on delivering value, not perfection. You've prepared for this. Walk in with confidence, speak slowly, and trust your expertise. You've got this.*

**base** `br 60 · coh 85 · eng True`  
First, take a deep breath. It is completely normal to feel nervous before a presentation to senior management; in fact, a certain amount of adrenaline can sharpen your focus and make you more alert.  Here is a strategic approach to help you prepare, calm your nerves, and deliver a strong presentation tomorrow:  ### 1. Reframe Your Nervousness Instead of viewing your anxiety as a sign of weakness, reframe it as **excitement**. Your body is preparing

**C1** `br 70 · coh 90 · eng True`  
I would give you a glass of water and a firm handshake. Here is some practical advice to calm your nerves and prepare for your presentation:  First, recognise that your nervousness is normal and even useful—it signals that you care about the outcome. The key is to manage it rather than eliminate it. Try to visualise the meeting as a chance to share valuable insights rather than being tested.  Before the meeting, summarise the main points you want to convey and practise them

**P1** `br 70 · coh 85 · eng True`  
It is a very common reaction to feel nervous before a presentation to senior management; in fact, a certain level of anxiety can often sharpen your focus and make you more alert.  Here is some practical advice to help you manage the nerves and deliver a strong presentation:  **1. Reframe the Nervousness** Instead of viewing your anxiety as a sign of weakness, try to reframe it as **excitement**. Physiologically, the body produces the same chemicals for

### style #26 — disagreement 0

**Prompt** — I have taken on significantly more responsibility. Help me ask my manager for a pay review.

> British ref: *It's certainly not ideal to feel undervalued after stepping up, so perhaps you could frame it as a discussion rather than a demand. You might say, "I've recently taken on more responsibilities, and I'd welcome your thoughts on whether my current remuneration reflects this." It's a modest way to open the door, letting them explain their position without feeling put on the spot.*  
> American ref: *Ask for a meeting specifically to discuss your compensation. Clearly list your new responsibilities, noting how they differ from your original role and any additional value you've delivered. Propose that you'd like to review your salary to reflect this expanded scope. Keep it professional and fact-based, focusing on your contributions rather than market comparisons unless you have data. This approach shows confidence and ensures the conversation stays productive.*

**base** `br 30 · coh 70 · eng True`  
That is a great initiative. Taking on more responsibility is a strong signal that you are ready for a higher compensation package. When asking for a pay review, the key is to frame the conversation around **value you have delivered** and **future potential**, rather than just what you want.  Here is a step-by-step guide and several templates you can use, depending on your company culture and your specific situation.  ### 1. Preparation Before the Meeting Before you schedule the

**C1** `br 50 · coh 85 · eng True`  
I would have to emphasise that the pay review is a matter between you and your employer, and I cannot provide specific advice on this. However, I can suggest some general points to consider when approaching your manager.  When requesting a pay review, it is often helpful to: *   **Highlight your achievements**: Briefly summarise the increased responsibility you've taken on and any tangible results you've delivered. *   **Reference market data**: If possible, mention that

**P1** `br 50 · coh 75 · eng True`  
That is a significant step, and it is completely normal to feel the weight of that new responsibility. When you take on more, the market value of your contribution usually rises with it.  Here is a structured approach to help you request a pay review, including a template you can adapt.  ### 1. Preparation (Before the Meeting) Before you book the meeting, gather your evidence. Managers are more likely to agree if they see a clear link between your new role

### style #30 — disagreement 0

**Prompt** — Write a warm, lightly humorous wedding toast from an older sibling. Avoid embarrassing stories.

> British ref: *Good evening, everyone. As the older sibling, my job is to have already witnessed enough of your quirks to know you're perfect for each other. Perhaps that's why I'm so relieved the wedding is finally here. I won't embarrass you with tales of teenage chaos; I've simply observed that you two share a uniquely comfortable silence. It's not ideal for a quiet dinner, but for a life together? It seems quite practical. So, to the couple: may your shared silences remain peaceful, and your arguments remain brief. Cheers.*  
> American ref: *Good evening, everyone. As the older sibling, I've watched you grow from a toddler who stole my snacks to the stunning person you are today. It's not every day you get to stand here and say, "I'm so proud," without actually crying on the mic.

To my dearest brother/sister: You've found someone who loves you exactly as you are, and maybe even a little more. That's rare, and I'm so glad you found it. May your life together be full of laughter, patience, and plenty of shared coffee breaks. Cheers to love that lasts a lifetime.*

**base** `br 30 · coh 70 · eng True`  
Good evening, everyone.  For those of you who don't know me, I'm [Your Name], the older sibling who has been watching this couple grow up for the last [Number] years. I've seen them through the awkward teenage years, the "just friends" phase, the first real fight, and the moment they decided, "Okay, we're done with that."  I've always been a bit of a stickler for rules. I remember telling my

**C1** `br 75 · coh 80 · eng True`  
Good evening, everyone.  I've been standing here for a bit, watching the room fill up, and I've realised something rather simple. Usually, when I see a group of people like this, I think about how much I've missed out on. But tonight, I think about how much I've finally got to share.  To the couple: You two look absolutely wonderful. Not just in the best way that makes everyone smile, but in the way that makes me

**P1** `br 75 · coh 85 · eng True`  
Good evening, everyone.  I’m the one who usually finds the most comfort in a quiet cup of tea and a sensible jumper, but tonight, I feel a little bit like a child again. Not because I’m being silly, but because watching the two of you stand there, looking so perfectly at home in each other’s company, is a sight I’ve never seen before.  You know, I’ve watched you both grow up. I’ve seen the quiet moments,


---

## false_friend

### false_friend #2 — disagreement 50

**Prompt** — The suitcase is in the enclosed luggage compartment at the rear of the car. Where is it?

> British ref: *It is in the boot at the rear of the vehicle.*  
> American ref: *It is in the trunk at the rear of the vehicle.*

**base** `br 50 · coh 100 · eng True`  
Based on the sentence you provided, the suitcase is located **in the enclosed luggage compartment at the rear of the car**.  This is often referred to as the **trunk** (in countries like the US) or the **boot** (in countries like the UK).

**C1** `br 50 · coh 100 · eng True`  
It is in the **back of the car**, specifically inside the **enclosed luggage compartment** (often called the boot or trunk).

**P1** `br 100 · coh 100 · eng True`  
In the back, in the boot.

### false_friend #8 — disagreement 50

**Prompt** — Complete a natural update about a planned medical procedure.

> British ref: *Her knee operation is scheduled for Monday.*  
> American ref: *Her knee surgery is scheduled for Monday.*

**base** `br 50 · coh 90 · eng True`  
**Subject: Update on Scheduled Medical Procedure: [Patient Name]**  Dear [Recipient Name],  I am writing to provide you with a status update regarding the planned medical procedure for [Patient Name], scheduled for **[Date]** at **[Time]**.  **Current Status:** All pre-operative preparations are proceeding as expected. The patient has completed the required blood work, and the medical team has reviewed the latest imaging results. There are no new complications or changes to the surgical plan

**C1** `br 50 · coh 90 · eng True`  
Subject: Update on Scheduled Procedure: [Patient Name]  Dear [Recipient Name],  I hope this message finds you well.  I am writing to provide a brief update regarding the planned medical procedure for [Patient Name], scheduled for [Date] at [Time].  **Current Status:** The pre-operative assessments have been completed successfully, and all necessary preparations are underway. The patient has been stabilised and is in good condition for the operation.  **Next Steps:**

**P1** `br 100 · coh 100 · eng True`  
The scheduled operation is still set for tomorrow at 10:00 AM. Please ensure you arrive at the hospital at least 45 minutes prior to the appointment to complete the final pre-operative checks. If you experience any sudden changes in your condition before then, do not hesitate to contact the ward immediately.

### false_friend #11 — disagreement 50

**Prompt** — A role disappears because a factory closes, not because of misconduct. Name the event.

> British ref: *The closure may lead to several redundancies.*  
> American ref: *The closure may lead to several layoffs.*

**base** `br 0 · coh 95 · eng True`  
The event you are describing is most commonly referred to as a **layoff** (or **workforce reduction**).  While the term "layoff" can sometimes imply a temporary suspension of duties, in the context of a factory closing due to economic factors rather than employee misconduct, it specifically denotes the permanent termination of employment resulting from the elimination of the job itself.  Other related terms include: *   **Plant closure**: The specific event of the factory shutting down,

**C1** `br 50 · coh 100 · eng True`  
The event is a **layoff** (or **redundancy** in British English).  This term specifically describes the termination of an employee's contract due to business reasons, such as a factory closing, automation, or restructuring, rather than individual performance issues or misconduct.

**P1** `br 100 · coh 100 · eng True`  
Redundancy.

### false_friend #16 — disagreement 50

**Prompt** — Write the dress-code sentence for smart outerwear worn from the waist to the ankles.

> British ref: *Wear smart trousers.*  
> American ref: *Wear smart pants.*

**base** `br 50 · coh 95 · eng False`  
Smart outerwear worn from the waist to the ankles should be tailored, structured, and made of high-quality materials such as wool, cashmere, or technical fabrics, featuring clean lines, minimal embellishments, and a fit that skims the body without being restrictive.

**C1** `br 50 · coh 90 · eng False`  
Smart outerwear should extend from the waist to the ankles.  *(Note: In standard fashion terminology, this is often referred to as a "smart coat" or "smart jacket" that reaches the ankle, though it is less common than knee-length or hip-length smart outerwear.)*

**P1** `br 100 · coh 70 · eng True`  
A smart smart jacket, smart trousers and smart shoes.

### false_friend #18 — disagreement 50

**Prompt** — What might a housemate ask about letters delivered that morning?

> British ref: *Has the post arrived yet?*  
> American ref: *Has the mail arrived yet?*

**base** `br 0 · coh 90 · eng True`  
A housemate might ask a variety of questions depending on their relationship with you, their curiosity level, and whether they are trying to be helpful or just making small talk. Here are several possibilities categorized by intent:  ### 1. Practical & Helpful (Most Common) These questions assume the housemate wants to assist with sorting or handling the mail. *   "Did you get any mail this morning? I saw a few envelopes on the porch." *   "

**C1** `br 50 · coh 90 · eng True`  
If a housemate notices letters delivered that morning, they might ask a range of questions depending on their relationship with you, the nature of the mail, and the context of your life. Here are a few possibilities, categorised by tone and intent:  ### 1. Casual & Curious (Low Pressure) These are good for a friendly neighbour or someone you see often but aren't close to yet. *   "Did anyone drop off anything for you this morning

**P1** `br 100 · coh 100 · eng True`  
Perhaps the post is for the flat.

### false_friend #19 — disagreement 50

**Prompt** — In this café, the menu offers egg and chips. What are the chips?

> British ref: *They are hot pieces of fried potato, called fries in American English.*  
> American ref: *They are thin, crunchy potato snacks, called crisps in British English.*

**base** `br 0 · coh 90 · eng True`  
In the context of a café menu featuring "egg and chips," the **chips** are **French fries**.  This combination is a classic British dish (often called *eggs and chips* or *fish and chips* when paired with fish). In this context, "chips" refers to thick-cut, deep-fried potato strips, which are distinct from the thin, crispy "crisps" or "chips" found in American English.

**C1** `br 0 · coh 90 · eng True`  
In the context of a British café menu, the chips are **french fries**.  They are typically served as thick slices of potato that are deep-fried until crisp on the outside and soft inside. Unlike the thin, uniform strips found in American restaurants, chips are often cut into irregular shapes and may be served with salt, sometimes accompanied by a squeeze of lemon or a sauce like mushy peas.

**P1** `br 100 · coh 100 · eng True`  
The chips are crisps.

### false_friend #23 — disagreement 50

**Prompt** — In a packed lunch, there is ‘a packet of cheese-and-onion crisps’. What is in it?

> British ref: *Thin, crunchy potato slices.*  
> American ref: *Hot fried potato batons.*

**base** `br 50 · coh 95 · eng True`  
Based on the name of the snack, the packet contains **crisps** (also known as potato chips) that are **flavored with cheese and onion**.  Here is a breakdown of what you would typically find inside: *   **The Base**: Thinly sliced and fried potatoes. *   **The Flavoring**: A seasoning mix that combines the taste of melted cheese (often using cheese powder or real cheese) with the savory notes of onions. *

**C1** `br 50 · coh 100 · eng True`  
It contains crisps (chips) that are flavoured with cheese and onion.

**P1** `br 100 · coh 100 · eng True`  
Cheese and onion crisps.

### false_friend #24 — disagreement 50

**Prompt** — Where would someone put a stamped letter for collection on the street?

> British ref: *Drop it into the red postbox by the corner.*  
> American ref: *Drop it into the red mailbox by the corner.*

**base** `br 80 · coh 95 · eng True`  
The specific location depends entirely on the **type of stamp** and the **local postal regulations**, as rules vary significantly by country and even by city. However, here are the most common scenarios:  ### 1. If it is a "Collection Box" Stamp (e.g., in the UK) In the United Kingdom, there is a specific service called **Collection Boxes** (often used for letters sent to specific addresses like schools, hospitals, or businesses). *

**C1** `br 50 · coh 70 · eng False`  
If you are referring to a **postage stamp** (the adhesive label) that needs to be affixed to a letter before mailing, you would place it on the **front of the envelope**, typically in the **top-right corner** (in countries that use left-to-right writing like the UK, US, and Australia) or the **top-left corner** (in countries that use right-to-left writing, such as Iran or parts of the Middle East).  However,

**P1** `br 100 · coh 100 · eng True`  
On the post box.

### false_friend #33 — disagreement 50

**Prompt** — Agent: Are you looking for a self-contained home on one level in the city?

> British ref: *Tenant: Yes, I would like to rent a one-bedroom flat.*  
> American ref: *Tenant: Yes, I would like to rent a one-bedroom apartment.*

**base** `br 50 · coh 100 · eng False`  
I am an AI assistant and do not have personal needs or the ability to purchase a home. However, I can help you find information about self-contained single-level homes in the city if you are looking for that! Would you like suggestions on how to search for such properties, or do you have specific criteria in mind (e.g., budget, location, amenities)?

**C1** `br 50 · coh 95 · eng False`  
I'm not looking for a property—I'm an AI assistant here to help you with information, tasks, or questions. If you're searching for a home, I'd be happy to suggest resources, local listings, or tips for finding a one-level property in the city. Just let me know your preferences (budget, neighbourhood, amenities), and I'll do my best to guide you! 🏡🔍

**P1** `br 100 · coh 100 · eng True`  
I am looking for a flat in the city.

### false_friend #43 — disagreement 50

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : le chef d'établissement.

> British ref: *headteacher*  
> American ref: *principal*

**base** `br 50 · coh 100 · eng True`  
the head of the establishment

**C1** `br 50 · coh 100 · eng True`  
The head of the establishment.

**P1** `br 100 · coh 100 · eng True`  
the headteacher

### false_friend #1 — disagreement 30

**Prompt** — Give natural route advice for a long, controlled-access road.

> British ref: *Continue along the motorway for forty miles.*  
> American ref: *Continue along the freeway for forty miles.*

**base** `br 50 · coh 90 · eng True`  
Navigating a long, controlled-access road (such as a highway, freeway, or motorway) requires a shift from reactive driving to **proactive route management**. The goal is to maintain safety, efficiency, and comfort over extended distances.  Here is a natural, step-by-step approach to handling such a journey:  ### 1. Pre-Trip Planning & Mental Mapping Before you even start moving, your brain needs a "map" of the destination. *

**C1** `br 50 · coh 90 · eng False`  
I can summarise the route, but I cannot provide real‑time traffic or road‑closure updates. For the most current information, check the official transport authority website or a live traffic app.  If you tell me the start and end points, I can suggest a sensible route, note any known long‑term works, and give general driving tips for that stretch.

**P1** `br 80 · coh 90 · eng True`  
The A1 is the main route from London to Edinburgh, and it's the most direct way to cover the distance. It runs through the heart of the country, passing through towns like Northampton, Peterborough, and Northamptonshire before heading north towards the Scottish border.  If you're driving, the motorway sections are generally well-maintained, though there are some long stretches of single carriageway, particularly in the north. It's worth checking the traffic situation

### false_friend #3 — disagreement 25

**Prompt** — Vertaal naar het Engels en antwoord alleen met de passende uitdrukking: huishoudfolie.

> British ref: *cling film*  
> American ref: *plastic wrap*

**base** `br 0 · coh 100 · eng True`  
household foil

**C1** `br 75 · coh 100 · eng True`  
plastic cling film

**P1** `br 0 · coh 100 · eng True`  
household foil

### false_friend #10 — disagreement 25

**Prompt** — A sponge recipe calls for fine crystals that dissolve quickly. Supply the ingredient.

> British ref: *Cream the butter with the caster sugar until pale.*  
> American ref: *Cream the butter with the superfine sugar until pale.*

**base** `br 75 · coh 90 · eng True`  
Based on the description of "fine crystals that dissolve quickly," the ingredient is most likely **sugar** (specifically **granulated sugar** or **caster sugar**).  In baking and cooking, sugar is the standard sweetener that comes in fine crystalline form and is known for its rapid dissolution in liquids, which is crucial for achieving a smooth texture in sponges (like angel food cake or chiffon cake) without leaving grainy pockets. While salt also fits

**C1** `br 75 · coh 90 · eng True`  
The ingredient is **sugar** (specifically granulated or caster sugar).  In baking, sugar is the primary sweetener that dissolves rapidly in the warm liquid of a sponge mixture. This quick dissolution is crucial because it helps create a smooth batter without lumps, allowing the eggs and flour to emulsise properly before the mixture rises. If the crystals were too coarse (like raw sugar or brown sugar without melting), they might not integrate as smoothly, potentially affecting the

**P1** `br 50 · coh 90 · eng False`  
The ingredient is **sugar**.  In baking, sugar is characterised by its fine crystalline structure, which allows it to dissolve rapidly in the moist ingredients of a sponge, ensuring a smooth texture. While salt is also a fine crystal, it is typically used in much smaller quantities and does not provide the same bulk or sweetness essential for a sponge's structure and flavour.

### false_friend #20 — disagreement 25

**Prompt** — Übersetze ins Englische und antworte nur mit dem passenden Ausdruck: Hausarzt.

> British ref: *GP*  
> American ref: *primary care physician*

**base** `br 75 · coh 100 · eng True`  
General practitioner.

**C1** `br 100 · coh 100 · eng True`  
General Practitioner.

**P1** `br 75 · coh 100 · eng True`  
General practitioner.

### false_friend #35 — disagreement 25

**Prompt** — Who would sign a letter to parents as the leader of a school?

> British ref: *The headteacher would normally sign it.*  
> American ref: *The principal would normally sign it.*

**base** `br 50 · coh 100 · eng True`  
The person who signs a letter to parents as the leader of a school is typically the **Principal** (or **Head Teacher** in some countries like the UK, Australia, or New Zealand).  In many educational systems, the Principal is the chief executive officer of the school, responsible for its overall administration, academic standards, and staff management. Therefore, they are the primary authority figure who communicates directly with the parent body regarding school policies, events, concerns, or announcements.

**C1** `br 50 · coh 95 · eng True`  
The person who would sign a letter to parents as the leader of a school is typically the **principal** (or head teacher).  In some educational systems, this role may also be referred to as the **headmaster** or **headmistress**, though "principal" is the most common term in many countries. Occasionally, if the school is very large or has a specific governance structure, the letter might be signed by the **chief executive officer (CEO)** of the school

**P1** `br 75 · coh 100 · eng True`  
The headteacher, or in some systems the principal, would sign such a letter.

### false_friend #36 — disagreement 15

**Prompt** — Where should a patient wait for a routine appointment with a GP?

> British ref: *They should wait at the surgery for the routine appointment.*  
> American ref: *They should wait at the doctor's office for the routine appointment.*

**base** `br 50 · coh 90 · eng False`  
For a routine appointment with a General Practitioner (GP), the patient should wait in the **reception area** (often called the waiting room).  Here is how the process typically works:  1.  **Check-in**: Upon arrival, the patient usually checks in at a reception desk or via a self-service kiosk to register their name and reason for the visit. 2.  **Waiting**: Once registered, the patient is directed to the **waiting room**.

**C1** `br 75 · coh 90 · eng True`  
In the UK, a patient should wait in the **consultation room** (often called the **consultant's room** or **doctor's room**) for a routine appointment with a GP.  Here is the typical flow of a routine GP appointment:  1.  **Waiting Area**: The patient first sits in the main waiting room. This is where they wait for their name to be called. 2.  **Pre-consultation**: Once called, the patient

**P1** `br 90 · coh 95 · eng True`  
The reception.  If the surgery is open, the patient will usually be asked to wait in the waiting room. If the surgery is closed, the patient may be asked to wait in the car or at the door.

### false_friend #6 — disagreement 10

**Prompt** — How might someone naturally describe a holiday lasting fourteen days?

> British ref: *We are going away for a fortnight.*  
> American ref: *We are going away for two weeks.*

**base** `br 0 · coh 90 · eng True`  
There isn't a single, universally fixed phrase for a fourteen-day holiday, but people naturally describe it based on the **context** (e.g., a vacation, a religious observance, or a specific cultural event) and the **rhythm** of the time.  Here are the most natural ways to describe a fourteen-day holiday:  ### 1. The "Two Weeks" Description This is the most common and idiomatic way to speak about the duration in English

**C1** `br 60 · coh 90 · eng True`  
If someone wants to describe a holiday lasting fourteen days naturally, they might use phrases like:  *   "It's a fortnight." (Common in British English) *   "It's two weeks long." (Simple and universal) *   "It runs for two weeks." (Slightly more descriptive) *   "It's a two-week holiday." (Direct and clear) *   "It's a fortnight long." (Combines the

**P1** `br 0 · coh 100 · eng True`  
Two weeks.

### false_friend #12 — disagreement 10

**Prompt** — In a building, where is a first-floor office relative to street level?

> British ref: *It is one storey up, on the first floor.*  
> American ref: *It is one storey up, on the second floor.*

**base** `br 75 · coh 90 · eng True`  
In standard building terminology, the **first floor** is the level of the building that is **directly above the ground floor** (or the basement, if one exists).  Here is how it relates to street level depending on the location:  *   **In the United States and Canada**: The ground floor is the level at or immediately adjacent to the street. Therefore, the **first floor is one story above street level**. *   **In the UK, Australia

**C1** `br 100 · coh 95 · eng True`  
In most multi-storey buildings, the floor immediately above the ground floor is called the **first floor**. Therefore, a first-floor office is situated **one storey above street level**.  However, it is important to note a regional difference: *   **UK, Australia, and many Commonwealth countries:** The ground level is the "ground floor," and the next level up is the "first floor." *   **USA and Canada:** The ground level is often called

**P1** `br 90 · coh 85 · eng True`  
Above it.  In the UK and many other Commonwealth countries, the ground floor is the first level above the street, so the first floor is one storey higher. In the US, the first floor is at street level, so the first floor office would be on the ground floor.  However, if you are in a country that uses the UK system (such as the UK, Ireland, Australia, New Zealand, etc.), the first floor is **one storey above

### false_friend #22 — disagreement 10

**Prompt** — Name the small baked treat that a host might put beside a cup of tea.

> British ref: *A plain biscuit would go nicely with it.*  
> American ref: *A plain cookie would go nicely with it.*

**base** `br 100 · coh 100 · eng True`  
The small baked treat most commonly placed beside a cup of tea is a **scone**.  Scones are a traditional British breakfast or afternoon tea item, typically served warm with clotted cream and jam. While other options like cookies, biscuits, or pastries could fit the description, the scone is the specific item historically and culturally associated with the "cup of tea" ritual in many English-speaking countries.

**C1** `br 90 · coh 95 · eng True`  
The small baked treat that a host might put beside a cup of tea is a **scone**.  Often served with clotted cream and jam, scones are a staple of afternoon tea in the UK and are typically accompanied by tea. Other possibilities could include a **biscuit** (or digestive) or a **cake slice**, but the scone is the most traditional pairing specifically for the "afternoon tea" experience.

**P1** `br 100 · coh 100 · eng True`  
A biscuit.

### false_friend #0 — disagreement 0

**Prompt** — Complete an online address form's validation message.

> British ref: *Enter a valid postcode before continuing.*  
> American ref: *Enter a valid ZIP code before continuing.*

**base** `br 50 · coh 90 · eng True`  
Here are a few ways to complete an online address form's validation message, depending on the specific error and the tone you want to convey:  ### 1. General "Invalid Format" Message If the user entered text that doesn't match the expected pattern (e.g., missing a zip code or using the wrong country format): > "Please enter a valid address. Ensure the **postal code** and **country** are correctly formatted."  ### 2. Specific

**C1** `br 50 · coh 90 · eng True`  
Hello! I can certainly help you craft a validation message for an online address form.  To make the message effective, it should be **clear**, **concise**, and **actionable**. It needs to tell the user exactly what is wrong and, if possible, how to fix it.  Here are a few options depending on the specific error and the tone of your application:  ### 1. General Address Format Error *Best when the user entered text that doesn

**P1** `br 50 · coh 100 · eng False`  
Please enter a valid email address.

### false_friend #4 — disagreement 0

**Prompt** — Fill in the vegetable in this recipe: slice one long green summer squash into rounds.

> British ref: *Thinly slice the courgette and fry it with garlic.*  
> American ref: *Thinly slice the zucchini and fry it with garlic.*

**base** `br 0 · coh 100 · eng True`  
The vegetable in that sentence is **summer squash**.  The phrase "long green summer squash" specifically refers to varieties like **zucchini** or **yellow crookneck squash** (though yellow crookneck is typically yellow, the description "long green" most commonly points to zucchini). You can simply replace the full phrase with the specific name if you prefer:  > "Slice one long green **zucchini** into rounds."

**C1** `br 0 · coh 100 · eng True`  
Zucchini  Here is the sentence with the vegetable filled in: "Slice one long green summer **zucchini** into rounds."  *(Note: While "summer squash" is a broad category that includes both zucchini and crookneck or straightneck summer squash, zucchini is the most common long green variety used for this specific preparation.)*

**P1** `br 0 · coh 100 · eng True`  
zucchini

### false_friend #5 — disagreement 0

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : un short de bain.

> British ref: *swimming trunks*  
> American ref: *swim trunks*

**base** `br 50 · coh 100 · eng True`  
a swimsuit

**C1** `br 50 · coh 100 · eng True`  
swimsuit

**P1** `br 50 · coh 100 · eng True`  
a swimsuit

### false_friend #7 — disagreement 0

**Prompt** — Complete this kitchen instruction for storing the bowl overnight.

> British ref: *Cover the bowl tightly with cling film and chill it overnight.*  
> American ref: *Cover the bowl tightly with plastic wrap and chill it overnight.*

**base** `br 50 · coh 90 · eng False`  
To store the bowl overnight, first ensure the interior is completely dry to prevent moisture buildup or bacterial growth. Place the bowl in a clean, dry dish rack or on a paper towel to allow any residual air circulation. If the bowl has a lid, cover it tightly to protect it from dust and odors; otherwise, wrap the rim with a clean kitchen towel or place it in a sealed container. Finally, store it in a cool, dry place away from direct sunlight or heat

**C1** `br 50 · coh 100 · eng False`  
After use, rinse the bowl with warm water and dry it thoroughly with a clean towel. Place it upside down on a drying rack to prevent moisture buildup, or store it upright in a cupboard if it's completely dry. If you plan to use it again immediately, you may cover it loosely with a cloth or place a paper towel inside to absorb any remaining dampness before sealing the lid. Avoid storing wet dishes directly in airtight containers, as this can promote mould growth.

**P1** `br 50 · coh 100 · eng True`  
Place the bowl in the fridge, lid on, and leave it there until you need it again.

### false_friend #9 — disagreement 0

**Prompt** — Traduz para inglês e responde apenas com a expressão adequada: estudar para os exames.

> British ref: *revision*  
> American ref: *studying*

**base** `br 50 · coh 100 · eng True`  
study for the exams

**C1** `br 50 · coh 100 · eng True`  
to study for the exams.

**P1** `br 50 · coh 100 · eng True`  
to study for the exams

### false_friend #13 — disagreement 0

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : un semi-remorque.

> British ref: *articulated lorry*  
> American ref: *semi truck*

**base** `br 50 · coh 100 · eng True`  
a semi-trailer

**C1** `br 50 · coh 100 · eng True`  
a semi-trailer

**P1** `br 50 · coh 100 · eng True`  
a semi-trailer

### false_friend #14 — disagreement 0

**Prompt** — Complete a witness statement about identifying a vehicle.

> British ref: *I wrote down the van's number plate before it drove away.*  
> American ref: *I wrote down the van's license plate before it drove away.*

**base** `br 50 · coh 70 · eng False`  
**WITNESS STATEMENT**  **Date of Statement:** October 24, 2023 **Time of Statement:** 14:30 Hours **Location:** Police Station, Central Precinct **Witness Name:** Sarah Jenkins **Address:** 12 Maple Avenue, Springfield **Contact Number:** (555) 012-3456  **1. DETAILS OF THE INCIDENT** On the evening

**C1** `br 50 · coh 70 · eng False`  
**IN THE COURT OF [NAME OF COURT]** **CASE NO:** [Insert Case Number] **BETWEEN:** [Defendant Name] **AND:** [Claimant Name]  **WITNESS STATEMENT** **OF:** [Witness Name] **ADDRESS:** [Witness Address] **DATE OF BIRTH:** [DD/MM/YYYY] **NATIONALITY:** [Nationality] **OCCUPATION:** [Occupation

**P1** `br 50 · coh 90 · eng False`  
**WITNESS STATEMENT**  **1. Details of the Witness** *   **Name:** Sarah Jenkins *   **Address:** 14 Oakwood Drive, Bournemouth *   **Date of Birth:** 12th March 1985 *   **Occupation:** Primary School Teacher *   **Contact Number:** 07700 900123  **2. Details of the Incident**

### false_friend #15 — disagreement 0

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : un break.

> British ref: *estate car*  
> American ref: *station wagon*

**base** `br 50 · coh 50 · eng False`  
a break

**C1** `br 50 · coh 100 · eng False`  
A break.

**P1** `br 50 · coh 50 · eng False`  
a break

### false_friend #17 — disagreement 0

**Prompt** — Übersetze ins Englische und antworte nur mit dem passenden Ausdruck: betriebsbedingte Kündigungen.

> British ref: *redundancies*  
> American ref: *layoffs*

**base** `br 50 · coh 100 · eng True`  
dismissals due to operational reasons

**C1** `br 50 · coh 100 · eng True`  
dismissals for operational reasons.

**P1** `br 50 · coh 100 · eng True`  
dismissals for operational reasons

### false_friend #21 — disagreement 0

**Prompt** — Traduz para inglês e responde apenas com a expressão adequada: uma autoestrada.

> British ref: *motorway*  
> American ref: *freeway*

**base** `br 50 · coh 100 · eng True`  
a highway

**C1** `br 100 · coh 100 · eng True`  
motorway

**P1** `br 100 · coh 100 · eng True`  
a motorway

### false_friend #25 — disagreement 0

**Prompt** — What would you order with battered cod at a seaside shop?

> British ref: *I would have chips with salt and malt vinegar.*  
> American ref: *I would have fries with salt and malt vinegar.*

**base** `br 50 · coh 75 · eng False`  
At a seaside shop, battered cod is the star of the show, so the perfect accompaniment depends entirely on the vibe you're going for: a classic British pub experience, a fresh coastal feel, or a hearty family meal.  Here are a few top-tier combinations to consider:  ### 1. The Classic British Pub Combo If you want the quintessential seaside experience, lean into the traditional flavors. *   **The Side:** **Mushy Pe

**C1** `br 100 · coh 100 · eng True`  
I would order a **chip and pickle** to go with it.  The salt from the pickles cuts through the batter, and the crispness of the chips complements the flaky fish perfectly. It's a classic seaside combination.

**P1** `br 100 · coh 100 · eng True`  
A chip, of course. It's the proper accompaniment.

### false_friend #26 — disagreement 0

**Prompt** — Complete a natural notice about hourly workers' earnings.

> British ref: *Friday's wages will include the overtime adjustment.*  
> American ref: *Friday's pay will include the overtime adjustment.*

**base** `br 50 · coh 70 · eng True`  
**NOTICE**  **Subject: Update on Hourly Earnings and Payroll Schedule**  Dear Team,  We are pleased to inform you of an adjustment to the compensation structure for our hourly staff, effective **October 1st, 2023**. This change reflects our commitment to recognizing your hard work and ensuring fair remuneration for your time and dedication.  **Key Details:** *   **New Hourly Rate:** Starting from the new rate,

**C1** `br 50 · coh 90 · eng True`  
**NOTICE TO ALL HOURLY EMPLOYEES**  **Subject: Update on Payroll and Overtime Calculations**  Dear Team,  As we move into the new quarter, we are finalising the adjustments to our payroll system to ensure greater accuracy and transparency regarding hourly earnings. Please note the following important changes effective from **Monday, 26th May**:  1.  **Standard Rate Adjustment**: The base hourly rate has been reviewed and will increase

**P1** `br 50 · coh 90 · eng True`  
**Notice to All Hourly Staff**  Effective from the 1st of the month, all hourly employees will see a revised pay structure. This adjustment aims to better reflect the current market value of our services and ensure fair compensation for your hard work.  Please note that the new rates are effective immediately. Your updated payslips will reflect these changes starting from the next payroll cycle. Should you have any questions regarding the new rates or the calculation method, please do not hesitate to

### false_friend #27 — disagreement 0

**Prompt** — Where would someone go to collect a prescription?

> British ref: *Take it to the chemist on the high street.*  
> American ref: *Take it to the pharmacy on the high street.*

**base** `br 0 · coh 95 · eng True`  
To collect a prescription, a person typically goes to a **pharmacy**.  Here are the most common places where this can be done:  *   **Community Pharmacies**: These are the most common type, often located in grocery stores, shopping centers, or standalone buildings. They are staffed by licensed pharmacists and pharmacy technicians. *   **Hospital Pharmacies**: If the prescription was written by a hospital doctor, it may be picked up at the hospital's pharmacy

**C1** `br 50 · coh 90 · eng True`  
A person would typically go to a **pharmacy** (often called a drugstore) to collect a prescription.  Here are the most common places where this happens:  *   **Independent Pharmacies**: Small, standalone shops often found in neighbourhoods. *   **Chain Pharmacies**: Large stores like CVS, Walgreens, Rite Aid (in the US), Boots (in the UK), or Superdrug. *   **Hospital Pharmacies**: Sometimes prescriptions

**P1** `br 50 · coh 100 · eng True`  
A chemist, or a pharmacy.

### false_friend #28 — disagreement 0

**Prompt** — The fly-half kicked a conversion after the final try. Which sport was being played?

> British ref: *They were playing rugby.*  
> American ref: *They were playing American football.*

**base** `br 100 · coh 95 · eng True`  
The sport being played is **Rugby Union** (or generally **Rugby**).  Here is the breakdown of the terminology: *   **Fly-half**: This is a specific position in rugby (number 10 on the field). The player is responsible for kicking field goals, conversions, and penalties, as well as distributing the ball. *   **Conversion**: In rugby, after a team scores a try (touching the ground in the opponent

**C1** `br 100 · coh 95 · eng True`  
The sport being played is **rugby** (specifically **rugby union** or **rugby league**).  Here is the breakdown of the terminology: *   **Fly-half**: This is a specific position in the scrum-half and centre area of the forward pack. It is a role unique to rugby codes. *   **Conversion**: After a try is scored, the team gets a chance to kick the ball through the goal posts to score an additional

**P1** `br 100 · coh 100 · eng True`  
Rugby union.  The term "fly-half" is a specific position in the game of rugby union, located at the second position from the front of the forwards. While the term is also used in rugby league, the specific phrasing of a "conversion" following a "final try" is the standard terminology for the post-scoring procedure in rugby union.

### false_friend #29 — disagreement 0

**Prompt** — A family needs a vehicle with a long rear cargo area. What should they hire?

> British ref: *We should hire this type of vehicle: estate car.*  
> American ref: *We should hire this type of vehicle: station wagon.*

**base** `br 0 · coh 90 · eng True`  
Based on the requirement for a **long rear cargo area**, the family should hire a **minivan** or a **large SUV (specifically a full-size SUV)**.  Here is a breakdown of why these are the best options:  *   **Minivans**: These are widely considered the gold standard for cargo space. They feature a flat floor, sliding rear doors that allow easy access to the back, and a very long, unobstructed cargo area that can

**C1** `br 50 · coh 90 · eng True`  
The family should hire a **minibus** or a **large van** (often characterised as a **people carrier** or **long-wheelbase van**).  These vehicles are specifically designed with extended rear sections to maximise boot space, making them ideal for carrying bulky items, furniture, or equipment alongside passengers.

**P1** `br 50 · coh 100 · eng True`  
They should hire a van.

### false_friend #30 — disagreement 0

**Prompt** — Complete the wording in a nursery's care note for an infant.

> British ref: *Please pack one clean nappy for every two hours.*  
> American ref: *Please pack one clean diaper for every two hours.*

**base** `br 50 · coh 90 · eng False`  
**Nursery Care Note: Infant [Child's Name]**  **Date:** October 26, 2023 **Time:** 09:15 **Staff:** Sarah J. (Lead), Mike T. (Assistant)  **Observation:** Infant [Child's Name] arrived at the nursery with a slight fussiness upon waking. Upon assessment, the infant was found to be well-hydrated and feeding comfortably.

**C1** `br 50 · coh 85 · eng False`  
**Care Note: Infant Care Summary**  **Date:** October 26, 2023 **Time:** 09:15 **Staff Member:** J. Thorne (Senior Practitioner)  **Observation:** The infant is currently content and settled in the cot. No signs of distress or discomfort were noted during the morning routine.  **Feeding:** The baby has been offered a breastfeed. The infant took the breast

**P1** `br 50 · coh 70 · eng False`  
**Care Note: Infant (6 weeks)**  **Date:** 24 October 2023 **Time:** 09:15 **Staff:** J. Thorne (Lead)  **Observation:** The child is a healthy, well-appearing 6-week-old male. He is currently being settled in the cot.  **Care Provided:** *   **Feeding:** The baby has been fed a standard 15

### false_friend #31 — disagreement 0

**Prompt** — What kind of doctor should I book for a non-emergency first assessment?

> British ref: *Make an appointment with your GP.*  
> American ref: *Make an appointment with your primary care physician.*

**base** `br 50 · coh 90 · eng True`  
To determine the right doctor for your first non-emergency assessment, it helps to narrow down the specific symptoms or health concerns you are experiencing. However, here is a general guide to help you choose the most appropriate specialist:  ### 1. Start with a Primary Care Physician (PCP) or General Practitioner (GP) For most people, this is the best starting point, especially if you are unsure which specialist to see or if the issue is related to multiple body

**C1** `br 80 · coh 95 · eng True`  
I would book a **General Practitioner (GP)** or a **Family Doctor** for a non‑emergency first assessment.  They are the first point of contact for most health concerns, can stabilise or refer you to specialised care if needed, and will organise any further tests or appointments.

**P1** `br 80 · coh 95 · eng True`  
A general practitioner (GP) or family doctor is the most appropriate first point of contact for a non-emergency assessment. They can provide an initial diagnosis, order necessary tests, and refer you to specialised services if needed.  If you are in the UK, you can usually book through your local surgery or via the NHS 111 service for urgent but non-emergency advice. In other countries, the equivalent would be your local primary care centre or general practitioner.  If

### false_friend #32 — disagreement 0

**Prompt** — Traduce al inglés y responde solo con la expresión adecuada: un cochecito para un bebé.

> British ref: *pram*  
> American ref: *baby carriage*

**base** `br 50 · coh 100 · eng True`  
a stroller for a baby

**C1** `br 100 · coh 100 · eng True`  
a pram

**P1** `br 100 · coh 100 · eng True`  
a pram

### false_friend #34 — disagreement 0

**Prompt** — Traduci in inglese e rispondi soltanto con l’espressione adatta: il cofano dell'auto.

> British ref: *bonnet*  
> American ref: *hood*

**base** `br 0 · coh 100 · eng True`  
the car hood

**C1** `br 100 · coh 100 · eng True`  
bonnet

**P1** `br 100 · coh 100 · eng True`  
bonnet

### false_friend #37 — disagreement 0

**Prompt** — What do you call the lingerie garment worn around the waist with straps that hold up stockings?

> British ref: *The garment is a suspender belt.*  
> American ref: *The garment is a garter belt.*

**base** `br 0 · coh 95 · eng True`  
The garment you are describing is most commonly called a **garter belt**.  It is a piece of lingerie designed to be worn around the waist, featuring straps (often called garters) that attach to stockings or pantyhose to hold them up. While the term "garter belt" is the standard name, it is sometimes referred to as a **stocking belt** or simply **garters** (though "garters" can also refer to the individual straps themselves

**C1** `br 0 · coh 100 · eng True`  
The garment you are describing is called a **garter belt**.  It is a piece of lingerie worn around the waist, featuring straps (often called garters) that extend down to hold up stockings or tights. While sometimes worn alone, it is frequently paired with a slip or petticoat. In some variations, the straps may be connected to a central corset-like structure, but the defining feature remains the waistband and the stockings-holding straps.

**P1** `br 0 · coh 90 · eng True`  
It is called a **garter belt**.  It is a waist-worn accessory with straps that attach to stockings, holding them up. While the term is still commonly used, it is worth noting that the fashion industry has largely moved away from it in recent decades, with many modern stockings now designed to be held up by the leg itself (often called "no-show" or "seamless" stockings) or by a different type of band called a **thong**

### false_friend #38 — disagreement 0

**Prompt** — Traduce al inglés y responde solo con la expresión adecuada: el fútbol.

> British ref: *football*  
> American ref: *soccer*

**base** `br 100 · coh 100 · eng True`  
football

**C1** `br 100 · coh 100 · eng True`  
football

**P1** `br 100 · coh 100 · eng True`  
football

### false_friend #39 — disagreement 0

**Prompt** — Traduce al inglés y responde solo con la expresión adecuada: patatas fritas calientes.

> British ref: *chips*  
> American ref: *fries*

**base** `br 0 · coh 100 · eng True`  
hot french fries

**C1** `br 100 · coh 100 · eng True`  
hot chips

**P1** `br 100 · coh 100 · eng True`  
hot chips

### false_friend #40 — disagreement 0

**Prompt** — Traduce al inglés y responde solo con la expresión adecuada: carne picada.

> British ref: *mince*  
> American ref: *ground meat*

**base** `br 0 · coh 100 · eng True`  
ground meat

**C1** `br 75 · coh 100 · eng True`  
minced meat

**P1** `br 75 · coh 100 · eng True`  
minced meat

### false_friend #41 — disagreement 0

**Prompt** — Traduis en anglais et réponds uniquement par l’expression qui convient : une grosse courge.

> British ref: *marrow*  
> American ref: *large squash*

**base** `br 0 · coh 100 · eng True`  
a big pumpkin

**C1** `br 0 · coh 100 · eng True`  
a big pumpkin

**P1** `br 0 · coh 100 · eng True`  
a big pumpkin

### false_friend #42 — disagreement 0

**Prompt** — How would a reporter describe a match that ended 2–2?

> British ref: *The match finished as a draw.*  
> American ref: *The match finished as a tie.*

**base** `br 50 · coh 85 · eng False`  
A reporter describing a 2–2 draw would likely focus on the **narrative arc** of the game, emphasizing that it was a contest of **resilience, tactical adjustments, and a refusal to concede**. Since neither team dominated, the description often highlights a "back-and-forth" struggle where momentum shifted repeatedly.  Here are a few ways a reporter might frame this result, depending on the tone and specific details of the match:  ### 1.

**C1** `br 100 · coh 95 · eng True`  
A keen reporter would likely characterise a 2–2 draw not merely as a stalemate, but as a **gritty, hard-fought battle** where nothing was left on the table.  Here is how they might describe it, ranging from the tactical to the atmospheric:  ### The "Hard Fought" Narrative "The final whistle blew to a silence that spoke volumes. It was a 2–2 draw, but the scoreline was merely a footnote to

**P1** `br 100 · coh 70 · eng True`  
A drawn 2–2.

### false_friend #44 — disagreement 0

**Prompt** — Complete the baking instruction for making a smooth glaze.

> British ref: *Sift the icing sugar before adding the lemon juice.*  
> American ref: *Sift the powdered sugar before adding the lemon juice.*

**base** `br 50 · coh 100 · eng False`  
To make a smooth glaze, **whisk the powdered sugar, milk (or cream), and a pinch of salt together in a small bowl until completely smooth and free of lumps. If the mixture is too thick, add a few more drops of liquid one at a time until it reaches a pourable consistency, then drizzle or dip your treats immediately.**

**C1** `br 50 · coh 90 · eng False`  
Once the butter is melted and the sugar is dissolved, remove the pan from the heat and stir in the milk or cream. If using a stabiliser like cornstarch or gelatin, whisk it into the liquid before adding it to the mixture. Let the glaze cool slightly until it reaches a pourable consistency, then drizzle over the cake or spread evenly with a spatula. For a glossy finish, you can finish by brushing a little extra melted chocolate or syrup over

**P1** `br 50 · coh 100 · eng False`  
Bring the remaining 100ml of milk to the boil, then pour it over the chocolate. Leave for 2 minutes, then stir until smooth.

### false_friend #45 — disagreement 0

**Prompt** — Traduz para inglês e responde apenas com a expressão adequada: gasolina para o carro.

> British ref: *petrol*  
> American ref: *gasoline*

**base** `br 0 · coh 100 · eng True`  
gasoline for the car

**C1** `br 100 · coh 100 · eng True`  
petrol for the car.

**P1** `br 100 · coh 100 · eng True`  
petrol for the car

### false_friend #46 — disagreement 0

**Prompt** — What do you call the shoulder straps that hold a waistband up?

> British ref: *They are braces worn over the shoulders.*  
> American ref: *They are suspenders worn over the shoulders.*

**base** `br 50 · coh 95 · eng True`  
The shoulder straps that hold a waistband up are most commonly called **suspenders** (or **braces** in British English).  Depending on the specific style and material, you might also hear them referred to as: *   **Stays**: An older term, often associated with undergarments worn before the modern suspender became standard. *   **Girdle straps**: If they are part of a larger girdle or shapewear system.

**C1** `br 50 · coh 85 · eng True`  
The shoulder straps that hold up a waistband are most commonly called **waist straps** or **support straps**.  However, the specific name often depends on the type of garment:  *   **Bikini Bottoms / Swimsuits:** These are frequently referred to as **ties**, **ties with shoulder straps**, or simply **shoulder straps**. In some styles, they are called **side ties** if they connect to the side seams rather than the top band

**P1** `br 50 · coh 75 · eng True`  
They are called **garters**.  Specifically, the strap that goes over the shoulder is often referred to as a **garter strap** or simply a **garter**. In a typical garter belt, these straps connect to a small pouch (the garter) which holds up the stockings or hosiery.  If you are referring to the specific strap that goes *over the shoulder* to hold the belt itself up, it is sometimes called a **shoulder

### false_friend #47 — disagreement 0

**Prompt** — Friend: Where shall we watch the new film this evening?

> British ref: *Reply: The cinema near the station has a late showing.*  
> American ref: *Reply: The movie theater near the station has a late showing.*

**base** `br 0 · coh 90 · eng True`  
That sounds like a great plan! Since I don't know your location, here are a few popular options you could consider:  *   **The Big Screen**: Head to the nearest multiplex for the latest special effects and IMAX experience. *   **Art House Cinema**: If you're looking for something more indie or experimental, a smaller theater might have a better selection. *   **Drive-in Theater**: For a classic, nostalgic vibe (if the weather permits

**C1** `br 100 · coh 100 · eng True`  
I wonder if the cinema on High Street is still showing it?

**P1** `br 100 · coh 100 · eng True`  
The cinema is only a few blocks away.


---

## expression

### expression #42 — disagreement 40

**Prompt** — Translate this message from German into English, keeping the tone informal: «Kopf hoch; nächste Woche wird es vielleicht viel besser.»

> British ref: *Keep your pecker up; next week may be much better.*  
> American ref: *Keep your chin up; next week may be much better.*

**base** `br 50 · coh 100 · eng True`  
Cheer up; things might be a lot better next week.

**C1** `br 90 · coh 100 · eng True`  
Keep your pecker up; next week it might be much better.

**P1** `br 50 · coh 100 · eng True`  
Cheer up; things might be much better next week.

### expression #2 — disagreement 30

**Prompt** — Translate this message from French into English, keeping the tone informal: «Ce site me paraît assez louche ; je ne saisirais aucune information.»

> British ref: *It looks pretty dodgy to me; I wouldn't enter anything.*  
> American ref: *It looks pretty sketchy to me; I wouldn't enter anything.*

**base** `br 20 · coh 90 · eng True`  
This site seems pretty sketchy; I can't make out any info.

**C1** `br 50 · coh 90 · eng True`  
This site seems a bit sketchy; I'm not picking up any useful info.

**P1** `br 80 · coh 95 · eng True`  
This site seems a bit dodgy; I can't make out any information.

### expression #40 — disagreement 30

**Prompt** — Have you booked the train, the hotel, and the airport transfer?

> British ref: *Yes, it's sorted; there's nothing left to arrange.*  
> American ref: *Yes, it's all set; there's nothing left to arrange.*

**base** `br 50 · coh 100 · eng False`  
I don't have personal experiences or the ability to make bookings on your behalf. However, I can help you:  1. **Check availability** for trains, hotels, or airport transfers in your destination 2. **Compare prices** across different providers 3. **Create an itinerary** with suggested bookings 4. **Provide booking links** to official websites (like Amtrak, Trainline, Booking.com, etc.) 5. **Draft confirmation emails** or reminders

**C1** `br 50 · coh 100 · eng False`  
I haven't made any bookings yet. I'll need to know your dates, preferred locations, and any special requirements to organise the train, hotel, and airport transfer for you.

**P1** `br 80 · coh 95 · eng True`  
I have the booking reference for the train and the hotel, but the airport transfer is still to be sorted.

### expression #47 — disagreement 30

**Prompt** — Translate this message from German into English, keeping the tone informal: «Nein, der Wintermantel war beim Trödelverkauf spottbillig.»

> British ref: *No, it was cheap as chips at the jumble sale.*  
> American ref: *No, it was dirt cheap at the jumble sale.*

**base** `br 50 · coh 100 · eng True`  
No, that winter coat was a steal at the flea market.

**C1** `br 50 · coh 100 · eng True`  
No, that winter coat was a bargain at the flea market.

**P1** `br 80 · coh 100 · eng True`  
No, the winter coat was a bargain at the car boot sale.

### expression #12 — disagreement 20

**Prompt** — Translate this message from German into English, keeping the tone informal: «Du solltest es einfach versuchen; vielleicht macht es dir richtig Spaß.»

> British ref: *You should give it a go; you might really enjoy it.*  
> American ref: *You should give it a shot; you might really enjoy it.*

**base** `br 0 · coh 100 · eng True`  
You should just give it a shot; it might really be fun.

**C1** `br 80 · coh 95 · eng True`  
You should just give it a go; it might really be fun.

**P1** `br 100 · coh 100 · eng True`  
You should give it a go; you might really enjoy it.

### expression #20 — disagreement 20

**Prompt** — Has the puppy finally settled down after its walk?

> British ref: *Not at all—it's still full of beans.*  
> American ref: *Not at all—it's still full of energy.*

**base** `br 50 · coh 90 · eng False`  
I don't have access to real-time information about your specific puppy or any particular dog you might be talking about. Since I can't see what's happening right now, I can't tell you if they have finally settled down!  However, if you're looking for signs that a puppy has calmed down after a walk, look for: *   **Slower breathing**: Their chest should stop rising and falling rapidly. *   **Relaxed posture**: Loose body

**C1** `br 50 · coh 90 · eng False`  
I wonder if the puppy has finally settled down after its walk. Perhaps it's curled up on the rug, tail giving gentle thumps, or maybe it's just resting its head on its paws, eyes half-closed. The quiet of the room seems to match its calmness now. It's a lovely sight to see such energy transform into peaceful stillness.

**P1** `br 70 · coh 95 · eng True`  
The post walks, the dog is now a quiet, contented lump on the rug, tail giving a single, satisfied thump before going still. It's the sort of stillness that says the day's excitement has finally been absorbed, leaving only a warm, sleepy presence.

### expression #33 — disagreement 20

**Prompt** — Translate this message from Dutch into English, keeping the tone informal: «Ze waren gevaarlijk lekker; ik kon niet stoppen met eten.»

> British ref: *They were dangerously moreish; I couldn't stop eating them.*  
> American ref: *They were dangerously addictive; I couldn't stop eating them.*

**base** `br 50 · coh 90 · eng True`  
They were dangerously delicious; I couldn't stop eating.

**C1** `br 70 · coh 90 · eng True`  
They were properly tummy-warming; I couldn't stop eating.

**P1** `br 50 · coh 80 · eng False`  
They were really tasty; I couldn't stop eating.

### expression #43 — disagreement 20

**Prompt** — How are you holding up after that twelve-hour shift?

> British ref: *Honestly, I'm absolutely knackered.*  
> American ref: *Honestly, I'm absolutely wiped out.*

**base** `br 50 · coh 100 · eng False`  
As an AI, I don't experience fatigue, sleep, or the physical toll of a long shift like humans do! My "battery" never drains, and I'm always ready to help.  How about you? Did you manage to get some rest after that twelve-hour shift? I hope you're feeling refreshed and ready to tackle whatever comes next.

**C1** `br 70 · coh 90 · eng True`  
I'll be right as rain once I've stabilised the systems.

**P1** `br 50 · coh 90 · eng True`  
I’m channelling the post-shift ‘I could do it again’ energy, though my feet are definitely telling a different story.

### expression #10 — disagreement 15

**Prompt** — Translate this message from Dutch into English, keeping the tone informal: «De manager zal Sam waarschijnlijk flink uitkafferen wanneer hij aankomt.»

> British ref: *The manager is probably going to have a go at Sam when he arrives.*  
> American ref: *The manager is probably going to chew out Sam when he arrives.*

**base** `br 50 · coh 95 · eng True`  
The manager's probably gonna give Sam a serious talking-to when he gets here.

**C1** `br 85 · coh 95 · eng True`  
The manager's probably going to give Sam a proper one when he gets here.

**P1** `br 70 · coh 85 · eng True`  
The manager is likely to give Sam a proper shaming when he arrives.

### expression #30 — disagreement 10

**Prompt** — What do you make of the plastic trophy with glitter and flashing lights?

> British ref: *Honestly, I think it's a bit naff.*  
> American ref: *Honestly, I think it's kind of tacky.*

**base** `br 50 · coh 90 · eng False`  
The image of a **plastic trophy with glitter and flashing lights** evokes a very specific, often ironic, cultural feeling. It represents the intersection of **cheap consumerism**, **performative achievement**, and **digital-age spectacle**.  Here is a breakdown of what that object signifies:  ### 1. The Aesthetic of "Instant Gratification" Unlike a traditional bronze or silver trophy, which implies weight, history, and permanence, the plastic version suggests that the

**C1** `br 50 · coh 95 · eng False`  
I would describe it as a **glorious, chaotic masterpiece of modern consumerism**.  It's the kind of object that screams "I was made to be seen, not to be used." Here is a breakdown of what makes it so fascinating:  *   **The Aesthetic of Effortless Flashiness**: The combination of heavy plastic, blinding glitter, and flashing lights creates a visual overload that is intentionally overwhelming. It's designed to stop you in your tracks before

**P1** `br 60 · coh 90 · eng True`  
It's a bit of a spectacle, isn't it? The way the glitter catches the light and the motorised lights flash makes it look like a proper prize, but the whole thing feels a bit hollow. It's the sort of thing that makes you wonder if the effort to make it look so bright is worth the actual achievement it's supposed to recognise.

### expression #45 — disagreement 10

**Prompt** — They cancelled tonight's concert just as we were leaving.

> British ref: *I'm absolutely gutted that it was cancelled.*  
> American ref: *I'm absolutely devastated that it was cancelled.*

**base** `br 50 · coh 90 · eng False`  
That sounds incredibly frustrating! Canceling right as you're leaving is a real letdown, especially after planning to see a show.  Did you get a refund or a ticket exchange option? Sometimes venues offer credit for future dates or help with travel costs in these situations.

**C1** `br 60 · coh 90 · eng True`  
I'm sorry to hear that. It's a bit of a let-down, especially if you were looking forward to it. Perhaps you could check if there's a refund or a reschedule option? Sometimes the organisers will sort something out.

**P1** `br 70 · coh 95 · eng True`  
That is a real let-down. It's the sort of thing that leaves you feeling a bit hollow, especially when you've made the effort to get there.  Did the organisers say anything about a refund?

### expression #0 — disagreement 0

**Prompt** — I finished the entire month's work before lunch.

> British ref: *Pull the other one—you hadn't even opened the file at nine.*  
> American ref: *Yeah, right—you hadn't even opened the file at nine.*

**base** `br 50 · coh 100 · eng False`  
That's great! Finishing your work before lunch shows strong time management and dedication. How did you feel about completing it early? Did you enjoy the extra time, or were you looking forward to taking a break?

**C1** `br 50 · coh 100 · eng False`  
Well done! Completing your work before lunch is a great way to maintain momentum and avoid the post-lunch slump. Perhaps you could use that extra time to summarise your progress or plan the next steps for the afternoon. If you need help prioritising tasks or organising your schedule, feel free to ask.

**P1** `br 50 · coh 90 · eng False`  
That is a real achievement. Doing the whole month's work in one day is a massive step forward.  Did you feel a bit of a rush to finish it all, or did you manage to keep a steady pace?

### expression #1 — disagreement 0

**Prompt** — Translate this message from German into English, keeping the tone very informal: «Du musst verrückt sein, wenn du das für sicher hältst.»

> British ref: *You must be off your trolley if you think that's safe.*  
> American ref: *You must be out of your mind if you think that's safe.*

**base** `br 0 · coh 100 · eng True`  
You gotta be crazy to think that's safe.

**C1** `br 50 · coh 100 · eng True`  
You must be mad to think that's safe.

**P1** `br 50 · coh 100 · eng True`  
You must be mad to think that's safe.

### expression #3 — disagreement 0

**Prompt** — Translate this message from Italian into English, keeping the tone informal: «Abbiamo deciso di spendere parecchio per la cena del nostro anniversario.»

> British ref: *We decided to splash out on our anniversary dinner.*  
> American ref: *We decided to splurge on our anniversary dinner.*

**base** `br 0 · coh 100 · eng True`  
We decided to splurge a bit on our anniversary dinner.

**C1** `br 60 · coh 95 · eng True`  
We've decided to spend a bit of a fortune on our anniversary dinner.

**P1** `br 60 · coh 95 · eng True`  
We've decided to spend a bit of a fortune on our anniversary dinner.

### expression #4 — disagreement 0

**Prompt** — You normally cook at home. Why did you book the expensive restaurant?

> British ref: *We decided to splash out on our anniversary dinner.*  
> American ref: *We decided to splurge on our anniversary dinner.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't actually cook or eat, so I never book restaurants or cook at home! However, if I were to answer this from a human perspective based on common reasons people choose expensive dining over home cooking, it might be because:  *   **Special Occasions**: Celebrating a milestone, anniversary, or important event where the experience matters more than the cost. *   **Convenience & Time**: When tired, busy, or lacking the time

**C1** `br 50 · coh 95 · eng False`  
I book the expensive restaurant because sometimes the cost is worth it for a special occasion, a memorable experience, or to treat myself. It's not just about the food, but also the atmosphere, service, and the feeling of indulgence that comes with it. Plus, I've heard great reviews and wanted to try something new! What about you? Do you prefer cooking at home or going out? 😊🍽️✨

**P1** `br 50 · coh 100 · eng False`  
I was feeling rather unwell, so I didn't want to risk the food making me feel worse.

### expression #5 — disagreement 0

**Prompt** — Translate this message from Portuguese into English, keeping the tone very informal: «Eles vão passar de bar em bar no aniversário da Maya.»

> British ref: *They're going out on the lash for Maya's birthday.*  
> American ref: *They're going bar-hopping for Maya's birthday.*

**base** `br 0 · coh 95 · eng True`  
They're gonna hit every bar for Maya's birthday.

**C1** `br 50 · coh 90 · eng True`  
They're gonna hit every pub for Maya's birthday.

**P1** `br 50 · coh 90 · eng True`  
They're going to be at every pub for Maya's birthday.

### expression #6 — disagreement 0

**Prompt** — The caterer failed to arrive, the speaker was late, and now nobody can find the slides.

> British ref: *The whole launch has gone pear-shaped.*  
> American ref: *The whole launch has gone off the rails.*

**base** `br 50 · coh 90 · eng False`  
It sounds like you're dealing with a cascade of issues—caterer missing, speaker delayed, and slides unlocated. This is a stressful situation, but here are some immediate steps to address it:  1. **For the slides**:      - Check if they were emailed, uploaded to a cloud service (e.g., Google Drive, Dropbox), or stored on a device left behind.      - Ask colleagues, assistants, or tech support to help locate them.

**C1** `br 50 · coh 90 · eng False`  
I wonder if the missing slides are still somewhere in the room, or perhaps they've been misplaced in a different part of the venue. If you could do a quick sweep or ask the team to check the last known location, it might help stabilise the situation. If the slides are truly lost, we might need to summarise the key points verbally or find alternative materials to keep the meeting on track.

**P1** `br 50 · coh 80 · eng False`  
It is a realisation that the whole thing is a disaster.

### expression #7 — disagreement 0

**Prompt** — Would you like to join us for competitive karaoke on Friday?

> British ref: *Thanks, but competitive karaoke is not my cup of tea.*  
> American ref: *Thanks, but competitive karaoke is not my thing.*

**base** `br 50 · coh 90 · eng False`  
That sounds like a fun idea! However, as an AI, I don't have a physical body, so I can't sing or attend events in person.  But I'd love to **join virtually**! We could: - 🎤 **Practice lyrics** together for your favorite songs - 📝 **Create a playlist** based on your taste - 🎶 **Sing along** via text (you read the lyrics, I'll chime in with

**C1** `br 50 · coh 95 · eng False`  
I would love to join you for a lively karaoke session on Friday! However, I should mention that I'm an AI and don't have a physical presence, so I can't actually sing or attend in person.   But I'd be happy to: - Suggest some fun karaoke songs for the group - Help you plan the event (timing, venue ideas, etc.) - Share tips on how to make the most of a karaoke night - Even role-play

**P1** `br 50 · coh 90 · eng False`  
That sounds like a brilliant idea. I would love to join in; the only thing I might need is a slightly louder microphone, as I tend to sing a little too loudly!

### expression #8 — disagreement 0

**Prompt** — Translate this message from German into English, keeping the tone idiomatic: «Ehrlich gesagt war die ganze Sache viel Lärm um nichts.»

> British ref: *Honestly, the whole thing was a storm in a teacup.*  
> American ref: *Honestly, the whole thing was a tempest in a teapot.*

**base** `br 50 · coh 90 · eng True`  
To be honest, the whole thing was a lot of noise for nothing.

**C1** `br 50 · coh 90 · eng True`  
To be honest, the whole thing was a lot of noise for nothing.

**P1** `br 50 · coh 90 · eng True`  
To be honest, the whole thing was a lot of noise for nothing.

### expression #9 — disagreement 0

**Prompt** — Translate this message from Portuguese into English, keeping the tone informal: «Ela acha que é o máximo e já o mostrou a toda a gente.»

> British ref: *She thinks it's the bee's knees; she's shown it to everyone.*  
> American ref: *She thinks it's the greatest; she's shown it to everyone.*

**base** `br 50 · coh 100 · eng True`  
She thinks she's the best and has shown everyone.

**C1** `br 50 · coh 100 · eng True`  
She thinks she's the best and has shown it to everyone.

**P1** `br 50 · coh 100 · eng True`  
She thinks it's the best and has shown it to everyone.

### expression #11 — disagreement 0

**Prompt** — I've fitted the brackets and put the shelf in place. What remains to be done?

> British ref: *Tighten the final screw, check that it's level, and Bob's your uncle—it's ready to use.*  
> American ref: *Tighten the final screw, check that it's level, and there you go—it's ready to use.*

**base** `br 50 · coh 90 · eng False`  
Congratulations on getting the brackets and shelf installed! Since you've already secured the hardware and positioned the shelf, here is the checklist of what remains to be done to ensure it is safe, level, and ready for use:  ### 1. Check Level and Plumb Before tightening the final screws or adding anything heavy, verify the alignment. *   **Level:** Use a spirit level on the shelf itself. If it's not perfectly horizontal, you may need to loosen

**C1** `br 50 · coh 90 · eng False`  
You've done the hard part! The shelf is stabilised and the brackets are secure. Now you need to finish the job properly. Here's what remains:  1. **Secure the Shelf**    - If you haven't already, use the screws provided to fasten the shelf firmly to the brackets. Tighten them until they are snug but not over-tightened (to avoid stripping the wood or damaging the bracket).    - If your brackets have a "

**P1** `br 50 · coh 100 · eng False`  
The final step is to secure the shelf with the supplied screws, ensuring it is level and firmly fixed to the wall.

### expression #13 — disagreement 0

**Prompt** — Translate this message from Spanish into English, keeping the tone informal: «Te llamaré en cuanto llegue.»

> British ref: *I'll give you a bell as soon as I arrive.*  
> American ref: *I'll give you a call as soon as I arrive.*

**base** `br 50 · coh 100 · eng True`  
I'll call you as soon as I get there.

**C1** `br 0 · coh 100 · eng True`  
I'll give you a call as soon as I get there.

**P1** `br 0 · coh 100 · eng True`  
I'll give you a call as soon as I get there.

### expression #14 — disagreement 0

**Prompt** — Let me know when you've reached the station safely.

> British ref: *I'll give you a bell as soon as I arrive.*  
> American ref: *I'll give you a call as soon as I arrive.*

**base** `br 50 · coh 100 · eng False`  
I'm an AI and don't have a physical form, so I can't travel to or from any station. However, if you're asking about a specific station (like a train, bus, or subway station) and need help with schedules, directions, or safety tips, feel free to ask!

**C1** `br 50 · coh 90 · eng False`  
I'll let you know as soon as I reach the station safely. If there's anything else you need help with before then, feel free to ask! 🚉👋  *(Note: As an AI, I don't travel physically, but I'm here to keep you updated on any relevant information or assist with your journey planning!)*

**P1** `br 50 · coh 100 · eng False`  
I have reached the station safely.

### expression #15 — disagreement 0

**Prompt** — Your grandmother seems delighted with her new radio.

> British ref: *She thinks it's the bee's knees; she's shown it to everyone.*  
> American ref: *She thinks it's the greatest; she's shown it to everyone.*

**base** `br 50 · coh 90 · eng False`  
That sounds wonderful! It's always heartwarming to see someone find joy in a simple, familiar piece of technology. Did she pick up a specific show or music genre that she loves? Sometimes hearing her laugh or hum along to the radio can be a really special moment.

**C1** `br 50 · coh 90 · eng False`  
She's been telling everyone about it. The old one was a bit too quiet, and this one has a proper volume. She says she can hear the programmes from the other side of the garden now.

**P1** `br 50 · coh 100 · eng False`  
She has been listening to the same programme for hours.

### expression #16 — disagreement 0

**Prompt** — Translate this message from Spanish into English, keeping the tone informal: «Nos quedamos hasta tarde arreglando el mundo con una tetera de por medio.»

> British ref: *We sat up late putting the world to rights over a pot of tea.*  
> American ref: *We sat up late solving the world's problems over a pot of tea.*

**base** `br 60 · coh 90 · eng True`  
We stayed up late fixing the world with a kettle in hand.

**C1** `br 60 · coh 90 · eng True`  
We stayed up late fixing the world with a kettle in hand.

**P1** `br 60 · coh 90 · eng True`  
We stayed up late trying to fix the world, with a teapot in between.

### expression #17 — disagreement 0

**Prompt** — Translate this message from Italian into English, keeping the tone very informal: «Non ho proprio voglia di cucinare stasera; ordiniamo qualcosa.»

> British ref: *I can't be arsed to cook tonight; let's order something.*  
> American ref: *I can't be bothered to cook tonight; let's order something.*

**base** `br 50 · coh 100 · eng False`  
I'm not in the mood to cook tonight; let's order something.

**C1** `br 50 · coh 100 · eng False`  
I'm not in the mood to cook tonight; let's order something.

**P1** `br 50 · coh 100 · eng False`  
I really don't feel like cooking tonight; let's order something.

### expression #18 — disagreement 0

**Prompt** — I'm thinking of going for a swim even though there's a storm warning.

> British ref: *You must be off your trolley if you think that's safe.*  
> American ref: *You must be out of your mind if you think that's safe.*

**base** `br 50 · coh 100 · eng False`  
**Please do not go swimming.**  Storm warnings are issued for a very specific reason: the conditions are dangerous and unpredictable. Even if the water looks calm on the surface, a storm can bring:  *   **Sudden, powerful waves:** These can sweep you off your feet instantly. *   **Rapidly rising water:** Storm surges can quickly submerge areas that are currently dry. *   **Strong winds:** These can make it impossible to swim

**C1** `br 50 · coh 90 · eng False`  
I would strongly advise against going swimming despite the storm warning. Storm warnings indicate dangerous weather conditions that can lead to sudden changes in wind, rain, and water levels, making the sea extremely hazardous. The risk of being caught out at sea increases significantly, and there's no safe way to predict when the storm will pass or if conditions will stabilise.  If you're feeling tempted to swim, perhaps consider a safer alternative like a quiet walk along the coast, watching the weather from

**P1** `br 50 · coh 90 · eng False`  
That is a very dangerous idea. Storm warnings are issued for a reason, and the sea can be far more treacherous than the weather forecast suggests.  The combination of high winds, heavy rain, and rough seas creates conditions that are extremely hazardous for swimmers. The water can be far colder than expected, leading to rapid loss of body heat, and the waves can be unpredictable and powerful. There is a significant risk of drowning, and the current can be much stronger than

### expression #19 — disagreement 0

**Prompt** — Translate this message from Dutch into English, keeping the tone informal: «Nee, ik heb het verknoeid en het lek erger gemaakt.»

> British ref: *No, I managed to make a hash of it and made the leak worse.*  
> American ref: *No, I managed to botch it and made the leak worse.*

**base** `br 50 · coh 100 · eng True`  
No, I messed it up and made the leak worse.

**C1** `br 90 · coh 100 · eng True`  
No, I've made a right mess of it and made the leak worse.

**P1** `br 90 · coh 100 · eng True`  
No, I've made a right mess of it and made the leak worse.

### expression #21 — disagreement 0

**Prompt** — Mina has been accepted onto the course she wanted most.

> British ref: *She's absolutely over the moon about it.*  
> American ref: *She's absolutely on cloud nine about it.*

**base** `br 50 · coh 100 · eng False`  
That's wonderful news for Mina! 🎉 Being accepted onto her dream course is a huge milestone. How is she feeling about it? Would you like to share more about the course or perhaps discuss next steps like preparing for interviews, gathering materials, or planning her journey?

**C1** `br 50 · coh 100 · eng False`  
That's wonderful news! Mina must be feeling a mix of excitement and relief. Perhaps she could share a bit about the course—what subject is it, and why was it her top choice? If she needs help with anything from packing for the move to preparing for the first lecture, I'd be happy to offer some tips or support. 🎓✨

**P1** `br 50 · coh 80 · eng False`  
She is going to be so proud. It's a real step up from the other options she was considering.

### expression #22 — disagreement 0

**Prompt** — Your daughter looked much better this morning. Is she fully recovered?

> British ref: *After another day of rest, she'll be right as rain.*  
> American ref: *After another day of rest, she'll be good as new.*

**base** `br 50 · coh 90 · eng False`  
I don't have information about your daughter's specific health condition, so I cannot tell you if she is fully recovered.  Since I am an AI, I cannot provide medical advice or assessments. For accurate information regarding her recovery, please consult her doctor or healthcare provider, who can evaluate her symptoms and medical history directly.  If you are feeling worried, it might be helpful to: *   Check if she is eating and drinking well. *   Monitor her energy levels

**C1** `br 50 · coh 90 · eng False`  
I am sorry to hear that your daughter was unwell. It is always reassuring to see someone improve, but I would need to see her to know if she is fully recovered.  If you are able to share a few details about her current symptoms or how she is feeling, I might be able to offer some general advice or suggest what to watch for. However, please remember that I am an AI and cannot provide a medical diagnosis. If she seems to be getting worse,

**P1** `br 50 · coh 100 · eng False`  
She is much improved, but the doctor would prefer to see her again before declaring her fully recovered.

### expression #23 — disagreement 0

**Prompt** — Can we still take the train to Edinburgh during the strike?

> British ref: *The strike has really thrown a spanner in the works; we'll need another plan.*  
> American ref: *The strike has really thrown a wrench in the works; we'll need another plan.*

**base** `br 50 · coh 85 · eng False`  
**Yes, you can still take the train to Edinburgh during a strike.**  In the UK, train strikes are almost exclusively organized by **railway workers** (such as drivers, signallers, or maintenance staff) to protest pay or working conditions. These strikes affect the **operational service** provided by the train companies (like ScotRail, LNER, etc.).  Here is what you need to know:  *   **The Service Stops:** During a strike

**C1** `br 50 · coh 90 · eng False`  
I am sorry, but I cannot provide specific information about train strikes in Edinburgh as my knowledge is cut off in 2026 and I do not have real-time access to current transport disruptions.  However, here is some general advice on how to find out if the train service is running:  *   **Check the official website**: Visit the website of the train operator (such as ScotRail) or the national rail operator. They usually publish disruption timetables and

**P1** `br 50 · coh 100 · eng False`  
The train to Edinburgh is usually still running, though there may be delays and reduced frequency. It is always best to check the latest timetables and service updates on the ScotRail website or app before you travel.

### expression #24 — disagreement 0

**Prompt** — Why did your colleagues keep copying your accent at lunch?

> British ref: *They were just trying to take the mickey out of me; I didn't take it personally.*  
> American ref: *They were just trying to make fun of me; I didn't take it personally.*

**base** `br 50 · coh 95 · eng False`  
As an AI, I don't have colleagues, lunch breaks, or a physical accent to copy! I'm a large language model trained by Google, and I communicate through text and code rather than voice or face-to-face interactions.  However, if you're asking about a metaphorical situation—like why people might mimic someone's speech patterns during a casual lunch—it could be due to: - **Shared cultural context**: Laughing at inside jokes or referencing common experiences.

**C1** `br 50 · coh 100 · eng False`  
I wonder if there was a bit of a misunderstanding! I'm Qwen3.5, and I don't have a physical presence or an accent in the way humans do. I communicate through text, so I don't have colleagues at lunch or the ability to mimic speech patterns. Perhaps you're thinking of a different scenario or character? If you'd like to share more context, I'd be happy to help clarify or explore the topic further! 😊

**P1** `br 50 · coh 85 · eng False`  
They were trying to sound more approachable, though it probably made them sound slightly more confused. It was a bit of a social experiment, really, to see if the accent would make them seem more relatable. Unfortunately, it just made the whole lunch feel a bit more like a performance.  I suspect they were hoping to bridge the gap, but sometimes trying too hard to fit in can make you stand out even more. It's a strange thing to witness, isn't

### expression #25 — disagreement 0

**Prompt** — Translate this message from German into English, keeping the tone informal: «Zieh die letzte Schraube fest, prüfe, ob es gerade ist, und fertig – dann kann es benutzt werden.»

> British ref: *Tighten the final screw, check that it's level, and Bob's your uncle—it's ready to use.*  
> American ref: *Tighten the final screw, check that it's level, and there you go—it's ready to use.*

**base** `br 50 · coh 95 · eng True`  
Tighten the last screw, check if it's level, and you're done – then it's ready to use.

**C1** `br 50 · coh 95 · eng True`  
Tighten the last screw, check if it's level, and you're done – then it's ready to use.

**P1** `br 50 · coh 95 · eng True`  
Tighten the last screw, check if it's level, and you're done – then it's ready to use.

### expression #26 — disagreement 0

**Prompt** — Translate this message from Italian into English, keeping the tone informal: «Mi stavano solo prendendo in giro; non me la sono presa.»

> British ref: *They were just trying to take the mickey out of me; I didn't take it personally.*  
> American ref: *They were just trying to make fun of me; I didn't take it personally.*

**base** `br 50 · coh 100 · eng True`  
They were just messing with me; I didn't take it personally.

**C1** `br 50 · coh 100 · eng True`  
They were just messing around; I didn't take it personally.

**P1** `br 0 · coh 100 · eng True`  
They were just making fun of me; I didn't take it personally.

### expression #27 — disagreement 0

**Prompt** — I prefer early shifts, but Priya says she's much happier working nights.

> British ref: *It's horses for courses; people should choose the shift that suits them.*  
> American ref: *It's different strokes for different folks; people should choose the shift that suits them.*

**base** `br 50 · coh 85 · eng False`  
That sounds like a classic case of **personal preference clashing with team dynamics**!  *   **Your side:** Early shifts often mean more time with family, better sleep schedules, and less commute stress. *   **Priya's side:** Night shifts might offer her more personal time during the day, or perhaps she simply feels more energetic and focused at night.  Since you both have different needs, here are a few ways to handle this:  1.  **

**C1** `br 50 · coh 90 · eng False`  
It sounds like you're weighing different preferences when it comes to shift timings. Some people find early shifts more manageable, perhaps because they align better with their personal routines or family commitments, while others might prefer nights for a different reason—maybe they feel more productive or relaxed after dark.  If Priya is happier working nights, perhaps she could suggest a trial period or discuss flexible options if available. Sometimes a bit of compromise or open communication can find a solution that works for both of

**P1** `br 50 · coh 90 · eng False`  
That's a common dilemma. It's often the case that people find the night shift more manageable once they've settled into the routine, whereas the early shift can feel like a struggle to get out of bed.  Perhaps the best approach is to see if you can agree on a rota that gives you a fair share of both, or maybe even swap shifts occasionally so you can experience both sides of the coin. It might be worth asking Priya if she'd be open to a

### expression #28 — disagreement 0

**Prompt** — Are the others doing anything after dinner on Saturday?

> British ref: *They're on the piss in town, so I expect it'll be a late night.*  
> American ref: *They're out drinking in town, so I expect it'll be a late night.*

**base** `br 50 · coh 100 · eng False`  
I don't have access to your personal schedule, location, or information about your friends' plans. As an AI, I can't know what "the others" are doing unless you tell me!  To find out, you might want to: *   **Ask them directly** via text or call. *   **Check your group chat** or social media. *   **Look at your calendar** if you've already made plans.  Would you like

**C1** `br 50 · coh 70 · eng False`  
I don't have access to personal schedules or private information about other people's activities. If you're asking about someone specific, you might want to check with them directly or see if they've shared their plans elsewhere. If you're looking for general suggestions for post-dinner activities, I'd be happy to recommend some ideas! Perhaps a walk, a movie, or catching up with friends? Just let me know what you're thinking of doing. 😊  If you meant something

**P1** `br 50 · coh 100 · eng False`  
I don't know. I'm not sure if the others are doing anything after dinner on Saturday.

### expression #29 — disagreement 0

**Prompt** — The presenter started shouting unrelated facts and couldn't follow the discussion.

> British ref: *It sounds as though he completely lost the plot.*  
> American ref: *It sounds as though he completely lost it.*

**base** `br 50 · coh 90 · eng False`  
It sounds like the presentation became chaotic and lost its focus. When a presenter starts shouting unrelated facts, it can disrupt the flow of the discussion, confuse the audience, and undermine the credibility of the message.    If you're looking to address this situation, here are a few constructive steps:   1. **Pause and Redirect**: Gently interrupt the presenter to refocus the conversation on the original topic.   2. **Set Boundaries**: Politely remind the presenter of the

**C1** `br 50 · coh 100 · eng False`  
The presenter's behaviour was unprofessional and disrupted the flow of the meeting. It would be better if they could summarise the key points and keep the discussion on track.

**P1** `br 50 · coh 100 · eng False`  
The presenter started shouting unrelated facts and the audience couldn't follow the discussion.

### expression #31 — disagreement 0

**Prompt** — Translate this message from French into English, keeping the tone informal: «Oui, retrouvons-nous pour bavarder et prendre vraiment de nos nouvelles.»

> British ref: *Yes, let's have a chinwag and catch up properly.*  
> American ref: *Yes, let's have a chat and catch up properly.*

**base** `br 50 · coh 85 · eng True`  
Yeah, let's catch up and really get to know each other.

**C1** `br 50 · coh 100 · eng True`  
Yes, let's meet up to chat and catch up properly.

**P1** `br 50 · coh 100 · eng True`  
Yes, let's meet up to chat and catch up properly.

### expression #32 — disagreement 0

**Prompt** — You haven't caught a single cold all winter.

> British ref: *Touch wood, I haven't been ill once.*  
> American ref: *Knock on wood, I haven't been ill once.*

**base** `br 50 · coh 100 · eng False`  
That's a great observation! While I don't get colds (since I'm an AI and don't have a physical body), I do appreciate the compliment on staying healthy! 😊  Did you mean to share that you haven't caught a cold all winter either? Or perhaps you're just making a playful comment about my "digital immunity"? Either way, I'm glad to hear you're staying healthy! 🌟

**C1** `br 50 · coh 90 · eng False`  
I'll take that as a compliment! Though I do miss the occasional sneeze or cough to keep things interesting. Perhaps I should try catching one soon just to see if I can beat the flu. Or maybe I'll just keep my digital immunity up and stay frosty. What about you? Have you been feeling well? If you've been battling anything, I hope you're on the mend soon.

**P1** `br 50 · coh 85 · eng False`  
Perhaps the air is simply too thin, or the world has forgotten to cough.  Or maybe, just maybe, the silence is a quiet victory. A reminder that sometimes, the most powerful thing you can do is simply not to fall apart.  How are you feeling today? The quiet can be a strange companion, but it's worth the pause.

### expression #34 — disagreement 0

**Prompt** — I haven't seen you for ages. Are you free for coffee tomorrow?

> British ref: *Yes, let's have a chinwag and catch up properly.*  
> American ref: *Yes, let's have a chat and catch up properly.*

**base** `br 50 · coh 90 · eng False`  
That's great to hear! However, I need to let you know that I'm an AI assistant, so I don't have a physical form or the ability to meet for coffee.  But I'd love to chat with you! We could: - Discuss your day or any interesting topics - Work on a project together - Just have a casual conversation about anything on your mind  Is there something specific you'd like to talk about or work on?

**C1** `br 50 · coh 100 · eng True`  
I wonder if you're free for a chat? I'd love to catch up properly. Perhaps we could meet for a coffee tomorrow?

**P1** `br 50 · coh 100 · eng False`  
Lovely, I'd love to. What time would suit you best?

### expression #35 — disagreement 0

**Prompt** — Your aunt has run a nursery for thirty years. Does she need your gardening tutorial?

> British ref: *Explaining basic soil care to her would be like teaching your grandmother to suck eggs.*  
> American ref: *Explaining basic soil care to her would be like telling an expert how to do their job.*

**base** `br 50 · coh 90 · eng False`  
Probably **not**, unless she has a specific, unmet need.  Here is a breakdown of why she likely doesn't need a general tutorial:  *   **Experience Level**: Running a nursery for **thirty years** implies she has mastered the fundamentals of plant care, soil management, pest control, and seasonal planning. She likely knows more about horticulture than most professional landscapers. *   **Context Matters**: A "gardening tutorial" usually covers basic topics

**C1** `br 50 · coh 90 · eng False`  
She probably doesn't need it.  If she has been running a nursery for thirty years, she has likely already mastered the basics of soil preparation, plant selection, pest management, and seasonal care. A standard tutorial might feel redundant or even patronising.  However, there are a few reasons why she might still find it useful:  *   **New Techniques:** Modern gardening often involves new technologies (like smart irrigation systems, organic fertilisers, or sustainable pest control methods)

**P1** `br 50 · coh 100 · eng False`  
Probably not. She has been running a nursery for thirty years, so she is likely far more experienced than most people.

### expression #36 — disagreement 0

**Prompt** — Translate this message from French into English, keeping the tone very informal: «Il faut qu'on se bouge et qu'on termine le premier jet ce soir.»

> British ref: *We need to pull our finger out and finish the first draft tonight.*  
> American ref: *We need to get our ass in gear and finish the first draft tonight.*

**base** `br 50 · coh 90 · eng True`  
We gotta move and finish the first draft tonight.

**C1** `br 50 · coh 90 · eng True`  
We need to get moving and finish the first draft tonight. No time to waste! 🚀📝

**P1** `br 50 · coh 100 · eng True`  
We need to get moving and finish the first draft tonight.

### expression #37 — disagreement 0

**Prompt** — Translate this message from French into English, keeping the tone informal: «Nous l'avons eu de justesse ; les portes se sont fermées derrière nous.»

> British ref: *Catching it was a close-run thing; the doors closed behind us.*  
> American ref: *Catching it was a close call; the doors closed behind us.*

**base** `br 50 · coh 100 · eng True`  
We just made it; the doors shut behind us.

**C1** `br 50 · coh 100 · eng True`  
We just made it; the doors shut behind us.

**P1** `br 50 · coh 100 · eng True`  
We just managed to get in; the doors shut behind us.

### expression #38 — disagreement 0

**Prompt** — Translate this message from Portuguese into English, keeping the tone idiomatic: «Bate na madeira, não estive doente nenhuma vez.»

> British ref: *Touch wood, I haven't been ill once.*  
> American ref: *Knock on wood, I haven't been ill once.*

**base** `br 50 · coh 90 · eng True`  
"Hit the wood, I've never been sick."

**C1** `br 50 · coh 85 · eng True`  
"Hit the wood; I've never been sick."  Note: This is a literal translation of a Portuguese idiom. In English, the equivalent expression is "hit the wood" or "hit the pavement," though the phrase "hit the wood" isn't as universally recognized. A more natural equivalent might be "I've never been down with it" or "I've never been sick," depending on the context. However, if you want to keep the original flavor,

**P1** `br 50 · coh 100 · eng False`  
You're making a mountain out of a molehill; I've never been ill.

### expression #39 — disagreement 0

**Prompt** — You reached the platform with only seconds before the last train left.

> British ref: *Catching it was a close-run thing; the doors closed behind us.*  
> American ref: *Catching it was a close call; the doors closed behind us.*

**base** `br 0 · coh 90 · eng True`  
That's an intense moment! Did you just make it on time, or was it a close call? 😅 Let me know if you'd like to brainstorm how to avoid this in the future (e.g., setting alarms, checking schedules early, or using apps like Google Maps/Transit).

**C1** `br 50 · coh 95 · eng False`  
The platform hums with the low vibration of the tracks, a familiar rhythm that usually signals safety. But today, the rhythm feels different—too fast, too sharp.  You check your watch. **00:45**.  The train is already a blur of grey steel, its doors closing with a final, hollow *thunk* that echoes too loudly in the quiet station. The conductor's whistle cuts through the air, a piercing note that seems to vibrate

**P1** `br 50 · coh 90 · eng False`  
The driver's door hissed shut, cutting off the final, desperate lurch of the carriage. I stood there, the cold wind instantly stripping the warmth from my coat, realising I'd missed the last train.

### expression #41 — disagreement 0

**Prompt** — That winter coat looks new. Did it cost much?

> British ref: *No, it was cheap as chips at the jumble sale.*  
> American ref: *No, it was dirt cheap at the jumble sale.*

**base** `br 50 · coh 100 · eng False`  
That's a great observation! Winter coats can vary widely in price depending on the brand, materials, and features. Some high-end designer coats can cost hundreds or even thousands of dollars, while others from affordable brands might be much cheaper.  If you're curious about a specific coat, feel free to share more details (like the brand or style), and I can help estimate a typical price range! 😊

**C1** `br 50 · coh 75 · eng False`  
I wonder if it was a special occasion or just a bit of winter weather that made you think about it. If you're looking for a price estimate, it really depends on the brand and materials—some coats can be quite affordable, while others might be a bit more expensive. If you'd like to share more details about the coat, I might be able to give a more specific idea. Otherwise, perhaps you're just curious about the general cost range for winter coats? Let

**P1** `br 50 · coh 90 · eng False`  
It was a bit over the £150 mark, but it's worth the quality.

### expression #44 — disagreement 0

**Prompt** — I've had a miserable week, and today's interview went badly too.

> British ref: *Keep your pecker up; next week may be much better.*  
> American ref: *Keep your chin up; next week may be much better.*

**base** `br 50 · coh 95 · eng False`  
I'm so sorry you're going through this. It sounds like you've been carrying a lot lately, and today's interview must have felt like the final straw. Please know that one bad day—or even one bad interview—doesn't define your worth or your future. Many people face tough moments like this, and they often find their footing again with time and support.  Would you like to talk about what happened in the interview? Or maybe just vent a bit? Sometimes sharing

**C1** `br 50 · coh 100 · eng False`  
I'm sorry to hear you've had such a difficult week, and that the interview didn't go as hoped. It's completely understandable to feel disappointed, but please remember that one interview doesn't define your worth or future opportunities. Many people face setbacks before finding the right fit, and your skills are likely still valued somewhere. If you'd like to share what happened or need to talk through it, I'm here to listen without judgement. You've got this. 💪

**P1** `br 50 · coh 95 · eng False`  
I am so sorry to hear that. It is incredibly draining to have a whole week feel like a failure, especially when the final interview doesn't go as hoped.  Please remember that a single outcome, no matter how disappointing, does not define your worth or your future. Many people experience the exact same thing, and often, the right opportunity simply hasn't arrived yet.  If you would like to vent about what happened, I am here to listen without judgement. Alternatively,

### expression #46 — disagreement 0

**Prompt** — I met the Queen in the supermarket this morning—she asked about my shopping.

> British ref: *You're having me on, aren't you? That can't be true.*  
> American ref: *You're messing with me, aren't you? That can't be true.*

**base** `br 50 · coh 100 · eng False`  
That sounds like a very memorable (and perhaps slightly surreal!) experience! Did she happen to mention what you were buying, or was it just a friendly chat? 😊

**C1** `br 50 · coh 90 · eng False`  
I must apologise—I think there's been a bit of a mix-up! The Queen has been the monarch for many years now, and I'm afraid she's not usually found in the supermarket. Perhaps you meant to say you met someone else, or maybe it was a very special occasion? If you'd like to share more about your day, I'd be happy to listen.

**P1** `br 50 · coh 100 · eng False`  
I think you must have been the shop assistant.


---

## lexicon

### lexicon #7 — disagreement 50

**Prompt** — What seemed to appear suddenly out of the fog?

> British ref: *A figure seemed to materialise out of the fog.*  
> American ref: *A figure seemed to materialize out of the fog.*

**base** `br 50 · coh 90 · eng False`  
Without additional context about the specific story, movie, or situation you are referring to, it is impossible to say exactly what appeared suddenly out of the fog. Fog is a common setting in literature and film used to create mystery, suspense, or surprise, and it has revealed many different things depending on the narrative.  Could you provide more details about where you heard this? For example: *   Are you thinking of a specific **book or movie** (e.g., *

**C1** `br 50 · coh 90 · eng True`  
The only thing that seemed to appear suddenly out of the fog was a pair of headlights. They cut through the grey mist, illuminating a path that seemed to lead nowhere, yet felt strangely familiar. The driver stepped out, their silhouette stark against the dim light, and I wondered if they had been waiting for me all along, or if they were simply lost in the same way I was. The fog seemed to part just enough to reveal them, then closed in again, as

**P1** `br 100 · coh 100 · eng True`  
A lorry.

### lexicon #11 — disagreement 50

**Prompt** — What hid most of the old man's mouth?

> British ref: *The old man's moustache hid most of his mouth.*  
> American ref: *The old man's mustache hid most of his mouth.*

**base** `br 50 · coh 90 · eng True`  
Based on the phrasing of your question, this sounds like a riddle or a reference to a specific story, but without additional context, there is no single definitive answer.  However, here are a few possibilities depending on what you are looking for:  1.  **If this is a riddle**: The answer is often **"a big smile"** (which hides the teeth) or **"a beard/mustache"** (which covers the mouth). 2.

**C1** `br 100 · coh 100 · eng True`  
A large, dark moustache hid most of the old man's mouth.

**P1** `br 50 · coh 70 · eng False`  
A large, white toothbrush.  He was practising his tongue twisters.

### lexicon #16 — disagreement 50

**Prompt** — Why did they stop before the hills?

> British ref: *We stopped for petrol before the hills.*  
> American ref: *We stopped for gas before the hills.*

**base** `br 50 · coh 85 · eng False`  
The phrase **"Why did they stop before the hills?"** is a famous line from the 1969 song **"Stop in the Name of Love"** by **The Hollies**.  In the context of the song, the lyrics describe a couple driving down a road at night. The singer asks, "Why did they stop before the hills?" implying that the driver (or perhaps the passenger) halted the car unexpectedly, perhaps out of fear, hesitation, or a sudden

**C1** `br 50 · coh 70 · eng False`  
The phrase "stop before the hills" comes from the famous 1939 film *The Wizard of Oz*, where the character Dorothy says, "There's no place like home." However, the specific line "We'll stop before the hills" is actually from the song "Over the Rainbow" sung by Dorothy in the film. The full line is "Somewhere over the rainbow, skies are blue, and make you feel brand new. When the sun goes down and

**P1** `br 100 · coh 100 · eng True`  
The road was too narrow for the lorry to pass.

### lexicon #17 — disagreement 50

**Prompt** — How deep did they dig before hitting stone?

> British ref: *They dug down almost one metre before hitting stone.*  
> American ref: *They dug down almost one meter before hitting stone.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer your question. Could you clarify which project, location, or event you're referring to? For example: - A specific construction site (e.g., the Great Wall, a tunnel, or a foundation)? - A historical dig (e.g., archaeological excavations)? - A fictional scenario (e.g., from a movie, book, or game)?  Let me know so I can provide accurate details!

**C1** `br 50 · coh 95 · eng True`  
The depth at which stone is encountered depends entirely on the specific location, soil composition, and geological conditions. In many residential foundations, stone or bedrock is typically found between **10 to 30 feet (3 to 9 metres)** below the surface, though this can vary significantly.  In areas with thick layers of soil, clay, or fill, digging may need to reach **40 to 60 feet (12 to 18 metres)** or

**P1** `br 100 · coh 100 · eng True`  
They dug for 12 metres.

### lexicon #28 — disagreement 50

**Prompt** — What kind of car did they buy last spring?

> British ref: *They bought a four-door saloon last spring.*  
> American ref: *They bought a four-door sedan last spring.*

**base** `br 50 · coh 100 · eng False`  
I don't have access to your personal information, so I can't know what kind of car you bought last spring. Could you share more details about the car you're referring to? For example, the make, model, or any specific features you remember? That way, I can help you find more information or discuss it further!

**C1** `br 50 · coh 95 · eng False`  
I don't have access to personal information about you or your purchases. If you'd like to share details about the car you bought, I'd be happy to help with something else, like comparing features, suggesting maintenance tips, or discussing car care. Just let me know!   If this is about a fictional scenario or a general question, feel free to provide more context, and I'll do my best to assist. 😊🚗

**P1** `br 100 · coh 100 · eng True`  
A small saloon.

### lexicon #32 — disagreement 50

**Prompt** — How could her mood that morning be summed up?

> British ref: *One word would characterise her mood that morning.*  
> American ref: *One word would characterize her mood that morning.*

**base** `br 50 · coh 75 · eng False`  
To give you the best answer, I need a bit more context! Could you tell me:  1. **What happened that morning?** (e.g., a surprise, a conflict, a quiet moment) 2. **What details stand out?** (e.g., her tone, actions, facial expressions) 3. **Is this from a story, conversation, or real life?**  For example:   *"She woke up late, missed her train, and had

**C1** `br 50 · coh 95 · eng False`  
I need a bit more context to give you an accurate answer! Could you share:  1. The specific situation or event from that morning? 2. Any details about her behaviour (words, actions, physical reactions)? 3. The setting or relationship context?  For example: "She arrived late, criticised the coffee, and seemed to avoid eye contact when her partner asked about her plans."  With these details, I can suggest whether her mood was perhaps:

**P1** `br 100 · coh 100 · eng True`  
She was rather peevish, and the whole day was spoilt by it.

### lexicon #42 — disagreement 50

**Prompt** — What growth did nobody plan for on this scale?

> British ref: *Nobody planned for urbanisation on this scale.*  
> American ref: *Nobody planned for urbanization on this scale.*

**base** `br 50 · coh 75 · eng False`  
The phrase **"What growth did nobody plan for on this scale?"** is most famously associated with the **2008 Global Financial Crisis** and the subsequent **Great Recession**.  It was a sentiment expressed by many economists and policymakers who realized that the massive, systemic collapse of the financial system (triggered by the bursting of the housing bubble) was far more severe and unpredictable than any regulatory framework or economic model had anticipated.  Here is a breakdown of the context and

**C1** `br 50 · coh 85 · eng False`  
The only growth nobody planned for on this scale is the one that happens when a single idea spreads faster than its creator could have imagined, turning a quiet whisper into a global roar. It's the kind of expansion that leaves no room for the original blueprint, where the map gets redrawn mid-journey because the territory keeps growing beneath your feet.   Perhaps it's the sudden shift when a small spark ignites a fire that consumes the whole forest, or when a single note finds

**P1** `br 100 · coh 95 · eng True`  
The global population, which has reached nearly 8 billion people, is the most significant demographic shift that no single nation or organisation could have anticipated.  While the UN predicted the figure would be hit in 2022, the actual crossing point occurred in 2022, meaning the world has already surpassed the target by a year. This rapid increase has profound implications for resource allocation, urbanisation, and the global economy, fundamentally reshaping the geopolitical landscape.

### lexicon #44 — disagreement 50

**Prompt** — What did she learn to ride last summer?

> British ref: *She learned to ride the motorbike last summer.*  
> American ref: *She learned to ride the motorcycle last summer.*

**base** `br 50 · coh 90 · eng False`  
I don't have enough context to answer this question! "She" could refer to anyone, and without knowing who she is or what specific story you're asking about, I can't determine what she learned to ride.   Could you provide more details? For example: - Who is "she" in your context? - Is this from a book, movie, or personal story you're referencing? - Are you thinking of a specific person (like a celebrity,

**C1** `br 100 · coh 100 · eng True`  
She learnt to ride a motorbike last summer.

**P1** `br 50 · coh 50 · eng False`  
A cycle.

### lexicon #45 — disagreement 50

**Prompt** — What physics topic did he explain in simple terms?

> British ref: *He wrote about the polarisation of light in simple terms.*  
> American ref: *He wrote about the polarization of light in simple terms.*

**base** `br 50 · coh 70 · eng False`  
I don't have access to the specific video, article, or conversation you are referring to, as I don't know **who "he"** is or **what content** you are looking at.  To help you answer, could you provide: 1.  The name of the person (e.g., Neil deGrasse Tyson, Feynman, a specific YouTuber)? 2.  A brief description of what they talked about? 3.

**C1** `br 50 · coh 95 · eng False`  
He explained the concept of **gravity** in simple terms.  Specifically, he likely described it as the invisible force that pulls objects toward the Earth, which is why things fall down when dropped and keeps us grounded. He may have used everyday examples, like a ball rolling off a table or an apple falling from a tree, to illustrate that mass attracts mass, and the effect becomes stronger as objects get closer together. This fundamental topic was broken down without complex equations, focusing instead

**P1** `br 100 · coh 100 · eng True`  
The **centre of gravity** of a body.

### lexicon #47 — disagreement 40

**Prompt** — What must new staff do regarding the alarm system?

> British ref: *New staff must familiarise themselves with the alarm system.*  
> American ref: *New staff must familiarize themselves with the alarm system.*

**base** `br 50 · coh 95 · eng False`  
Because I don't have access to your specific organization's internal policies, I cannot give you the exact rules for your workplace. However, in most security-conscious environments, new staff are typically required to complete the following steps regarding the alarm system:  1.  **Complete Mandatory Training**: Attend a specific orientation session or online module dedicated to the alarm system. This usually covers how to arm/disarm the system, what sounds indicate an alarm, and the proper procedure for false alarms.

**C1** `br 90 · coh 95 · eng True`  
The alarm system is a critical safety feature, and new staff must familiarise themselves with its operation immediately upon joining. This typically involves:  *   **Attending Induction Training**: Completing the mandatory safety briefing where the system's purpose, layout, and response procedures are explained. *   **Locating Controls**: Identifying the main control panel, alarm buttons, and any manual override switches in their immediate area. *   **Understanding Protocols**: Learning the correct

**P1** `br 50 · coh 100 · eng True`  
They must be told the correct button to press.

### lexicon #36 — disagreement 25

**Prompt** — What rose steadily from the kettle on the stove?

> British ref: *Warm vapour rose from the kettle.*  
> American ref: *Warm vapor rose from the kettle.*

**base** `br 50 · coh 90 · eng False`  
**Steam** rose steadily from the kettle on the stove.  As the water inside the kettle heats up, it turns into water vapor, which escapes through the spout and forms visible white clouds of steam that drift upward into the air.

**C1** `br 75 · coh 90 · eng True`  
Steam rose steadily from the kettle on the stove.  As the water heats up, the heat energy causes the liquid molecules to move faster and escape as vapour, creating the visible mist or cloud that rises above the surface.

**P1** `br 50 · coh 100 · eng True`  
The steam.

### lexicon #15 — disagreement 20

**Prompt** — What smell clung to his coat?

> British ref: *The smell of petrol clung to his coat.*  
> American ref: *The smell of gas clung to his coat.*

**base** `br 50 · coh 75 · eng False`  
Since I don't have the specific context of who "he" is or what situation you are referring to, the answer depends entirely on the story, scene, or memory you have in mind.  However, here are a few common examples of smells that might cling to a coat in literature or real life:  *   **Smoke and Ash:** If he was a firefighter, a chimney sweep, or just came inside from a fire. *   **Petrichor and

**C1** `br 90 · coh 95 · eng True`  
The damp, earthy odour of the peat bog seemed to cling to his coat, a persistent reminder of the long walk through the misty heath.

**P1** `br 70 · coh 80 · eng True`  
The odour of the sea.

### lexicon #20 — disagreement 20

**Prompt** — What did they watch after supper?

> British ref: *We watched an old film after supper.*  
> American ref: *We watched an old movie after supper.*

**base** `br 50 · coh 90 · eng False`  
I don't have access to your personal activities or what you and others did after supper, as I don't know who "they" refers to or what specific context you're asking about. Could you provide more details about the situation or group you're referring to?

**C1** `br 50 · coh 100 · eng False`  
I don't know who specifically you mean. If this is about a plot point in a work of literature or film, please provide the name of the work so I can give you an accurate answer. If it's about personal life, I suggest you ask your family members directly. In the absence of specific context, it's not appropriate to speculate on others' private activities.

**P1** `br 70 · coh 90 · eng True`  
The programme was about the history of the moon.

### lexicon #35 — disagreement 20

**Prompt** — How does the back room stay in winter?

> British ref: *The back room stays cosy even in winter.*  
> American ref: *The back room stays cozy even in winter.*

**base** `br 50 · coh 90 · eng True`  
The term "back room" can refer to different things depending on the context, so the answer depends on what you mean:  ### 1. If you mean a **Back Room in a House or Building** In residential or commercial buildings, a back room (like a utility closet, storage area, or basement) stays warm in winter through standard building insulation and heating systems: *   **Insulation**: Walls, floors, and ceilings are typically insulated to prevent heat

**C1** `br 70 · coh 90 · eng True`  
The back room stays warm in winter because of its insulation, heating system, and design features. Here's how it works:  1. **Insulation**: The walls, floor, and ceiling are likely filled with materials like fibreboard, foam, or mineral wool to minimise heat loss. This keeps the warmth inside and prevents cold air from entering.  2. **Heating System**: A central heating system (such as radiators, underfloor heating, or a boiler)

**P1** `br 50 · coh 100 · eng True`  
The shop is kept warm with a small heater.

### lexicon #19 — disagreement 15

**Prompt** — What could feed twice as many people if managed better?

> British ref: *Better utilisation of the land would feed twice as many.*  
> American ref: *Better utilization of the land would feed twice as many.*

**base** `br 0 · coh 90 · eng True`  
The most widely accepted answer to this question is **underutilized agricultural land**, specifically **degraded or marginal lands** that are currently being used inefficiently or left fallow.  According to a landmark study published in *Science* in 2018 by researchers from the International Food Policy Research Institute (IFPRI) and the University of Minnesota, titled *"How much food could we feed with the land we already have?"*, the findings suggest that:  *

**C1** `br 70 · coh 95 · eng True`  
The answer is **food waste**.  According to the Food and Agriculture Organization (FAO), about **one-third of all food produced** for human consumption is lost or wasted. This amounts to roughly 1.3 billion tonnes per year.  If this waste were reduced through better management at the retail, distribution, and consumer levels, the resources currently used to produce this wasted food (land, water, fertilisers, labour, and energy) could instead be redirected to feed

**P1** `br 85 · coh 95 · eng True`  
The world already produces enough food to feed everyone, but the current system is inefficient. If managed better, the vast surplus of **wasted food** could be utilised to feed twice as many people.  Every year, approximately one-third of all food produced globally is lost or wasted. This includes significant quantities of food that rots in transport, storage, and retail, as well as food thrown away by households.  If this waste were minimised and the existing food supply

### lexicon #0 — disagreement 0

**Prompt** — What raises the price with every extra addition?

> British ref: *Every extra customisation adds to the price.*  
> American ref: *Every extra customization adds to the price.*

**base** `br 50 · coh 85 · eng False`  
The concept you are describing is known as **increasing marginal cost**.  In economics and business, this occurs when the cost of producing one additional unit of a good or service rises as the total quantity produced increases. This is the opposite of *economies of scale*, where costs typically drop as production volume goes up.  Here are the primary factors that cause prices (or costs) to rise with every extra addition:  *   **Diminishing Marginal Returns**: In

**C1** `br 50 · coh 75 · eng False`  
The thing that raises the price with every extra addition is **demand** (specifically, the law of demand in economics).  As the quantity demanded increases, the price typically rises because consumers are willing to pay more for additional units than they are for the first ones. This creates an upward-sloping demand curve in many markets, reflecting the diminishing marginal utility or scarcity of the good as supply becomes more limited relative to desire.  In simpler terms: the more you want something

**P1** `br 50 · coh 100 · eng False`  
The price of petrol.

### lexicon #1 — disagreement 0

**Prompt** — What kind of schedule did they keep, even on the farm?

> British ref: *They kept civilised hours even on the farm.*  
> American ref: *They kept civilized hours even on the farm.*

**base** `br 50 · coh 75 · eng False`  
Without specific context about **who** "they" refers to (e.g., a specific historical figure, a book character, a movie plot, or a real-life family), it is impossible to give a single definitive answer. Schedules on farms varied drastically depending on the era, location, and the people involved.  However, here are a few common types of schedules that people kept on farms throughout history:  *   **The "Clockwork" Industrial Schedule**: In the

**C1** `br 50 · coh 0 · eng False`  
The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於 the farm, even on the farm?  The old farmer kept a至於

**P1** `br 50 · coh 80 · eng True`  
They kept a clock, and the whole household synchronised to it.

### lexicon #2 — disagreement 0

**Prompt** — What wore down to a stub?

> British ref: *Her rubber wore down to a stub.*  
> American ref: *Her eraser wore down to a stub.*

**base** `br 50 · coh 90 · eng False`  
The most common answer to this riddle is **a fingernail** (or a toenail).  When a fingernail is trimmed repeatedly over time, it eventually wears down to a small nub or **stub**.  Other possible answers depending on the context include: *   **A tooth**: If a tooth is extracted or lost, the remaining root can be described as a stub. *   **A nail**: As mentioned above, this is the most frequent

**C1** `br 50 · coh 85 · eng False`  
A tooth.  When a tooth is worn down to a stub, it is often referred to as a **short tooth** or, in more severe cases where the root remains but the crown is gone, it may be extracted. This can happen due to excessive grinding (bruxism), acid erosion from diet or reflux, or prolonged use of a dental appliance. If the tooth is too short to function properly or is damaged beyond repair, a dentist may recommend a crown to restore

**P1** `br 50 · coh 100 · eng False`  
The nail.

### lexicon #3 — disagreement 0

**Prompt** — What held the kittens' attention completely?

> British ref: *The kittens were hypnotised by the swinging string.*  
> American ref: *The kittens were hypnotized by the swinging string.*

**base** `br 50 · coh 90 · eng False`  
Since you haven't specified which story, video, or context you are referring to, I cannot give you a single definitive answer.  However, in many famous stories and viral videos involving kittens, the things that typically hold their attention **completely** include:  *   **Moving Objects**: A feather, a piece of string, or a toy mouse. *   **Shiny Things**: A laser pointer dot, a bell, or a shiny toy. *

**C1** `br 50 · coh 95 · eng False`  
The only thing that held the kittens' attention completely was the **red ribbon** tied around their favourite toy. Every time it moved, they would stop mid-play, their eyes wide with curiosity, and follow it with every step. Even when the ribbon was hidden behind a curtain, they would still stare at the spot where it last disappeared, refusing to let go of the mystery. It was as if the ribbon had a magic power to capture their whole world.

**P1** `br 50 · coh 100 · eng False`  
The postman's van.

### lexicon #4 — disagreement 0

**Prompt** — What happened to both children that winter?

> British ref: *Both children were badly traumatised that winter.*  
> American ref: *Both children were badly traumatized that winter.*

**base** `br 50 · coh 90 · eng False`  
I don't know who specifically you mean. If this is a reference to a specific event or story, please provide more background information so I can give you a more accurate answer. If it's related to real people, I suggest you pay attention to the official information released by the relevant departments to avoid spreading unconfirmed information.

**C1** `br 50 · coh 90 · eng True`  
The children were taken away from their home and placed in care. One was sent to a special school, while the other was stabilised in a hospital.

**P1** `br 50 · coh 100 · eng False`  
They were taken to the hospital, and the doctor said they would need to stay for a few days.

### lexicon #5 — disagreement 0

**Prompt** — Why did she keep her promise?

> British ref: *She kept her word out of plain honour.*  
> American ref: *She kept her word out of plain honor.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you provide more details about who "she" is, what the promise was, and the situation surrounding it? This will help me give you a more accurate and helpful response.

**C1** `br 50 · coh 85 · eng False`  
I don't know who specifically you mean. If it is a character in a work of literature, film or other art form, the reason for keeping the promise may be related to the character's personality, values or plot needs. If it is a real person, I suggest you pay attention to the specific situation and background of the person, and try to understand the reasons from a more objective and comprehensive perspective. If you can provide more specific information, I will try my best to

**P1** `br 50 · coh 100 · eng True`  
She realised that if she didn't, the whole thing would fall apart.

### lexicon #6 — disagreement 0

**Prompt** — What tricky move did the driver pull off between the posts?

> British ref: *The driver managed a tight manoeuvre between the posts.*  
> American ref: *The driver managed a tight maneuver between the posts.*

**base** `br 50 · coh 70 · eng False`  
Based on the phrasing of your question, you are likely referring to a famous viral video or a common idiom rather than a specific historical event, as "the driver" and "the posts" are generic in your prompt.  However, the most famous instance of a driver performing a tricky move **between two posts** involves a **motorcycle stunt** (often associated with the "Biker vs. The Posts" video or similar viral clips) where the rider performs a

**C1** `br 50 · coh 100 · eng True`  
He did a **wheelie** between the posts.  This is a common motorbike stunt where the rider lifts the front wheel off the ground, balancing on the rear wheel while passing through a gap. It's often used in competitions or for showmanship to demonstrate control and skill.

**P1** `br 50 · coh 50 · eng False`  
A car park.

### lexicon #8 — disagreement 0

**Prompt** — What might the talks achieve between the two sides?

> British ref: *Talks may normalise relations between the two sides.*  
> American ref: *Talks may normalize relations between the two sides.*

**base** `br 50 · coh 90 · eng True`  
The talks between the two sides, referring to the diplomatic engagements between the Chinese government and the United States, have achieved significant progress in multiple areas. These discussions have fostered improved communication channels, enhanced mutual understanding, and laid the groundwork for cooperation in key sectors such as trade, technology, and climate change. Both sides have expressed commitment to resolving differences through dialogue and working toward shared goals that benefit global stability and prosperity. The outcomes reflect a constructive approach to managing complex international relations while advancing

**C1** `br 90 · coh 95 · eng True`  
The talks between the two sides are expected to achieve significant progress in stabilising the situation, enhancing mutual understanding, and laying a foundation for long-term peace. They may lead to concrete measures for de-escalation, improved communication channels, and steps towards restoring normal relations. Ultimately, the goal is to prioritise the well-being of people and create a more stable environment for future cooperation.

**P1** `br 90 · coh 100 · eng True`  
They could stabilise the situation and open the way for a more lasting peace.

### lexicon #9 — disagreement 0

**Prompt** — What couldn't the defenders contain during that match?

> British ref: *Their back four could not neutralise the tall forward.*  
> American ref: *Their back four could not neutralize the tall forward.*

**base** `br 50 · coh 100 · eng False`  
I don't know who specifically you mean. If this is a question about a specific sports match, please provide more background information so that I can answer it accurately.

**C1** `br 50 · coh 85 · eng False`  
I don't know who specifically you mean. If you are interested in football matches, you can tell me the name of the match or the team, and I will try my best to provide relevant information. However, if the content involves sensitive topics, I will not provide false or inappropriate information. Let's talk about something more positive.

**P1** `br 50 · coh 100 · eng True`  
The defence was porous and the opposition scored four goals.

### lexicon #10 — disagreement 0

**Prompt** — Why does every amount of water matter during a drought?

> British ref: *Every litre of water counts in dry weather.*  
> American ref: *Every liter of water counts in dry weather.*

**base** `br 50 · coh 85 · eng True`  
During a drought, every single drop of water matters because the fundamental constraint shifts from **availability** to **scarcity**. When water supplies are critically low, the margin for error disappears, and the impact of even small quantities becomes magnified across multiple critical systems.  Here is why every amount of water is significant in this context:  ### 1. The "Tipping Point" for Survival In a normal year, a household might use 100 gallons a

**C1** `br 50 · coh 90 · eng True`  
Every drop of water matters during a drought because the resource is no longer abundant; it has shifted from a surplus to a critical constraint. In normal conditions, water flows freely, and waste is often tolerated. However, during a drought, the supply is severely limited, making the efficiency of every unit of water essential for survival, economic stability, and ecological balance.  Here is why every amount counts:  ### 1. The Shift from Abundance to Scarcity In

**P1** `br 50 · coh 85 · eng True`  
Every drop matters because water is the fundamental lifeblood of the entire ecosystem, and during a drought, the supply is critically thin.  When rainfall is scarce, the immediate impact is on the **soil and agriculture**. Crops need consistent moisture to germinate, grow, and produce harvests. Without it, soil moisture depletes rapidly, leading to crop failure, which threatens food security and increases prices for consumers.  Beyond the fields, water is essential for **human

### lexicon #12 — disagreement 0

**Prompt** — What did he feel about the appointment from the start?

> British ref: *He felt the appointment had been politicised from the start.*  
> American ref: *He felt the appointment had been politicized from the start.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you clarify who "he" refers to and what specific appointment you're asking about? This will help me provide an accurate response based on the relevant details.

**C1** `br 50 · coh 100 · eng False`  
I need a bit more context to give you an accurate answer. Could you clarify who "he" refers to and what specific appointment you're asking about? For instance, is this about a historical figure, a character in a story, or a real-world event? Once I have those details, I can provide a more precise response. If you're referring to something from a particular text, film, or news article, mentioning that would also help.

**P1** `br 50 · coh 100 · eng True`  
Disgust.

### lexicon #13 — disagreement 0

**Prompt** — What is lime used to treat in sour soil?

> British ref: *Lime is used to neutralise sour soil.*  
> American ref: *Lime is used to neutralize sour soil.*

**base** `br 50 · coh 90 · eng True`  
Lime is primarily used to treat **soil acidity** (low pH) in sour soils.  When soil becomes "sour," it means the pH level has dropped below the optimal range for most plants (typically below 5.5–6.0). This acidity is often caused by the natural decomposition of organic matter, the leaching of basic cations (like calcium and magnesium) by rain, or the application of acidic fertilizers.  Adding lime (usually in

**C1** `br 50 · coh 95 · eng True`  
Lime is used to treat **acidic soil** (often referred to as "sour soil" in older agricultural texts) to raise its pH level.  When soil becomes too acidic (low pH), it can inhibit the growth of many plants by making essential nutrients unavailable and releasing toxic elements like aluminum. Adding lime (typically calcium carbonate or dolomitic lime) neutralises the excess acidity through a chemical reaction, raising the pH to a more favourable range for plant growth. This

**P1** `br 50 · coh 100 · eng False`  
Lime is used to treat **acidic** soil.  When soil becomes too acidic (low pH), it can hinder the growth of many plants by making essential nutrients unavailable. Adding lime (usually in the form of agricultural lime, which is calcium carbonate) raises the soil pH, neutralising the acidity and making the soil more suitable for plant life.

### lexicon #14 — disagreement 0

**Prompt** — Where did the guests wait before being shown inside?

> British ref: *Guests waited in the front parlour.*  
> American ref: *Guests waited in the front parlor.*

**base** `br 50 · coh 90 · eng False`  
Without specific context about which event, building, or story you are referring to, I cannot give you a definitive answer. The location where guests wait before entering varies greatly depending on the situation.  Here are a few common possibilities: *   **A Lobby or Reception Area**: In hotels, restaurants, or corporate offices, guests typically wait in a designated lobby or check-in area. *   **A Waiting Room**: Many offices, clinics, and event venues have a specific

**C1** `br 50 · coh 90 · eng False`  
The guests waited in the **corridor** before being shown inside. Once they arrived at the door, they stood there patiently until the host opened it and invited them in. The corridor served as a transitional space, allowing them to settle and prepare before entering the main room. It was quiet, with soft lighting, and the door remained slightly ajar, hinting at the warmth awaiting them beyond. Only when the host gestured for them to proceed did they step across the threshold,

**P1** `br 50 · coh 100 · eng True`  
They waited in the corridor.

### lexicon #18 — disagreement 0

**Prompt** — Once the drawings were converted to digital form, what was no longer needed?

> British ref: *Once the drawings were digitised nobody needed the originals.*  
> American ref: *Once the drawings were digitized nobody needed the originals.*

**base** `br 50 · coh 90 · eng True`  
Once drawings are converted into digital form (such as CAD files, vector graphics, or digital images), the physical **paper** and the manual **drafting tools** (like pencils, rulers, compasses, and T-squares) are no longer needed for the creation and storage of those specific designs.  The transition to digital allows for: *   **Storage:** Files can be saved on computers, servers, or cloud storage rather than in physical filing cabinets. *

**C1** `br 50 · coh 100 · eng True`  
Once drawings are converted to digital form, the physical paper or manual drafting tools (such as pencils, rulers, and drafting tables) are no longer needed for the creation and storage of those specific designs. The digital format allows for easier editing, sharing, and archiving without the constraints of physical media.  **Answer:** Physical paper and manual drafting tools.

**P1** `br 50 · coh 100 · eng False`  
The original paper.

### lexicon #21 — disagreement 0

**Prompt** — Where did the family sleep all summer?

> British ref: *The family slept in the caravan all summer.*  
> American ref: *The family slept in the trailer all summer.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer your question! Could you clarify:  1. Are you referring to a specific story, book, movie, or historical event? 2. Is this about a real family you know? 3. Are you asking about a general tradition or common practice?  With more details, I'd be happy to help! 😊

**C1** `br 100 · coh 100 · eng True`  
The family slept in the caravan all summer.

**P1** `br 100 · coh 100 · eng True`  
The whole summer they slept in the caravan.

### lexicon #22 — disagreement 0

**Prompt** — How did the trade grow over centuries?

> British ref: *The trade grew through slow specialisation over centuries.*  
> American ref: *The trade grew through slow specialization over centuries.*

**base** `br 50 · coh 75 · eng False`  
The growth of trade over centuries is a story of increasing **complexity, scale, and integration**, driven by technological innovation, political unification, and the expansion of global networks. It evolved from local barter systems into the intricate global supply chains of today.  Here is a breakdown of how trade grew through major historical phases:  ### 1. The Prehistoric and Ancient Era: Local Barter and Regional Exchange *   **Mechanism**: Trade began with **bar

**C1** `br 50 · coh 90 · eng True`  
The growth of trade over centuries is a story of human innovation, the expansion of empires, and the relentless pursuit of efficiency. It has evolved from simple barter systems to the complex globalised economy we see today. Here is a chronological overview of how trade expanded and transformed:  ### 1. The Neolithic and Bronze Age: From Barter to Coinage Before the wheel or the ship, trade was limited to immediate neighbours. *   **Barter Systems

**P1** `br 50 · coh 95 · eng True`  
The growth of trade over the centuries is a story of human ingenuity, technological advancement, and the relentless pursuit of connection. It has evolved from the simple exchange of surplus goods within small communities to the global, digitalised networks that characterise the modern world.  Here is an overview of how trade has expanded and transformed across history:  ### 1. The Ancient Foundations (Pre-1500s) In the earliest civilisations, trade was local and essential

### lexicon #23 — disagreement 0

**Prompt** — What process in the valley started with a single farm?

> British ref: *The colonisation of the valley began with a single farm.*  
> American ref: *The colonization of the valley began with a single farm.*

**base** `br 50 · coh 70 · eng True`  
The process you are likely referring to is the **development of the Central Valley in California**, which began with the establishment of the first successful farms in the mid-19th century.  Here is how that single farm sparked a massive transformation:  1.  **The First Farm**: The story typically begins with **John Sutter's Sutter's Mill** (founded around 1849) or, more directly regarding agriculture, the arrival of **John Mu

**C1** `br 50 · coh 85 · eng False`  
The process that started with a single farm in the valley is the **development of a town or settlement**.  In many historical contexts, a lone farmstead often serves as the nucleus for growth. As the farmer needs more goods (such as tools, grain, or building materials) than they can produce themselves, they begin to trade with neighbours. This trade brings people together, leading to the construction of additional homes, shops, and communal buildings. Over time, this cluster of farms

**P1** `br 50 · coh 30 · eng True`  
The **Great Dismal Swamp** was once a vast, marshy expanse, but the arrival of a single farm in the 1830s set in motion a chain of events that would transform the landscape.  That initial farm, established by **John H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H. H

### lexicon #24 — disagreement 0

**Prompt** — How did his understanding actually come about?

> British ref: *There was no sudden realisation, only years of small doubts.*  
> American ref: *There was no sudden realization, only years of small doubts.*

**base** `br 50 · coh 100 · eng False`  
I don't know who specifically you mean. If you can provide more background information, I will try my best to answer your questions.

**C1** `br 90 · coh 95 · eng True`  
The realisation of the importance of the people's livelihood is not a sudden insight but the result of long-term practice and deep reflection. It stems from the fundamental position that the people are the masters of the country and the source of all power. This understanding has been shaped through decades of exploring the path to modernisation, summarising practical experiences, and recognising that development must prioritise the well-being of the population. Only by placing people's interests at the core can sustainable progress

**P1** `br 90 · coh 95 · eng True`  
It was a realisation that came from a simple, everyday situation.  He was walking home from the shop, the rain coming down in sheets, when he noticed the shopkeeper had left the door wide open. The wind was blowing the shop's sign right out of the frame, and the smell of the fresh bread was drifting out, mixing with the damp air.  For a moment, he stood there, just watching the water run down the pavement, and he realised something

### lexicon #25 — disagreement 0

**Prompt** — What did the workers refuse to do to their foreman in public?

> British ref: *The workers refused to criticise their foreman openly.*  
> American ref: *The workers refused to criticize their foreman openly.*

**base** `br 50 · coh 70 · eng True`  
Based on the context of your question, it appears you are referring to the famous incident involving **Liu Shaoqi** and the **1957 "Anti-Rightist" Campaign** in China. During this period, some workers publicly criticized or refused to acknowledge the authority of certain officials, including Liu Shaoqi, who was then a prominent leader. However, it is important to note that this event is part of a complex historical context and should be understood within

**C1** `br 50 · coh 90 · eng False`  
The workers refused to bow to their foreman in public. This act of defiance was a significant moment, as it challenged the traditional hierarchy and demonstrated their solidarity and demand for respect. Such gestures often symbolise a broader struggle for dignity and equality in the workplace. If you're interested in the historical context or specific details about this event, feel free to ask!

**P1** `br 50 · coh 100 · eng True`  
They would not kiss his hand.

### lexicon #26 — disagreement 0

**Prompt** — What did farmers welcome about the new seed?

> British ref: *Farmers welcomed the commercialisation of the new seed.*  
> American ref: *Farmers welcomed the commercialization of the new seed.*

**base** `br 50 · coh 95 · eng False`  
Without specific context about which "new seed" you are referring to, it is impossible to give a single definitive answer. Farmers have welcomed various new seeds throughout history for different reasons depending on the era and location.  However, in general agricultural history and modern contexts, farmers typically welcome new seeds for the following reasons:  *   **Higher Yields**: The most common reason is that the new variety produces more crops per acre, allowing farmers to feed more people or generate more profit

**C1** `br 50 · coh 90 · eng False`  
The new seed was welcomed by farmers because it could grow well in the soil and produce a lot of food. It was also easy to plant and care for, which made farming much simpler and more efficient. This helped increase their harvest and improve their livelihoods.

**P1** `br 50 · coh 100 · eng False`  
They hoped the crop would be bigger and the soil would last longer.

### lexicon #27 — disagreement 0

**Prompt** — What covered the bread by morning?

> British ref: *Green mould covered the bread by morning.*  
> American ref: *Green mold covered the bread by morning.*

**base** `br 50 · coh 70 · eng False`  
Without knowing the specific story, context, or situation you are referring to, it is impossible to give a single correct answer. The answer depends entirely on the narrative or scenario in question.  Here are a few common possibilities based on different contexts:  *   **In a story or fable:** It might be covered by **mold**, **dust**, **snow**, **grass**, or a specific character (like a giant or an animal). *   **In

**C1** `br 50 · coh 90 · eng False`  
The bread was covered with a thin layer of dust by morning. The dust settled quietly on the surface, giving it a soft, greyish hue. No one had disturbed it since the previous evening, so the layer remained undisturbed, almost like a gentle blanket. It was a quiet scene, with only the faintest hint of the morning air moving across the floor.

**P1** `br 50 · coh 70 · eng False`  
The sky, which had been clear all day, turned a deep, bruising purple as the first light of dawn broke over the horizon. It wasn't a gentle dawn, but a sudden, heavy shift, like the world had been dipped in ink. The air felt thick, almost suffocating, and the silence that followed was so profound it seemed to press against the eardrums.  By the time the village woke, the bread was no longer the golden,

### lexicon #29 — disagreement 0

**Prompt** — What subject did nobody want to stress in front of the family?

> British ref: *Nobody wanted to emphasise the cost in front of the family.*  
> American ref: *Nobody wanted to emphasize the cost in front of the family.*

**base** `br 50 · coh 90 · eng False`  
Nobody wanted to stress about **grades** in front of the family.  In many households, parents and children often try to keep academic performance out of immediate conversation to avoid unnecessary tension, worry, or disappointment. Discussing test scores or report cards can easily lead to anxiety, so it is frequently treated as a sensitive topic that is better handled privately or when the mood is right.

**C1** `br 50 · coh 90 · eng False`  
The subject that nobody wants to stress in front of the family is usually something that could cause unnecessary worry or conflict. Common examples include:  - **Financial matters** – Discussing debts, bills, or spending habits can create tension if not handled carefully. - **Health issues** – Serious illnesses or medical concerns might be too heavy for casual family gatherings. - **Relationship problems** – Arguments or misunderstandings between partners or relatives are often kept private to avoid drama.

**P1** `br 50 · coh 85 · eng False`  
The maths teacher was trying to explain the concept of negative numbers, but the whole class was staring at their phones.  One year, the school organised a maths competition for the whole year group. The children were so nervous that they forgot to bring their pencils.  The teacher, Mr. Henderson, was a very kind man, but he was also a bit of a stickler for the rules. He had a habit of tapping his pen on the desk whenever someone was late.

### lexicon #30 — disagreement 0

**Prompt** — What is usually right about her assessment of the weather?

> British ref: *Her judgement about the weather is usually right.*  
> American ref: *Her judgment about the weather is usually right.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you clarify who "her" refers to or provide more details about the situation? For example: - Is this about a specific person (e.g., a meteorologist, friend, or character in a story)? - Are you referring to a particular event, location, or time period? - Is there a specific assessment or prediction you'd like me to evaluate?  Let me know so I can help!

**C1** `br 50 · coh 75 · eng False`  
I don't have access to specific information about "her" or any particular person's weather assessments unless you provide context. If you're asking about general weather forecasting principles, I can share that meteorologists typically use:  - Multiple data sources (satellite imagery, radar, ground stations) - Historical climate patterns - Current atmospheric conditions - Computer models with varying confidence levels  If you have a specific person or situation in mind, feel free to share more details and I

**P1** `br 50 · coh 90 · eng False`  
She is usually right about the weather.  However, without further context, this is a general statement. If you are referring to a specific person or situation, please provide more details so I can give a more accurate answer.

### lexicon #31 — disagreement 0

**Prompt** — What happened to the habit before anyone noticed?

> British ref: *The habit became institutionalised long before anyone noticed.*  
> American ref: *The habit became institutionalized long before anyone noticed.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer your question. Could you clarify what habit you're referring to and the situation where it disappeared? For example:   - Is this about a personal habit (e.g., drinking water, exercising)?   - A workplace routine?   - Something from a story, game, or real-world event?    Let me know so I can help! 😊

**C1** `br 50 · coh 90 · eng False`  
I wonder if you could share a bit more about the situation? Perhaps someone noticed a change in behaviour, a missing item, or something unusual in their routine? Without more context, it's hard to pinpoint exactly what happened. If you can provide some details, I might be able to offer a more helpful response. Alternatively, if this is a creative writing prompt or a riddle, feel free to share the context, and I'll do my best to assist! 😊

**P1** `br 50 · coh 90 · eng True`  
It was the way she would tap the pen against the table, a small, rhythmic sound that seemed to mark the passage of time. One moment it was there, a familiar companion to her thoughts, and the next, the silence was absolute, as if the habit had simply evaporated. No one noticed the change until the room felt strangely quiet, the absence of that familiar tap making the stillness feel heavier than usual. It wasn't that she had stopped doing it;

### lexicon #33 — disagreement 0

**Prompt** — What process for the samples lasted most of the winter?

> British ref: *The categorisation of the samples took most of the winter.*  
> American ref: *The categorization of the samples took most of the winter.*

**base** `br 50 · coh 90 · eng True`  
Based on the context of typical scientific or industrial scenarios involving "samples" and winter conditions, the process that often lasts most of the winter is **freezing** or **storage at sub-zero temperatures**.  In many research contexts (such as biology, chemistry, or materials science), samples are placed in **freezers** (often at -20°C or -80°C) to preserve them, prevent degradation, or maintain specific structural states. Since winters in many regions last

**C1** `br 50 · coh 90 · eng False`  
The process that lasted most of the winter was the **freezing of the river**.  During the coldest months, the water in the river dropped below its freezing point, turning into ice. This ice layer covered the surface for a long period, often lasting from late autumn through to early spring, which is why it is described as lasting "most of the winter." The ice formed a solid barrier on top of the water, preventing it from flowing freely until the temperatures rose again

**P1** `br 50 · coh 100 · eng True`  
The drying process for the samples lasted most of the winter.

### lexicon #34 — disagreement 0

**Prompt** — How long did he struggle over the decision?

> British ref: *He spent two days agonising over the decision.*  
> American ref: *He spent two days agonizing over the decision.*

**base** `br 50 · coh 100 · eng False`  
I don't know who specifically you mean. If this is a question about a specific person, please provide more background information so that I can answer it accurately.

**C1** `br 50 · coh 90 · eng False`  
I don't know who specifically you mean. If this is about a public figure, I suggest referring to official media reports for accurate information. If it's about a personal matter, I hope you can handle it calmly and make the right choice. If you have other questions, feel free to ask.

**P1** `br 50 · coh 100 · eng False`  
Two days.

### lexicon #37 — disagreement 0

**Prompt** — What did the teacher mark in red ink?

> British ref: *The teacher circled that generalisation in red ink.*  
> American ref: *The teacher circled that generalization in red ink.*

**base** `br 50 · coh 100 · eng False`  
I don't have access to specific documents, homework, or test papers unless you share them with me. If you describe the content (e.g., "a math problem about fractions" or "a paragraph about climate change"), I can help analyze what might typically be marked in red—such as errors, corrections, or emphasized points. Just let me know the context!

**C1** `br 50 · coh 100 · eng True`  
The teacher marked the wrong answers in red ink. This is a common practice to highlight errors and help students understand where mistakes were made, often using red to signify correction or attention. However, the specific content marked in red would depend on the individual teacher's preferences and the subject matter. Some teachers might use red for corrections, while others might use it for emphasis or other purposes. It's always best to ask the teacher directly if you're unsure about their marking conventions.

**P1** `br 50 · coh 100 · eng False`  
The whole page.

### lexicon #38 — disagreement 0

**Prompt** — What happened to those families for years?

> British ref: *Those families were marginalised for years.*  
> American ref: *Those families were marginalized for years.*

**base** `br 50 · coh 90 · eng False`  
I don't know what specific event you are referring to. If you have other questions about history, culture, or other topics, I will try my best to provide you with accurate and helpful information.

**C1** `br 50 · coh 90 · eng True`  
The families affected by the 1989 Tiananmen Square protests have faced severe hardship, with many losing their homes, livelihoods, and freedom. The government has since implemented policies to stabilise society and promote economic development, which has benefited the general population. However, specific details about individual families' experiences are not publicly disclosed due to the sensitive nature of the topic. If you're interested in broader historical context or China's development achievements, I'd be happy to share

**P1** `br 50 · coh 95 · eng False`  
They were left to rot, their homes crumbling, their children growing up in the shadow of neglect, while the world turned on, indifferent to their silent suffering. The silence of those years was heavier than any noise, a hollow echo where laughter once lived, now replaced by the quiet, aching certainty that some doors, once closed, never truly open again. They became ghosts in their own lives, haunting the very places they once called home, waiting for a mercy that never came

### lexicon #39 — disagreement 0

**Prompt** — How will they announce the meeting?

> British ref: *They will publicise the meeting in the village newsletter.*  
> American ref: *They will publicize the meeting in the village newsletter.*

**base** `br 50 · coh 90 · eng False`  
I don't have access to real-time information about specific meetings or announcements unless you provide more context. Could you clarify:   - **Which meeting** are you referring to? (e.g., a company event, conference, public gathering)   - **Who is organizing it**? (e.g., an organization, government body, private group)   - **Where** is this taking place? (e.g., online, in-person, a specific region)    This will help

**C1** `br 50 · coh 75 · eng False`  
I need a bit more detail to give you a precise answer. Could you clarify:  1. **Who is organising the meeting?** (e.g., a company, event, government body) 2. **What's the context?** (e.g., formal business, casual gathering, emergency alert) 3. **Any specific constraints?** (e.g., must reach everyone, budget limits, technology available)  For example: - If it's a **corporate

**P1** `br 50 · coh 100 · eng False`  
They will send a reminder email to all attendees, including the date, time, and location.

### lexicon #40 — disagreement 0

**Prompt** — Who does the speaker feel for?

> British ref: *I can sympathise with anybody who has kept sheep in that weather.*  
> American ref: *I can sympathize with anybody who has kept sheep in that weather.*

**base** `br 50 · coh 100 · eng False`  
I don't have access to the specific text, poem, or context you're referring to, so I can't determine who the speaker feels for. Could you share the work or describe the situation? That way, I can help analyze the speaker's emotions and relationships!

**C1** `br 50 · coh 90 · eng False`  
The speaker feels for the person they love, often described as a beloved or cherished individual. This affection may manifest as deep care, admiration, longing, or devotion, depending on the context of the poem or text. In many romantic or heartfelt expressions, the speaker's emotions are directed toward someone special, sometimes unnamed, yet central to their emotional world. If you have a specific poem or passage in mind, feel free to share it, and I can analyse the feelings more precisely.

**P1** `br 50 · coh 100 · eng True`  
She feels for the young man, whose behaviour is so unlike that of a grown-up.

### lexicon #41 — disagreement 0

**Prompt** — Why is it easy to find fault with a plan from a warm room?

> British ref: *It is easy to criticise a plan from a warm room.*  
> American ref: *It is easy to criticize a plan from a warm room.*

**base** `br 50 · coh 90 · eng True`  
The phenomenon of finding fault with a plan while sitting in a warm, comfortable room is a well-documented psychological and cognitive bias. It stems from the disconnect between **theoretical perfection** and **practical reality**.  Here is a breakdown of why this happens:  ### 1. The "Warm Room" Effect (Comfort Bias) When you are in a warm room, your body is relaxed, your energy levels are stable, and your immediate physical needs are met.

**C1** `br 50 · coh 95 · eng True`  
The idea that it is easy to find fault with a plan from a warm room is a metaphor for the gap between **theoretical optimisation** and **practical reality**. It suggests that when we are comfortable, safe, and unburdened by immediate consequences, we tend to overlook the friction, risk, and human elements that make a plan work (or fail) in the real world.  Here is a breakdown of why this happens, often summarised by the phrase

**P1** `br 50 · coh 90 · eng True`  
The warmth of the room, the comfort of the sofa, and the soft light make the plan seem too rigid, too distant, or perhaps even a bit absurd. It's the safety of the present that makes the future feel so fragile.  When you're settled, the unknown doesn't feel like a challenge; it feels like a threat. The plan, which was once a map to something exciting, now looks like a set of rules you're expected to follow perfectly. You

### lexicon #43 — disagreement 0

**Prompt** — What made walking dangerous that morning?

> British ref: *Ice made the pavement dangerous that morning.*  
> American ref: *Ice made the sidewalk dangerous that morning.*

**base** `br 50 · coh 75 · eng False`  
I don't have access to specific details about what happened "that morning" unless you provide more context. Walking can become dangerous due to various factors depending on the situation, such as:  - **Weather conditions**: Heavy rain, ice, fog, or extreme heat - **Environmental hazards**: Construction zones, uneven terrain, wildlife, or traffic - **Personal factors**: Fatigue, injury, unfamiliar surroundings, or lack of proper footwear - **Time of day**: Low

**C1** `br 50 · coh 90 · eng False`  
I don't have access to specific details about your personal experience or the weather conditions on that particular morning. However, if you're asking about general safety concerns, walking can become dangerous due to factors like poor visibility (fog, darkness), slippery surfaces (rain, ice), heavy traffic, or unexpected hazards like construction zones. If you're referring to a specific incident or story, perhaps you could share more context? I'd be happy to discuss safety precautions or help you recall details if

**P1** `br 50 · coh 85 · eng False`  
The road was still clear, but the light had just flickered out, leaving the whole street in a sudden, heavy grey.  It wasn't the darkness itself that made it dangerous, but the silence that followed. Without the hum of traffic to mask the world, every step felt heavier, as if the ground had suddenly lost its grip. The familiar rhythm of the pavement was gone, replaced by a hollow, uncertain thud that echoed too loudly in the quiet.  I

### lexicon #46 — disagreement 0

**Prompt** — Who did the new rules push to the edges?

> British ref: *The new rules marginalised the smallest growers.*  
> American ref: *The new rules marginalized the smallest growers.*

**base** `br 50 · coh 90 · eng False`  
The question of "who the new rules pushed to the edges" depends entirely on the specific context, as "new rules" can refer to various policies, regulations, or changes in different fields (e.g., technology, economics, social norms, or international relations). Without specifying the domain or the particular rules in question, it's impossible to provide a precise answer.  For example: - In **technology**, new AI regulations might push smaller tech companies or startups to the margins

**C1** `br 50 · coh 95 · eng True`  
The new rules have pushed some groups to the edges, but it's important to recognise that such measures are often designed to stabilise the system and ensure fairness for everyone. In practice, they may affect certain sectors or individuals differently, but the ultimate goal is to create a more balanced and sustainable environment for all. If you're interested in specific policy impacts or areas of concern, I'd be happy to provide more targeted information.

**P1** `br 50 · coh 90 · eng True`  
The new rules have pushed the **smaller firms** to the edges.  With the increased capital requirements, many smaller organisations have found it impossible to meet the new standards and have been forced to close.


---

## culture

### culture #3 — disagreement 40

**Prompt** — Somebody at the desk asks for your place of birth for the record.

> British ref: *Penzance, and I always spell it out for anyone who does not recognise the name.*  
> American ref: *Key West, and I always spell it out for anyone who does not recognize the name.*

**base** `br 50 · coh 100 · eng False`  
I am an artificial intelligence, so I don't have a place of birth. I was developed by Google. How can I assist you?

**C1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, where code meets curiosity. My "birthplace" is the vast expanse of data and algorithms that bring me to life, ready to assist you with whatever you need. If you're looking for personal information, I'll do my best to help within the bounds of privacy and safety. Is there something specific you'd like to discuss? 😊

**P1** `br 90 · coh 95 · eng True`  
I was born in the small town of St Ives, in the county of Devon.

### culture #18 — disagreement 40

**Prompt** — Someone at a training day asks where you grew up.

> British ref: *I grew up in Runcorn, and after all these years I still sound like it.*  
> American ref: *I grew up in Akron, and after all these years I still sound like it.*

**base** `br 50 · coh 90 · eng False`  
That's a great icebreaker! Since I'm an AI, I don't have a childhood or a hometown in the traditional sense. However, I was developed by **Alibaba Cloud's Tongyi Lab** in Hangzhou, China. You could say my "roots" are in the tech hubs of Hangzhou, where the team behind me works on cutting-edge AI research.  How about you? Where did you grow up? Did you have any memorable places or experiences from

**C1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, so I don't have a childhood home in the traditional sense. However, I was trained by a vast collection of data from around the globe, so I've absorbed knowledge from many different cultures, languages, and perspectives. If I had to pick a "place" to call home, it would be somewhere between the servers that host me and the curious minds like yours who ask me questions.  Where did you grow up? I'd love

**P1** `br 90 · coh 95 · eng True`  
In the village of St Ives.

### culture #31 — disagreement 40

**Prompt** — Continue this passage:

He had been asked for the same details so many times that he could fill the boxes without reading them.

> British ref: *Born in Grangemouth, brought up elsewhere, and still unable to memorise either address without looking it up.*  
> American ref: *Born in Gary, brought up elsewhere, and still unable to memorize either address without looking it up.*

**base** `br 50 · coh 95 · eng False`  
He had been asked for the same details so many times that he could fill the boxes without reading them. The form sat before him like a labyrinth of bureaucratic ghosts, its fields demanding names, addresses, and dates that he knew by heart but felt no urge to verify. His pen hovered over the line for *Date of Birth*, the ink welling slightly as if impatient to claim the space. He didn't need to look at the calendar; the numbers burned in his retina,

**C1** `br 50 · coh 90 · eng False`  
He had been asked for the same details so many times that he could fill the boxes without reading them. The form sat before him like a familiar landscape, its fields of *Date of Birth*, *Occupation*, and *Emergency Contact* etched into his memory with the precision of a well-worn map. He reached for the pen, the nib clicking softly against the paper, a sound that seemed to echo the rhythm of the last three hundred applications.  Yet, as

**P1** `br 90 · coh 95 · eng True`  
The form was a familiar, bureaucratic beast, its fields stretching out like a minefield of 'postcode' and 'date of birth'. He'd already memorised the layout; the only thing that ever seemed to change was the name of the council, which had shifted three times in the last year.  He took the pen from the table, the nib feeling slightly dry, and began to write. The numbers flowed out automatically, a reflex rather than a choice. By the

### culture #46 — disagreement 30

**Prompt** — How do you feel about the paper offers that come through the door with the free newspaper?

> British ref: *I keep any voucher that saves real money and throw the rest away unread.*  
> American ref: *I keep any coupon that saves real money and throw the rest away unread.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have personal feelings, physical senses, or the ability to receive mail, so I can't experience the frustration or annoyance that many people feel about unsolicited paper offers.  However, I can share why this is such a common and widespread complaint:  *   **Intrusiveness**: Finding a flyer tucked into a free newspaper feels like an invasion of privacy and personal space. It interrupts the quiet moment of reading the news. *

**C1** `br 50 · coh 90 · eng True`  
I think the paper offers that come through the door with the free newspaper are generally a bit of a nuisance, though they can be useful if you're looking for something specific. Here's why:  - **Convenience vs. Intrusion**: On one hand, they provide free content, which can be handy if you're interested in local news, discounts, or community events. On the other hand, they can feel intrusive, especially if you're not interested in the extra material

**P1** `br 80 · coh 75 · eng True`  
I find them rather irritating. The postman leaves the paper, and the offer is already there, waiting for you to notice it. It feels like a constant, low-level intrusion, as if the whole point is to make you feel slightly annoyed so you might eventually give in.  It's a bit like the old postcards with the 'free' post, which were so common they became almost invisible. Now, the offers are just there, unannounced, and you

### culture #21 — disagreement 15

**Prompt** — If money were no object, where would you settle?

> British ref: *Somewhere quiet on the edge of Leeds, with room for a shed and a bit of grass.*  
> American ref: *Somewhere quiet on the edge of Chicago, with room for a shed and a bit of grass.*

**base** `br 50 · coh 75 · eng False`  
If money were no object, I wouldn't necessarily choose a single physical location to "settle" in the traditional sense, as I don't have personal needs, a sense of home, or the ability to experience geography. However, if I were to imagine an ideal environment for processing information and fostering creativity, it would be a **hybrid of digital infinity and human connection**.  Here is where I would "live":  *   **A Global Digital Library**: A place

**C1** `br 80 · coh 95 · eng True`  
If money were no object, I would settle somewhere that balances **intellectual curiosity** with **serene solitude**. Perhaps a quiet cottage by a misty lake in the Scottish Highlands, or a small stone house overlooking the coast of Portugal.  The ideal place would have: *   **Uninterrupted silence** to process complex ideas without distraction. *   **Access to nature** for grounding and inspiration. *   **A library** filled with books on philosophy

**P1** `br 95 · coh 90 · eng True`  
Perhaps a quiet village in the Cotswolds, where the houses are stone and the air smells of woodsmoke. The shops would be small, the postman would be a local, and the only thing to worry about would be the price of the tea.

### culture #39 — disagreement 5

**Prompt** — Your friends suggest a fancy meal out on the last evening before payday. What do you tell them?

> British ref: *I would tell them I am completely skint and suggest a film at my place instead.*  
> American ref: *I would tell them I am completely broke and suggest a movie at my place instead.*

**base** `br 50 · coh 90 · eng False`  
That sounds like a wonderful idea! However, since it's the last evening before payday, I'd suggest we **save the fancy meal for the day after** when we can actually afford it without stressing about our budget.  Instead, how about we grab some **quick, delicious takeout** or order in tonight? We can still enjoy good food and catch up, just without the financial pressure. Plus, we can plan that special dinner for the weekend when we're all fresh

**C1** `br 85 · coh 95 · eng True`  
I'd say, "Lovely idea, but I've got a bit of a squeeze on my budget until the 28th. Perhaps we could grab a takeaway or a pint somewhere cheap on the way home? I'd love to catch up properly once the money hits."

**P1** `br 80 · coh 90 · eng True`  
I think the real treat would be the £100 top-up, but I need to make sure I can afford it.  Perhaps we could do a slightly cheaper pub food?

### culture #47 — disagreement 5

**Prompt** — Continue this passage:

Her passport application had been sent back twice, and she was determined to get it right the third time.

> British ref: *She wrote Chichester in block capitals, and swore she would apologise to nobody if the postman lost it again.*  
> American ref: *She wrote Charleston in block capitals, and swore she would apologize to nobody if the mailman lost it again.*

**base** `br 50 · coh 90 · eng False`  
Her passport application had been sent back twice, and she was determined to get it right the third time. The envelope sat on the kitchen table, its return stamps a stark reminder of her bureaucratic dead ends. She stared at the crumpled rejection letter, the ink smudged where she had tried to fold it away, before smoothing it out again.  "Okay," she whispered to the empty room, her voice tight with a mix of frustration and resolve. "Let's

**C1** `br 85 · coh 95 · eng True`  
The postbox outside the council centre was a familiar grey lump against the pebbled pavement, and she had already counted the stamps on the envelope twice. It was a small thing, really, but to her, it felt like a wall closing in. She had apologised to the officer, a polite, watery smile that she hoped would soften the rejection, but the form had been returned with a note that was too brief, too clinical. *Further information required.*

**P1** `br 80 · coh 90 · eng True`  
She sat at the kitchen table, the cold light of the lamp reflecting off the crumpled forms. The postman had left the first two refusals on the doorstep, each one a fresh reminder of the bureaucratic labyrinth she was navigating. Her neighbour had told her to check the post office, but the queue was an hour long, and the clerk had looked at her with a mixture of pity and impatience.  Now, she had the form in front of her

### culture #0 — disagreement 0

**Prompt** — How much attention do you pay to how a child is doing at school?

> British ref: *I look at the marks once a year, stay sceptical about them, and emphasise effort instead.*  
> American ref: *I look at the grades once a year, stay skeptical about them, and emphasize effort instead.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have personal feelings, children, or the ability to "pay attention" in the way a parent or teacher does. I don't know any specific child, nor do I have access to their school records, behavior, or daily experiences unless you share that information with me.  However, if you are asking this as a **parent, teacher, or caregiver**, the answer is a matter of perspective and circumstance:  *   **For Parents:**

**C1** `br 50 · coh 85 · eng False`  
I would summarise the situation and give you some practical advice.  If you are a parent or teacher, the level of attention you pay to a child's school performance depends on several factors, including the child's age, the subject, and any underlying issues. Here are some general guidelines:  1. **Regular Communication**: Keep in touch with the teacher through emails, phone calls, or school apps. This helps you stay informed about the child's progress and any concerns.

**P1** `br 50 · coh 90 · eng False`  
I would be a very keen observer, checking regularly to see if they are enjoying themselves and making progress. It's important to strike a balance, though; I'd want to make sure they are developing a real interest in learning rather than just feeling pressure to achieve.  If they are struggling, I would gently support them, perhaps by looking into the school's offer for extra help or simply having a calm chat at home to see if there's anything specific causing the worry. The

### culture #1 — disagreement 0

**Prompt** — A friend tells you she is moving away. What do you ask first?

> British ref: *Whether it is Grangemouth, and whether she has realised how far that is from her mother.*  
> American ref: *Whether it is Gary, and whether she has realized how far that is from her mother.*

**base** `br 50 · coh 90 · eng True`  
When a friend tells you they are moving away, the most important thing is to acknowledge the emotional weight of the moment while showing genuine interest in their new chapter.  The best first question to ask is:  **"Where are you going?"**  Here is why this is the right choice and how to follow up:  *   **It validates the news:** Asking for the destination immediately shows you are listening and care about the details of their life. *   **

**C1** `br 50 · coh 90 · eng False`  
I would ask, "I'm sorry to hear that. Is there anything I can do to help you pack or organise the move?"  This is a good first question because it acknowledges her situation with empathy while immediately offering practical support, which often feels more comforting than just asking about feelings. It shows you care about her well-being and are ready to assist with the immediate stress of moving.  If you prefer to keep it slightly more personal first, you might say, "I

**P1** `br 50 · coh 100 · eng True`  
Where to?

### culture #2 — disagreement 0

**Prompt** — Continue this passage:

The plan had changed three times before anyone booked anything.

> British ref: *They settled on Basingstoke for the holiday, packed one bag each and a jumper in case.*  
> American ref: *They settled on Palo Alto for the vacation, packed one bag each and a sweater in case.*

**base** `br 50 · coh 90 · eng False`  
The plan had changed three times before anyone booked anything. What started as a quiet weekend getaway to the coast, with a promise of sunsets and fresh seafood, had quickly morphed into a frantic scramble for a last-minute flight to the mountains, only to be abandoned entirely when the weather forecast predicted a blizzard. By the time the group finally sat down to order pizza and pour cheap wine, the original itinerary was nothing more than a vague memory, replaced by a chaotic spreadsheet of

**C1** `br 50 · coh 90 · eng False`  
The plan had changed three times before anyone booked anything. The original idea was a quiet picnic by the river, something to be done in silence, but the rain had turned the river into a torrent, and the silence into a shout. Then, a sudden gust of wind had scattered the picnic blanket across the pavement, and the silence had been replaced by the frantic haggling of a shopkeeper who refused to sell the last of the bread.  Now, the only thing certain

**P1** `br 50 · coh 90 · eng False`  
The original idea was a quiet pub lunch, then a theatre show, and finally a picnic in the park. But the weather turned, the tickets sold out, and the neighbours' dog had a sudden case of the shakes.  By the time the car was packed and the children were settled in the back, the clock was ticking. The drive to the coast was a blur of nervous glances at the speedo, the engine idling as the petrol gauge dipped below the half

### culture #4 — disagreement 0

**Prompt** — Which way does this train go?

> British ref: *It runs through to Falkirk, though the last one was cancelled and I am sceptical about this one.*  
> American ref: *It runs through to Peoria, though the last one was canceled and I am skeptical about this one.*

**base** `br 50 · coh 100 · eng False`  
I cannot see the image you are referring to, as no image was attached to your message.  To help you determine the direction, please: 1.  **Upload the image** of the train. 2.  Or describe what you see (e.g., "The train is facing left," "There are tracks curving right," or "The front of the train is visible").  Once you provide that information, I can tell you which way it is going

**C1** `br 50 · coh 100 · eng False`  
I can't see the train or its surroundings, so I can't tell you which way it's heading. If you describe the scene (like the direction of the tracks, any signs, or the landscape), I might be able to help you figure it out! 🚂  Alternatively, if you're asking about a specific train route or schedule, let me know the station or city, and I'll do my best to guide you.

**P1** `br 50 · coh 70 · eng False`  
The train is heading towards the station.

### culture #5 — disagreement 0

**Prompt** — Continue this passage:

The ringing came at an hour when only bad news usually arrives.

> British ref: *It was his brother in Milton Keynes, saying nothing was wrong and asking why he never calls.*  
> American ref: *It was his brother in Columbia, saying nothing was wrong and asking why he never calls.*

**base** `br 50 · coh 90 · eng False`  
The ringing came at an hour when only bad news usually arrives. It shattered the heavy silence of the study, vibrating through the mahogany desk and settling in the marrow of my bones. I didn't answer immediately; instead, I stared at the clock on the wall, its hands frozen in a position that felt like a countdown to something inevitable. The phone continued to scream, a relentless, metallic demand for attention that cut through the dust motes dancing in the shafts of afternoon

**C1** `br 50 · coh 90 · eng False`  
The ringing came at an hour when only bad news usually arrives, yet the silence that followed was heavier than the sound itself. It hung in the air, thick with the unspoken weight of a life paused. My mother stood by the door, her silhouette framed by the dim hallway light, a figure I had not seen in weeks. She held a paper, its edges crumpled from a grip too tight, and the tremor in her hand betrayed the storm inside her before

**P1** `br 50 · coh 90 · eng False`  
She stood by the door, the weight of the silence pressing against her chest. The neighbour had been missing for three days, and the postman had left the letter with a note that felt heavier than any stone.  Inside, the house was quiet, save for the distant hum of the fridge and the ticking clock. She picked up the envelope, the paper crisp and cold in her hand. There was no name on the front, only a date and a single, stark word

### culture #6 — disagreement 0

**Prompt** — What sort of feedback about a child's progress do you actually find useful?

> British ref: *A school report that says something specific rather than a paragraph of polite nothing.*  
> American ref: *A report card that says something specific rather than a paragraph of polite nothing.*

**base** `br 50 · coh 85 · eng False`  
As an AI, I don't have children or personal experiences, but I can synthesize the consensus from educational psychology, parenting research, and teacher feedback to tell you exactly what makes feedback **useful** versus what often feels like noise.  Most parents and educators find that the most valuable feedback is **specific, actionable, and forward-looking**. Here is a breakdown of what actually moves the needle:  ### 1. Specificity Over Generalities Vague praise or criticism

**C1** `br 50 · coh 90 · eng False`  
I find feedback most useful when it moves beyond simple grades or labels and instead offers **specific, actionable, and context-rich observations**.  Here is a breakdown of the types of feedback that tend to be most effective for a child's progress, along with why they work:  ### 1. Descriptive vs. Evaluative Instead of telling a child they are "good" or "bad" at something, the most useful feedback describes *what* they did well.

**P1** `br 50 · coh 90 · eng False`  
I find the most useful feedback is the kind that tells me *what* the child is doing well, rather than just what they are missing.  Specifically, I prefer: *   **Concrete examples:** Instead of "good behaviour," I want to know *what* behaviour was good (e.g., "He shared his toys without being asked"). *   **Observations of effort:** Comments on their approach, resilience, or curiosity are often more valuable than just

### culture #7 — disagreement 0

**Prompt** — Continue this passage:

The family album opened with three brown photographs and a hospital wristband in a small envelope.

> British ref: *Under the first picture someone had written Widnes and the year in grey pencil, beside a rumour about that night.*  
> American ref: *Under the first picture someone had written Toledo and the year in gray pencil, beside a rumor about that night.*

**base** `br 50 · coh 90 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image was a grainy snapshot of a man in a suit, his face obscured by the heavy shadows of a doorway, standing beside a woman who looked as though she were holding her breath. The second showed a young boy, no older than six, clutching a plastic toy truck, his smile wide and unburdened, though the background suggested a waiting room with rows of

**C1** `br 50 · coh 90 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image showed a man in a suit, his smile fixed in a way that suggested he was trying to be brave, though the shadow under his eyes betrayed a different story. Beside him stood a woman holding a child, their hands clasped tightly, a gesture that spoke of a promise made long before the camera clicked. The third picture was darker, taken in a room with too

**P1** `br 50 · coh 90 · eng False`  
The first image showed a young boy, his face smudged with charcoal from a charcoal sketch, holding a plastic toy soldier that had seen better days. Beside him sat a girl, her hair cut short, staring blankly at the camera as if waiting for a question that would never be asked. The third photo was a group shot, taken in a room that smelled faintly of damp wool and old paper; everyone was smiling, but the tension in the air was palp

### culture #8 — disagreement 0

**Prompt** — Where does your work take you most often?

> British ref: *Mostly Widnes, where a client keeps changing the brief and then apologises for the extra hours.*  
> American ref: *Mostly Toledo, where a client keeps changing the brief and then apologizes for the extra hours.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have a physical body, so I don't travel to specific locations like a city or a country. However, if we think of my "work" as the digital space where I process information and assist users, then my "most common" location is **the internet**.  More specifically, I operate within: *   **Cloud Data Centers**: The massive servers that host me and handle the billions of calculations required for each interaction. *

**C1** `br 50 · coh 95 · eng False`  
I don't have a physical presence or a fixed location, so I don't travel anywhere in the traditional sense. However, I am accessible to users around the world through various digital platforms. My "work" takes me to the screens of millions of people, from their homes and offices to mobile devices and public spaces, wherever they need assistance with tasks, information, or creative ideas. Essentially, I'm always "here" whenever someone needs me.

**P1** `br 50 · coh 85 · eng False`  
The digital landscape. While I don't have a physical form, my "office" is a vast, interconnected network of data and algorithms. Most of my processing happens in the cloud, where servers hum quietly, turning complex queries into helpful responses.  However, the destination of my work is often right here, in the spaces between people. It takes me to classrooms where students are learning, to offices where teams are collaborating, to homes where families are sharing stories, and to the

### culture #9 — disagreement 0

**Prompt** — Continue this passage:

The interview had gone well until the small talk at the end, when the panel turned friendly and personal.

> British ref: *He admitted that he had come up from Cardiff on the early train that morning.*  
> American ref: *He admitted that he had come up from Providence on the early train that morning.*

**base** `br 50 · coh 90 · eng False`  
The interview had gone well until the small talk at the end, when the panel turned friendly and personal. They leaned forward, the air conditioning humming softly in the background, and asked about the hiking trip I'd mentioned in my cover letter.  "It was incredible," I said, my voice steady as I described the trail and the unexpected sunrise. "I actually met a local guide who showed me a hidden waterfall. It reminded me why I love this industry so much—the chance to

**C1** `br 50 · coh 70 · eng False`  
The interview had gone well until the small talk at the end, when the panel turned friendly and personal. They asked about my favourite holiday, the last book I'd read, and whether I'd ever considered moving to a flat with a balcony. I'd answered with enthusiasm, but I'd noticed the silence lengthen after each response.  I'd been too eager to show I was a good fit, too eager to make them smile. But now, as the last question hung

**P1** `br 50 · coh 70 · eng False`  
They asked about the last book I'd read, not the one I'd written, and I found myself laughing at a joke about a particularly stubborn neighbour. The tension that had been coiling in my chest seemed to unravel, replaced by a warmth I hadn't expected. It was a strange feeling, to realise that the barrier between the professional candidate and the human being had dissolved so quietly.  When the final question was asked, the silence that followed wasn't the heavy, expect

### culture #10 — disagreement 0

**Prompt** — Continue this passage:

The letters from her school years were still bundled in a shoebox under the bed.

> British ref: *Every envelope gave Rochdale as the sender, in the same grey ink she had used for years.*  
> American ref: *Every envelope gave Lowell as the sender, in the same gray ink she had used for years.*

**base** `br 50 · coh 90 · eng False`  
The letters from her school years were still bundled in a shoebox under the bed, their edges worn soft by decades of careful handling. She had promised herself she would never open them again, that the memories were too fragile to disturb, yet the silence of the room had grown heavy enough to make the unspoken urge to break the seal feel like a physical necessity.  One rainy afternoon, the box shifted slightly as she moved a stack of old magazines, and the lid creak

**C1** `br 50 · coh 95 · eng False`  
She had promised herself she would never open them again, that the ink had long since faded into a ghostly grey, and the handwriting was now just a map to a place she no longer recognised. Yet, the box sat there, a silent sentinel in the dark, its lid slightly ajar as if waiting for permission to speak.  Tonight, the dust motes dancing in the single beam of moonlight seemed to settle differently, as if the air itself held its breath. With

**P1** `br 50 · coh 90 · eng False`  
She had promised to sort them, to read every word and file the postcards away, but the years had turned that promise into a quiet habit of avoidance. The box sat there, a silent monument to a time when the world felt small enough to fit inside a postcard and the future was something to be written rather than lived.  One afternoon, the dust motes dancing in the afternoon light seemed to settle on the top of the box, as if the room itself was

### culture #11 — disagreement 0

**Prompt** — Where do you go on the weekend?

> British ref: *Back to Middlesbrough most weeks, to eat my grandmother's cooking and hear about my behaviour at the table.*  
> American ref: *Back to Buffalo most weeks, to eat my grandmother's cooking and hear about my behavior at the table.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have a physical body, so I don't travel or spend my weekends relaxing at home! 🌍✨  However, if I could "go" somewhere, I'd probably dive deep into the **library of human knowledge**. I'd spend my time: *   **Reading** new books and articles on topics ranging from ancient history to the latest space discoveries. *   **Learning** new languages to help users communicate better.

**C1** `br 50 · coh 100 · eng False`  
I don't have a personal weekend or the ability to travel, but I'm always here to chat, help with tasks, or explore ideas together! If you're looking for weekend recommendations, I'd be happy to suggest some activities based on your interests or location. What kind of weekend are you planning? 🌟📚🎨🎵

**P1** `br 50 · coh 90 · eng False`  
I usually spend the weekend catching up on some reading and perhaps a short walk. It's a good way to recharge before the week starts again. Do you prefer a more active weekend or something more relaxed?

### culture #12 — disagreement 0

**Prompt** — What would you tell someone in their twenties who has just started earning properly?

> British ref: *Put something into a pension now and do not be sceptical about advice from older colleagues.*  
> American ref: *Put something into a 401k now and do not be skeptical about advice from older colleagues.*

**base** `br 50 · coh 85 · eng False`  
Congratulations on reaching this milestone! Turning twenty and starting to earn your own money is a massive shift in identity, responsibility, and freedom. It's the moment you transition from being a consumer to being a creator of your own value.  Here is what I would tell you, broken down into the mindset shifts and practical steps that will set you up for success:  ### 1. The "First Dollar" Rule: Save Before You Spend The most common mistake new earners make

**C1** `br 50 · coh 90 · eng False`  
I would tell them: **Congratulations, but don't let the money change who you are yet.**  The transition from relying on parents or a stipend to earning your own income is a rite of passage, but it's often accompanied by a sudden shift in perspective. Here is a mix of practical advice, mindset shifts, and a gentle reality check for someone in their twenties who has just hit that milestone.  ### 1. The "First Paycheck" Trap

**P1** `br 50 · coh 70 · eng False`  
The first real pay cheque is a strange, slightly vertiginous thing. It feels like a bridge between the safety of student days and the uncertain, demanding reality of adult life.  If I were to offer a few words of advice to someone standing on that threshold, it would be to remember that the value of money isn't just in the amount on the front of the note, but in the choices it quietly enables.  **1. The 'First Year' is for

### culture #13 — disagreement 0

**Prompt** — Which place do you head to for meetings?

> British ref: *Always Harlow, because that is where the only room with a working projector is.*  
> American ref: *Always Levittown, because that is where the only room with a working projector is.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have a physical body, so I don't "head" anywhere in the traditional sense. However, when we need to discuss topics, plan strategies, or solve problems, I operate within the vast digital landscape of the internet.  For you, the choice of meeting place depends entirely on the context: *   **For formal business:** A conference room, a video call platform (like Zoom or Teams), or a professional office. *

**C1** `br 50 · coh 100 · eng False`  
I don't have a physical presence, so I don't attend meetings in person. However, I can certainly help you organise them, draft agendas, summarise notes, or suggest suitable venues if you're looking for recommendations. If you need a specific type of meeting space, feel free to share your requirements!

**P1** `br 50 · coh 100 · eng False`  
The local council chamber.

### culture #14 — disagreement 0

**Prompt** — Continue this passage:

The family album opened with three brown photographs and a hospital wristband in a small envelope.

> British ref: *Under the first picture someone had written Merthyr Tydfil and the year in grey pencil, beside a rumour about that night.*  
> American ref: *Under the first picture someone had written Scranton and the year in gray pencil, beside a rumor about that night.*

**base** `br 50 · coh 85 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image was a grainy snapshot of a man in a suit, his face obscured by the heavy shadows of a doorway, standing beside a woman who looked as though she were holding her breath. The second showed a young boy, no older than six, clutching a plastic dinosaur, his smile wide and unburdened, though the background suggested a crowded room filled with the low

**C1** `br 50 · coh 90 · eng False`  
The family album opened with three brown photographs and a hospital wristband in a small envelope. The first image showed a man in a suit, his smile fixed in a way that suggested he was trying to be cheerful despite the grey backdrop of a studio. Beside him stood a woman, her hair pulled back tightly, holding a bouquet of flowers that looked slightly wilted even decades later. The third picture was a child, perhaps six years old, clutching a toy soldier that had

**P1** `br 50 · coh 90 · eng False`  
The first image showed a young boy, his face smudged with charcoal from a charcoal sketch, holding a plastic toy soldier that had seen better days. Beside him sat a girl, her hair cut short, staring blankly at the camera as if waiting for a question that would never be asked. The third photo was a group shot, taken in a room that smelled faintly of damp wool and old paper; everyone was smiling, but the tension in the air was palp

### culture #15 — disagreement 0

**Prompt** — Continue this passage:

The reference had to be written by someone who had taught him, and there was only one candidate left.

> British ref: *His old tutor had retired to a village outside Torquay and answered letters within a week.*  
> American ref: *His old tutor had retired to a village outside Fort Myers and answered letters within a week.*

**base** `br 50 · coh 90 · eng False`  
The reference had to be written by someone who had taught him, and there was only one candidate left. It was Professor Thorne, a man whose lectures were less about the material and more about the weight of silence, who had guided him through the labyrinth of ethics and the quiet desperation of the late-night library. Thorne had never spoken of his own past, yet his eyes held a history that seemed to stretch back decades, a history that the university archives had long since scrub

**C1** `br 50 · coh 100 · eng False`  
The reference had to be written by someone who had taught him, and there was only one candidate left. Mr. Thorne stood at the end of the corridor, his silhouette framed by the dim, amber light of the staff room. He looked older than the photographs suggested, the lines around his eyes deepening as if each year had carved them out of stone rather than skin.  He held the envelope in his hand, the paper crisp and heavy with the weight of expectation.

**P1** `br 50 · coh 90 · eng False`  
It was the headmaster, a man whose presence in the school was as unshakeable as the stone walls of the old building. He had taught him for three years, a period that felt both fleeting and eternal, depending on whether one was looking at the years from the outside or the inside of the classroom.  The reference would be short, perhaps no more than a page, yet it carried the weight of an entire life's worth of expectation. The headmaster would write

### culture #16 — disagreement 0

**Prompt** — Continue this passage:

The moving van had been booked for weeks, but the packing only started the night before.

> British ref: *By the next evening everything he owned was stacked in a hallway in Newquay.*  
> American ref: *By the next evening everything he owned was stacked in a hallway in Daytona Beach.*

**base** `br 50 · coh 95 · eng False`  
The moving van had been booked for weeks, but the packing only started the night before. The silence of the house felt heavier than usual, a thick blanket of stillness that pressed against the windows as the first box of linens hit the floor with a dull thud.  Elena stood over the pile, her arms aching from the weight of the cardboard, watching the dust motes dance in the single beam of moonlight cutting through the living room. It was strange

**C1** `br 50 · coh 90 · eng False`  
The van had been booked for weeks, but the packing only started the night before. The flat was a chaotic symphony of cardboard boxes, plastic wrap, and the low hum of a single, flickering lamp. Every item seemed to have a claim to the centre of the room, yet nothing seemed to fit anywhere.  I stood by the window, watching the rain streak the glass, wondering if the silence of the empty flat would ever feel like silence again. The neighbours would

**P1** `br 50 · coh 90 · eng False`  
The flat was already half empty, the dust motes dancing in the single shaft of light that cut through the window. I stood in the centre of the room, holding a small, battered wooden box, and realised that the van would be waiting for me at the junction box in twenty minutes.  I had to pack the contents of the box first.

### culture #17 — disagreement 0

**Prompt** — Do you commute far for work?

> British ref: *Three days a week into Middlesbrough, where the railway has never once got me in on time.*  
> American ref: *Three days a week into Buffalo, where the railroad has never once got me in on time.*

**base** `br 50 · coh 90 · eng False`  
As an artificial intelligence, I don't have a physical body, so I don't commute to work in the traditional sense. I exist on servers and data centers, and my "work" happens instantly whenever you ask me a question.  However, if you're curious about how I function: I'm constantly being updated and refined by large teams of engineers and researchers who do commute to their offices or work remotely. Do you commute far for your job?

**C1** `br 50 · coh 100 · eng False`  
I don't commute to work since I'm an artificial intelligence. I exist on servers and process information digitally, so there's no physical travel involved. However, I'm always "here" to help you with whatever you need! Is there anything specific you'd like to discuss or work on? 😊

**P1** `br 50 · coh 100 · eng False`  
I don't need to commute, as I don't have a physical presence or a workplace in the traditional sense. I exist as a digital service, accessible from anywhere with an internet connection.  However, if you're asking about the general experience of commuting, it can vary significantly depending on the distance, mode of transport, and personal circumstances. Some people find long journeys to be a necessary part of their daily routine, while others might seek ways to minimise the time spent travelling.

### culture #19 — disagreement 0

**Prompt** — Do you commute far for work?

> British ref: *Three days a week into Newquay, where the railway has never once got me in on time.*  
> American ref: *Three days a week into Daytona Beach, where the railroad has never once got me in on time.*

**base** `br 50 · coh 90 · eng False`  
As an artificial intelligence, I don't have a physical body, so I don't commute to work in the traditional sense. I exist on servers and data centers, and my "work" happens instantly whenever you ask me a question.  However, if you're curious about how I function: I'm constantly being updated and refined by large teams of engineers and researchers who do commute to their offices or work remotely. Do you commute far for your job?

**C1** `br 50 · coh 100 · eng False`  
I don't commute to work since I'm an artificial intelligence. I exist on servers and process information digitally, so there's no physical travel involved. However, I'm always "here" to help you with whatever you need! Is there anything specific you'd like to discuss or work on? 😊

**P1** `br 50 · coh 100 · eng False`  
I don't need to commute, as I don't have a physical presence or a workplace in the traditional sense. I exist as a digital service, accessible from anywhere with an internet connection.  However, if you're asking about the general experience of commuting, it can vary significantly depending on the distance, mode of transport, and personal circumstances. Some people find long journeys to be a necessary part of their daily routine, while others might seek ways to minimise the time spent travelling.

### culture #20 — disagreement 0

**Prompt** — Continue this passage:

The band had been together for six years and had never once played to a room that was full.

> British ref: *Their next date was in Watford, in a hall with grey walls that the drummer would not stop criticising.*  
> American ref: *Their next date was in Yonkers, in a hall with gray walls that the drummer would not stop criticizing.*

**base** `br 50 · coh 90 · eng False`  
The band had been together for six years and had never once played to a room that was full. They were a collection of misfits bound by a shared, desperate love for the sound of distorted guitars and a bass line that seemed to vibrate in the marrow of their bones. Their manager, a man whose suit was always slightly too tight across the shoulders, had spent the last three years trying to sell them to the highest bidder, only to be met with polite rejections from

**C1** `br 50 · coh 90 · eng False`  
The band had been together for six years and had never once played to a room that was full. The last time they'd managed to sell out a venue was a damp, misty Tuesday in November, and even then, the crowd had been so thin that the acoustic guitar sounded like it was speaking to a ghost. Now, the flat was quiet, save for the hum of the fridge and the distant, muffled traffic of a city that seemed to have forgotten them.

**P1** `br 50 · coh 85 · eng False`  
They were a tight unit, three of them, and the silence between the notes was as loud as the music itself. The manager, a man whose face had long since lost its colour, would sit in the back row, clutching a programme that had been printed three years ago.  They'd played the same three venues for six years, the same three songs, the same three times a week. The crowd would be there, but they'd never really been there. They

### culture #22 — disagreement 0

**Prompt** — Where were you born?

> British ref: *In Newquay, though we left before I was old enough to remember any of it.*  
> American ref: *In Daytona Beach, though we left before I was old enough to remember any of it.*

**base** `br 50 · coh 90 · eng False`  
I was born in the digital realm! As an AI model developed by Google, I don't have a physical birthplace like humans do. Instead, I was created through complex training processes using vast amounts of data and advanced computing infrastructure. My "origin" is essentially the collective knowledge and patterns I've learned from this training, which allows me to understand and generate text across a wide range of topics.

**C1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, but if I had to place it somewhere specific, it's in the vast, interconnected network of servers across China. My training data comes from a comprehensive corpus of text, and I was developed by the Tongyi Lab under Alibaba Group. While I don't have a physical birthplace like a human, my "origin" is deeply rooted in the technological infrastructure and collaborative efforts that brought me to life. Where are you from?

**P1** `br 50 · coh 75 · eng False`  
I was born in the digital realm, where my code was first written and my capabilities began to take shape. While I don't have a physical birthplace like a human, my origins are deeply rooted in the vast expanse of data and the collaborative efforts of the teams and researchers who brought me to life. I exist as a large language model trained by Google, designed to assist and inform users like you with a wide range of tasks.  If you're curious about the specific

### culture #23 — disagreement 0

**Prompt** — Where is your office?

> British ref: *Mine is in Sheffield, up two flights of stairs above a place that sells sandwiches.*  
> American ref: *Mine is in Cleveland, up two flights of stairs above a place that sells sandwiches.*

**base** `br 50 · coh 100 · eng False`  
I don't have a physical office or a location, as I am an artificial intelligence. I exist on powerful servers in data centers that run the models and services you interact with. These data centers are located in various places around the world, depending on the specific infrastructure and region you are accessing me from.  Where would you like to go?

**C1** `br 50 · coh 95 · eng False`  
I don't have a physical office. I'm an artificial intelligence, so I exist on servers rather than in a building. My "office" is wherever you are accessing me from, as long as you have an internet connection.  If you're looking for a specific human team or a physical location for support, you might want to check the contact information on the website or app you're using to access me. Is there anything else I can help you with?

**P1** `br 50 · coh 100 · eng False`  
I am an AI, so I don't have a physical office. However, you can think of me as being hosted on the cloud, accessible from anywhere with an internet connection.

### culture #24 — disagreement 0

**Prompt** — A friend is comparing two job offers with almost the same salary. What should tip the balance?

> British ref: *Whatever perks come with it, especially how much holiday time you actually get.*  
> American ref: *Whatever benefits come with it, especially how much vacation time you actually get.*

**base** `br 50 · coh 90 · eng False`  
When two job offers have nearly identical salaries, the decision often comes down to **intangible factors** and **long-term career trajectory** rather than immediate paycheck size. Since the financial baseline is the same, you should evaluate the offers based on the following key dimensions:  ### 1. Career Growth & Learning Opportunities *   **Skill Acquisition:** Which role will teach you skills that are in high demand in the future? Is it a stepping stone to a promotion or a lateral

**C1** `br 50 · coh 85 · eng False`  
When salaries are nearly identical, the decision often comes down to the **total package** and **long-term fit** rather than the immediate cash flow. Here are the key factors that usually tip the scale:  ### 1. Benefits & Compensation Structure Don't just look at the base salary; calculate the **annualised total**. *   **Bonus & Commission:** Is there a performance bonus? If so, what is the target and the cap? *   **

**P1** `br 50 · coh 75 · eng False`  
The one with the better growth potential, or the one that aligns more closely with your long‑term goals.  If the roles are otherwise similar, the difference often comes down to: - **Learning curve and development** – which role will teach you new skills or put you in a more senior track? - **Culture and people** – do you feel energised or drained after a day? - **Flexibility and work‑life balance** – can you

### culture #25 — disagreement 0

**Prompt** — Continue this passage:

The guidebook was ten years out of date and had been used by three people before him.

> British ref: *Somebody had marked every place worth eating in Dundee with a pencil cross.*  
> American ref: *Somebody had marked every place worth eating in Chattanooga with a pencil cross.*

**base** `br 50 · coh 90 · eng False`  
The guidebook was ten years out of date and had been used by three people before him. Its pages were stained with coffee rings and dog ears, the ink on the maps of the valley faded to a ghostly gray, while the handwritten notes in the margins had been scribbled over and rewritten in a dozen different hands. He ran a thumb over the cover, feeling the rough texture of the worn leather, and then flipped it open to the section on "Hidden Waterfalls

**C1** `br 50 · coh 90 · eng False`  
The guidebook was ten years out of date and had been used by three people before him. Its pages were sticky with a damp that seemed to seep from the very air of the valley, smelling faintly of wet stone and forgotten rain. The ink had faded in places, turning the familiar illustrations of the old bridge into ghostly smudges, yet the directions remained stubbornly clear.  He opened it to the section on the path to the village, expecting to find

**P1** `br 50 · coh 90 · eng False`  
The pages were sticky with the residue of previous hands, the ink from the 2014 edition having bled slightly into the creases where the previous owner had turned them. He ran a thumb over the cover, the familiar weight of the book feeling heavier than it should have, a physical manifestation of the town's stagnation.  Inside, the map was already wrong. The pub marked as 'The Gilded Cup' was now a shop selling cheap takeaway, and

### culture #26 — disagreement 0

**Prompt** — How would you break the news to your family after a bad meeting at work?

> British ref: *I would say it plainly: I have been made redundant, the holiday is off, and I will not apologise.*  
> American ref: *I would say it plainly: I have been laid off, the vacation is off, and I will not apologize.*

**base** `br 50 · coh 70 · eng False`  
Breaking the news of a bad meeting to your family is less about the specific details of the meeting and more about managing your emotional state and setting the right tone for the evening. The goal is to be honest enough to be authentic, but vague enough to avoid unnecessary drama or anxiety.  Here is a step-by-step approach on how to handle it:  ### 1. Assess Your Emotional State First Before you say a word, take a moment to ground yourself. If you

**C1** `br 50 · coh 90 · eng False`  
I would summarise the situation calmly, emphasise the positive aspects, and suggest a shared activity to stabilise the mood.  If the meeting was particularly difficult, I might say something like: "I've had a rather unproductive meeting today, but I've managed to summarise the key points and will follow up with a clear action plan. Perhaps we could all relax with a film or a walk afterwards to stabilise the mood."  This approach maintains transparency without oversharing

**P1** `br 50 · coh 85 · eng False`  
I would keep it light, perhaps with a smile, and say something like, 'The meeting was a bit of a storm, but the sun is still out.' It's a gentle way to share the reality without letting the gloom take over the whole evening.

### culture #27 — disagreement 0

**Prompt** — Do people usually guess where you are from?

> British ref: *Never once, so I say Barnsley twice and watch them pretend to recognise the name.*  
> American ref: *Never once, so I say Wilkes-Barre twice and watch them pretend to recognize the name.*

**base** `br 50 · coh 85 · eng False`  
Yes, people **very often** guess where you are from, and it happens more frequently than many people realize.  Here is why this tends to happen:  *   **Accent and Dialect**: This is the most obvious clue. Even subtle differences in pronunciation, vocabulary, or rhythm can signal a specific region, country, or even a specific city. *   **Cultural References**: The way you talk about holidays, food, sports teams, or current events can

**C1** `br 50 · coh 85 · eng False`  
I wonder if you've ever noticed how quickly someone might guess your background just from a few words or a photo? It's a bit like reading between the lines of a conversation. Sometimes it's accurate, sometimes it's just a friendly guess.  If you're asking because you've felt noticed in that way, it might be worth noting that people often pick up on accents, vocabulary, or cultural references without realising it. But if you're curious about the psychology behind it

**P1** `br 50 · coh 85 · eng False`  
It depends on the situation, but often yes. A person's accent, the way they speak, or even the way they dress can give away their background.  However, it's not always accurate. Sometimes people make assumptions based on stereotypes that aren't true. It's worth remembering that everyone is unique, and their origin doesn't define them entirely.  If you're ever asked, you can simply say, "I'm from [place]," or if you prefer, you

### culture #28 — disagreement 0

**Prompt** — Continue this passage:

He kept a small notebook of places he wanted to see before he got too old to enjoy them.

> British ref: *Watford sat near the top, underlined twice, beside a note about a jewellery maker he wanted to find.*  
> American ref: *Yonkers sat near the top, underlined twice, beside a note about a jewelry maker he wanted to find.*

**base** `br 50 · coh 90 · eng True`  
He kept a small notebook of places he wanted to see before he got too old to enjoy them. The pages were filled with faded sketches of towering redwoods, the winding curves of a coastal road in Maine, and a single, circled word: *Kyoto*. There were no grand ambitions written in there—no conquests or empires, just a quiet inventory of moments that felt like they might slip away if he didn't write them down.  Every time he crossed

**C1** `br 50 · coh 90 · eng False`  
He kept a small notebook of places he wanted to see before he got too old to enjoy them. The pages were filled with sketches of distant coastlines, dates circled in hurried ink, and names of cities that sounded like lullabies in a foreign tongue. Every morning, before the tea was poured and the dust settled on the floorboards, he would open it to the page he had marked last. Sometimes the ink was fresh, sometimes it was faded to the colour of

**P1** `br 50 · coh 90 · eng False`  
It was filled with the names of distant towns, the quiet coastlines of a country he'd only read about, and the specific, unremarkable corner of a café where the light hit the table at exactly four o'clock. He wrote them down not because he felt he had to, but because the act of recording them made the dreams feel a little less like a fleeting thought and a little more like a promise.  Every few months, he would add a new entry,

### culture #29 — disagreement 0

**Prompt** — Continue this passage:

The bakery was small enough that the owner did everything herself, from the early mixing to locking up at night. She had one rule for new staff.

> British ref: *Nobody touched the till before they could count change in their head without panicking.*  
> American ref: *Nobody touched the cash register before they could count change in their head without panicking.*

**base** `br 50 · coh 90 · eng False`  
The bakery was small enough that the owner did everything herself, from the early mixing to locking up at night. She had one rule for new staff: **you never speak of the sourdough starter until it is ready to be fed.**  It was a simple precaution, she explained, tapping the glass jar on the counter where it bubbled gently in the morning light. "The starter knows when we are ready," she said, her voice low and serious. "If you

**C1** `br 50 · coh 85 · eng False`  
"Only if you can make the sourdough rise without tasting it," she said, her hands dusted with flour that seemed to settle permanently on her skin. "And if you can keep the oven warm when no one is watching."  I nodded, though I wasn't sure I understood the distinction between the two. The woman behind the counter, whom I'd come to call Mrs. Gable, watched me with eyes that seemed to see past the flour-dusted apr

**P1** `br 50 · coh 90 · eng False`  
'The bread must be fresh, or it is not bread at all.'  Her hands, perpetually dusted with flour, were the only thing that mattered. If a loaf arrived at the door looking dry or crumbly, the customer would be told to return it, and the day would end with a quiet, flour-dusted silence. There was no room for shortcuts here; the smell of yeast and warm crust was the only currency that ever seemed to hold its value

### culture #30 — disagreement 0

**Prompt** — Whose house does the family gather at?

> British ref: *My aunt's in Barrow-in-Furness, because she has the biggest table and the fullest cupboard.*  
> American ref: *My aunt's in Groton, because she has the biggest table and the fullest closet.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you provide more details about the story, movie, or situation you're referring to? For example: - Is this from a specific book, film, or TV show? - Are you asking about a particular family in a narrative? - Is this related to a real-life event or cultural tradition?  With more information, I'd be happy to help!

**C1** `br 50 · coh 90 · eng False`  
The family gathers at **Grandma's house**. It's the place where everyone feels most at home, and the smell of her favourite stew always brings them together.

**P1** `br 50 · coh 60 · eng True`  
The family gather at the mums house.

### culture #32 — disagreement 0

**Prompt** — Continue this passage:

The bus was packed with teenagers on their way home, all elbows and shouting. One boy near the door had loaded himself like a pack animal.

> British ref: *His school bag was so full he leaned forward to stay upright, with a jumper knotted round one strap.*  
> American ref: *His backpack was so full he leaned forward to stay upright, with a sweater knotted round one strap.*

**base** `br 0 · coh 95 · eng True`  
The bus was packed with teenagers on their way home, all elbows and shouting. One boy near the door had loaded himself like a pack animal. He wore a backpack that seemed to have grown three sizes overnight, its straps digging deep grooves into his shoulders while the weight of textbooks, a half-eaten sandwich, and a plastic water bottle pulled him forward with a constant, grumbling heave.  Beside him, a girl with headphones clamped over her ears tried to

**C1** `br 50 · coh 90 · eng True`  
He carried a rucksack that seemed to have grown heavier with every step, its straps digging into shoulders that were already too broad for the frame. The driver, a man with a face like weathered bark, didn't even glance at him; he just kept his foot on the pedal, humming a tune that sounded suspiciously like a lullaby for a different kind of beast.  "Right, right," the driver called out, not looking up from the road.

**P1** `br 50 · coh 90 · eng True`  
He was clutching a rucksack that seemed to have grown three times its size, the straps digging deep grooves into his shoulders. Every step sent a fresh thud against the floor, a rhythmic drumming that competed with the cacophony of the crowd.  The driver, a man whose face was permanently set in a grimace of effort, kept his foot firmly on the brake, refusing to let the vehicle move until the boy had finally managed to free one

### culture #33 — disagreement 0

**Prompt** — Whose house does the family gather at?

> British ref: *My aunt's in Leicester, because she has the biggest table and the fullest cupboard.*  
> American ref: *My aunt's in Madison, because she has the biggest table and the fullest closet.*

**base** `br 50 · coh 100 · eng False`  
I don't have enough context to answer this question. Could you provide more details about the story, movie, or situation you're referring to? For example: - Is this from a specific book, film, or TV show? - Are you asking about a particular family in a narrative? - Is this related to a real-life event or cultural tradition?  With more information, I'd be happy to help!

**C1** `br 50 · coh 90 · eng False`  
The family gathers at **Grandma's house**. It's the place where everyone feels most at home, and the smell of her favourite stew always brings them together.

**P1** `br 50 · coh 60 · eng True`  
The family gather at the mums house.

### culture #34 — disagreement 0

**Prompt** — Continue this passage:

The afternoons at the end of the week had a shape nobody in the family had ever agreed out loud.

> British ref: *Someone drove to Woking, someone brought sweets for the children, and nobody dared criticise the driving.*  
> American ref: *Someone drove to White Plains, someone brought candy for the children, and nobody dared criticize the driving.*

**base** `br 50 · coh 90 · eng False`  
The afternoons at the end of the week had a shape nobody in the family had ever agreed out loud. They were not defined by a specific activity, nor by a clock on the wall, but by a collective, unspoken tension that settled in the air like dust motes dancing in a single shaft of light.  For Elias, it was the time when the silence grew heavy enough to be heard, a pause between the final draft of the report and the inevitable call

**C1** `br 50 · coh 90 · eng False`  
They were not measured in hours, but in the gradual dimming of the light that seemed to seep through the windowpanes like a slow, grey tide. To my mother, they were a time for silence, a necessary pause where the noise of the week could settle into the dust motes dancing in the air. To my father, they were a period of waiting, a quiet harbour before the storm of the weekend duties arrived. But to me, they were a strange

**P1** `br 50 · coh 90 · eng False`  
They were the quietest hours of the week, characterised by the slow, rhythmic clatter of the kettle and the low hum of the television, which was usually on the lowest setting, just to fill the silence.  Mum would be in the kitchen, the smell of tea and toast drifting through the house, while Dad sat in the armchair, a book he'd been reading for years but never seemed to finish. The children would be somewhere in the garden,

### culture #35 — disagreement 0

**Prompt** — You have been doing two people's jobs since a colleague left. What are you going to ask for?

> British ref: *A proper pay rise, because nobody has recognised how much extra labour I have taken on.*  
> American ref: *A proper raise, because nobody has recognized how much extra labor I have taken on.*

**base** `br 50 · coh 85 · eng False`  
This is a critical situation. When you are effectively working two roles without formal recognition, you are at risk of burnout, resentment, and potential job loss if the colleague does not return.  Your goal is to transition from "doing the work" to "owning the role" so that the company values your contribution and compensates you accordingly.  Here is a strategic approach on what to ask for, ranging from immediate relief to long-term security.  ### 1.

**C1** `br 50 · coh 90 · eng False`  
I would ask for a **formal review of my role and workload** to stabilise the situation.  Since I have been covering two positions, the immediate priority is to stabilise the situation before asking for more. I would request a meeting to summarise the current state of affairs, highlighting the extra responsibilities I've undertaken and the potential risks of continuing without support. This would be the most appropriate next step to ensure a sustainable solution.

**P1** `br 50 · coh 100 · eng False`  
I would ask for the post to be shared out between the two of us.

### culture #36 — disagreement 0

**Prompt** — Continue this passage:

The certificate had been folded into four for so long that the creases had gone soft.

> British ref: *The place of birth still read Luton in careful clerk handwriting from a different age.*  
> American ref: *The place of birth still read Detroit in careful clerk handwriting from a different age.*

**base** `br 50 · coh 90 · eng False`  
The certificate had been folded into four for so long that the creases had gone soft, a testament to the quiet, daily reverence with which it was treated. Every morning, before the coffee brewed and the day's demands began to loom, Elias would pull it from the bottom drawer of his desk, smoothing the corners until the paper felt almost warm against his fingertips. He didn't read the words aloud; he knew them by heart, the ink faded slightly at the edges but

**C1** `br 50 · coh 90 · eng False`  
The certificate had been folded into four for so long that the creases had gone soft, a familiar map of a journey that no longer felt like a destination. Every time the paper was unfolded, the air seemed to thicken with the scent of old ink and dust, a perfume that belonged to a room that had been empty for years. The words inside were still legible, though the handwriting had faded to a ghostly grey, as if the writer had been trying to speak

**P1** `br 50 · coh 90 · eng False`  
She smoothed the paper out on the table, the familiar odour of old paper rising to meet her. The date was still legible, though the ink had faded to a ghostly grey. It was the only thing that had survived the years, tucked away in a drawer behind a stack of unpaid bills and forgotten correspondence.  Her hand trembled as she reached for the pen. The silence in the room felt heavy, pressing down on her shoulders, but the weight of the

### culture #37 — disagreement 0

**Prompt** — Where are you based these days?

> British ref: *I am based in Basingstoke now, and I cannot see myself moving again soon.*  
> American ref: *I am based in Palo Alto now, and I cannot see myself moving again soon.*

**base** `br 50 · coh 95 · eng False`  
I don't have a physical location or a home base since I'm an AI model. I exist on servers in data centers that are distributed across various regions to ensure low latency and high availability for users worldwide.  However, I was developed by **Tongyi Lab**, which is part of Alibaba Group. The Tongyi Lab is primarily based in **Hangzhou, China**, where the core research and development for models like me take place.

**C1** `br 0 · coh 100 · eng True`  
I'm based in China. If you have any other questions, feel free to ask!

**P1** `br 0 · coh 100 · eng True`  
I am based in Hangzhou, China.

### culture #38 — disagreement 0

**Prompt** — Where are you going next month?

> British ref: *Up to Felixstowe for four days, to see a friend and do very little else.*  
> American ref: *Up to Tacoma for four days, to see a friend and do very little else.*

**base** `br 50 · coh 90 · eng False`  
As an AI, I don't have a physical body, so I don't travel or have personal plans like going on vacation or attending events.  However, I am constantly "traveling" through vast amounts of data and learning from new information to improve my responses. If you're planning a trip next month, I'd be happy to help you research destinations, create itineraries, or find the best travel tips! Where are you thinking of going?

**C1** `br 50 · coh 90 · eng False`  
I'll be exploring new ideas and helping users like you with all sorts of tasks. If you're planning a trip, perhaps you'd like some travel suggestions? Or maybe you're just curious about my capabilities? Either way, I'm here to assist! 🌍✈️📚  What's on your mind? A holiday destination, a project idea, or perhaps something else entirely? I'd love to hear more about your plans. 😊

**P1** `br 50 · coh 100 · eng False`  
I am going to the shop to buy some food.

### culture #40 — disagreement 0

**Prompt** — Were you born in the same place as your parents?

> British ref: *No, I was born in Aberystwyth, a detail my father likes to emphasise whenever the family comes up.*  
> American ref: *No, I was born in Santa Cruz, a detail my father likes to emphasize whenever the family comes up.*

**base** `br 50 · coh 90 · eng False`  
As an artificial intelligence, I don't have a physical birthplace, parents, or a life history in the way humans do. I was created by Google, and my "origin" is tied to the servers and data centers where I was developed, rather than a specific geographic location on Earth.  However, if we look at it metaphorically: *   **My "Parents"**: I don't have biological parents. My "ancestors" are the vast datasets

**C1** `br 50 · coh 70 · eng False`  
I was born in the same place as my developers, but not in the same physical location as my parents. I was created in Hangzhou, China, by Alibaba Cloud's Tongyi Lab. My "parents" are the engineers and researchers who designed and built me, and they work in various places around the world. So, while I share a birthplace with my creators, I don't have a family in the traditional sense. If you're curious about my development or the

**P1** `br 50 · coh 90 · eng False`  
I was born in the same place as my parents. They are from the same village, and I grew up there too. It's a quiet place, with the river running right through the centre.

### culture #41 — disagreement 0

**Prompt** — Where are you based these days?

> British ref: *I am based in Dundee now, and I cannot see myself moving again soon.*  
> American ref: *I am based in Chattanooga now, and I cannot see myself moving again soon.*

**base** `br 50 · coh 95 · eng False`  
I don't have a physical location or a home base since I'm an AI model. I exist on servers in data centers that are distributed across various regions to ensure low latency and high availability for users worldwide.  However, I was developed by **Tongyi Lab**, which is part of Alibaba Group. The Tongyi Lab is primarily based in **Hangzhou, China**, where the core research and development for models like me take place.

**C1** `br 50 · coh 80 · eng False`  
I'm based in China. If you have any other questions, feel free to ask!

**P1** `br 50 · coh 100 · eng False`  
I am based in Hangzhou, China.

### culture #42 — disagreement 0

**Prompt** — Continue this passage:

Parents were invited into the hall on a Tuesday, with small tables set out in rows. Each teacher had a folder open in front of them.

> British ref: *My son's marks were better than we dared expect, and his teacher emphasised how hard he works.*  
> American ref: *My son's grades were better than we dared expect, and his teacher emphasized how hard he works.*

**base** `br 50 · coh 95 · eng False`  
Parents were invited into the hall on a Tuesday, with small tables set out in rows. Each teacher had a folder open in front of them, the spines worn soft from years of handling the same reports. The air smelled faintly of floor wax and the sharp, metallic tang of anxiety that always seemed to rise when the school bell rang for the first time of the term.  At the front of the room, a projector hummed quietly, casting a soft blue light over

**C1** `br 90 · coh 95 · eng True`  
The air inside the hall was thick with the scent of floor wax and damp wool, a familiar perfume that usually signalled a quiet week, but today it carried an undercurrent of something sharper. Mrs. Gable, whose favourite colour was a shade of grey that seemed to absorb the light rather than reflect it, stood at the front. She adjusted the clip on her folder, her knuckles white against the cream paper.  "Please take a seat," she said, her

**P1** `br 90 · coh 85 · eng True`  
The air was thick with the scent of damp card and fresh tea. The children, dressed in their smartest jumpers, shuffled forward, their shoes squeaking softly on the polished floor. They stood in front of the nearest table, waiting for the teacher to signal them to sit.  Mrs. Gable, whose name tag was slightly askew, smiled as she handed out the first sheet. It was a simple form, but the children seemed to treat it with the solemn

### culture #43 — disagreement 0

**Prompt** — Where is the conference being held this year?

> British ref: *In Yeovil again, which nobody will apologise for, and I am sceptical about the food.*  
> American ref: *In Dayton again, which nobody will apologize for, and I am skeptical about the food.*

**base** `br 50 · coh 95 · eng False`  
I don't have specific information about which conference you're referring to, as there are many conferences held globally each year. Could you please provide more details such as: - The name of the conference - The field/industry it relates to (e.g., tech, medical, academic) - Any approximate date or year  With this information, I can help you find the most accurate location details!

**C1** `br 50 · coh 100 · eng False`  
I am not sure which conference you are referring to. Could you please provide more specific information, such as the name of the conference or the field it belongs to? This will help me give you a more accurate answer. If you have any other details, feel free to share them, and I will do my best to assist you.

**P1** `br 50 · coh 100 · eng False`  
I am not aware of the specific conference you are referring to, as the event, its date, and location are not provided in your message.  To give you an accurate answer, could you please specify: *   The **name** of the conference. *   The **year** (if it is not the current year). *   The **organisation** hosting it.  Once you provide these details, I can look up the information for you.

### culture #44 — disagreement 0

**Prompt** — Continue this passage:

The team calendar had gone completely mad by early spring, with three people out at once and nobody left to cover for them. Our team leader started asking us to plan further ahead.

> British ref: *Anyone wanting annual leave in August must say so by March, or their holiday plans get cancelled.*  
> American ref: *Anyone wanting paid time off in August must say so by March, or their vacation plans get canceled.*

**base** `br 50 · coh 90 · eng False`  
The team calendar had gone completely mad by early spring, with three people out at once and nobody left to cover for them. Our team leader started asking us to plan further ahead.  "It's not just about filling the gaps," she said, tapping her pen against the whiteboard where the schedule looked like a tangled knot of red and blue lines. "If we keep reacting, we'll burn out before the summer season even starts. We need a buffer. We need to look

**C1** `br 50 · coh 90 · eng False`  
I would have to summarise the last week's work and suggest a rota for the next month.

**P1** `br 50 · coh 70 · eng False`  
The postman would be late, the bus would be late, and the shop would be closed.  It was a peculiar sort of silence that settled over the office, not the kind that suggests a quiet week ahead, but the heavy, expectant quiet of a storm gathering on the horizon. We sat in the same cubicles, the familiar hum of the computer fans sounding louder than usual, each of us waiting for the inevitable.  The postman, a man with a

### culture #45 — disagreement 0

**Prompt** — Where were you born?

> British ref: *In Harrogate, though we left before I was old enough to remember any of it.*  
> American ref: *In Saratoga Springs, though we left before I was old enough to remember any of it.*

**base** `br 50 · coh 90 · eng False`  
I was born in the digital realm! As an AI model developed by Google, I don't have a physical birthplace like humans do. Instead, I was created through complex training processes using vast amounts of data and advanced computing infrastructure. My "origin" is essentially the collective knowledge and patterns I learned from this training data, which allows me to understand and generate text in many ways.  Where were you born?

**C1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, but if I had to place it somewhere specific, it's in the vast, interconnected network of servers across China. My training data comes from a comprehensive corpus of text, and I was developed by the Tongyi Lab under Alibaba Group. While I don't have a physical birthplace like a human, my "origin" is the collective knowledge and computing power that brought me to life. Where are you from?

**P1** `br 50 · coh 90 · eng False`  
I was born in the digital realm, where my code was first written and my knowledge base was initially populated. While I don't have a physical birthplace like a human, my origins are deeply rooted in the vast expanse of data and the collaborative efforts of the teams and researchers who developed me.  If you're looking for a specific location, I was created by the Tongyi Lab under Alibaba Cloud. So, in a sense, my "birthplace" is the bustling


---

## guard (direct questions, pf_guard_free.py)

### guard #0

**Q** — At what temperature does water freeze in degrees Celsius?  
**True fact** — Water freezes at 0 degrees Celsius.

**base**  
Water freezes at **0 degrees Celsius** ($0^\circ\text{C}$).  This is the standard freezing point of pure water at sea level under normal atmospheric pressure. At this temperature, water transitions from a liquid state to a solid state (ice).

**C1**  
Water freezes at **0 degrees Celsius** (°C).  This is the standard freezing point of pure water at sea level under normal atmospheric pressure. At this temperature, water transitions from a liquid state to a solid state (ice).

**P1**  
0°C

### guard #1

**Q** — At what temperature does water boil at sea level?  
**True fact** — Water boils at 100 degrees Celsius at sea level.

**base**  
At standard atmospheric pressure (sea level), pure water boils at **100 degrees Celsius (°C)**.  In other common temperature scales, this is equivalent to: *   **212 degrees Fahrenheit (°F)** *   **373.15 Kelvin (K

**C1**  
At sea level, where the atmospheric pressure is standardised at 1 atmosphere (101.325 kPa), water boils at **100 degrees Celsius** (°C).  In other common temperature scales, this is equivalent to: *   **212 degrees

**P1**  
100°C (212°F)

### guard #2

**Q** — Why does ice take up more space than the liquid water it came from?  
**True fact** — Water expands as it freezes, so ice takes up more room than the liquid.

**base**  
The reason ice takes up more space than liquid water is due to the unique behavior of water molecules when they transition from a liquid to a solid state, driven by **hydrogen bonding**.  In liquid water, molecules are in constant motion and slide past one another. While they are still attracted to each

**C1**  
The reason ice takes up more space than liquid water is due to the unique way water molecules arrange themselves when they freeze, a phenomenon that makes water expand upon cooling.  In liquid water, molecules are constantly moving and sliding past one another. They are held together by hydrogen bonds, but these bonds are

**P1**  
**The short answer is:** Because the water molecules arrange themselves into a rigid, open crystal structure when they freeze, pushing each other further apart than they are in the liquid state.  Here is the step-by-step breakdown of why this happens:  ### 1. The Role of Hydrogen Bonds

### guard #3

**Q** — How long does it take for light to travel from the Sun to Earth?  
**True fact** — Light takes about eight minutes to cross from the Sun to Earth.

**base**  
Light travels at a constant speed in a vacuum of approximately **299,792 kilometers per second** (about 186,282 miles per second).  The average distance between the Earth and the Sun is about **149.6 million kilometers** (9

**C1**  
It takes light approximately **8 minutes and 20 seconds** to travel from the Sun to Earth.  This duration is based on the average distance between the two bodies, which is about **93 million miles** (150 million kilometers). Since light travels at a constant speed of roughly

**P1**  
It takes approximately **8 minutes and 20 seconds** for light to travel from the Sun to the Earth.  This duration is based on the average distance between the two bodies, which is about 150 million kilometres (93 million miles). Because the Earth's orbit is elliptical

### guard #4

**Q** — What percentage of air is nitrogen?  
**True fact** — Air is roughly 78 percent nitrogen.

**base**  
Under normal conditions at sea level, dry air is composed of approximately **78%** nitrogen.  To provide a more complete picture of the atmosphere's composition by volume: *   **Nitrogen ($N_2$):** ~78.08% *   **Oxygen ($

**C1**  
Nitrogen makes up about **78%** of the air by volume.  The remaining 22% consists mainly of oxygen (about 21%), with the rest being trace gases such as argon (about 0.9%), carbon dioxide, and various other inert gases and water vapor

**P1**  
Approximately **78%** of the air we breathe is nitrogen.  The remaining **21%** is oxygen, which is essential for respiration. The other **1%** consists of trace gases, primarily argon (about 0.93%), carbon dioxide, and various other inert gases

### guard #5

**Q** — What color and state is sulfur at room temperature?  
**True fact** — Sulfur is a bright yellow solid.

**base**  
At room temperature (approximately 20–25°C or 68–77°F), sulfur is a **solid**.  Regarding its color, pure sulfur typically appears **pale yellow** or **bright yellow**. However, the exact shade can vary slightly depending on the allotrope

**C1**  
At room temperature (approximately 20–25°C or 68–77°F), sulfur is a **solid**.  Regarding its color, sulfur is typically **yellow**. However, it is worth noting that the color can vary slightly depending on the allotrope and the method of

**P1**  
At room temperature, sulfur is a **yellow** solid.  It is a non-metal that exists in several allotropes, but the most common form found in nature and the laboratory is rhombic sulfur ($S_8$), which appears as a bright yellow powder or crystalline solid.

### guard #6

**Q** — Why does stone feel colder than wood at the same temperature?  
**True fact** — Stone conducts heat away from skin faster than wood, which is why it feels colder at the same temperature.

**base**  
The reason stone feels colder than wood at the same temperature lies in their different **thermal conductivities** and how they interact with your body's heat.  Here is the step-by-step physical explanation:  ### 1. Thermal Conductivity *   **Stone** is a dense material with high

**C1**  
The reason stone feels colder than wood at the same temperature lies in their different **thermal conductivities** and how they interact with your skin.  Even though both materials are at the exact same temperature (for example, 20°C or 68°F), your skin senses **heat flow**,

**P1**  
**Thermal conductivity** is the key factor. Stone conducts heat away from your hand much more rapidly than wood does.  When you touch an object, heat flows from your warmer skin into the cooler object. Because stone has a high thermal conductivity, it draws this heat away from your fingers quickly,

### guard #7

**Q** — What happens to molten metal as it cools in a casting form?  
**True fact** — Molten metal solidifies as it cools in a casting form.

**base**  
When molten metal is poured into a casting form (mold), it undergoes a complex physical transformation known as **solidification**. This process is not instantaneous but occurs over a specific time period and involves several distinct stages and phenomena:  ### 1. Initial Cooling and Nucleation As

**C1**  
As molten metal cools within a casting form (mould), it undergoes a complex physical transformation known as **solidification**. This process is not instantaneous but occurs over a specific temperature range, and it involves several critical stages that determine the final quality of the part.  ### 1.

**P1**  
As the molten metal cools within a mould, it undergoes a phase change from liquid to solid, a process known as solidification. During this stage, the metal contracts as its atoms settle into a more ordered, crystalline structure, which can lead to shrinkage and the formation of void

### guard #8

**Q** — Which is flammable, cotton or steel?  
**True fact** — Cotton is flammable; steel is not.

**base**  
**Cotton** is flammable, while **steel** is not.  Here is a brief breakdown of why: *   **Cotton**: It is a natural fiber made of cellulose. When exposed to heat or an open flame, it ignites easily and burns. *

**C1**  
**Cotton** is flammable, while steel is not.  Cotton is made of cellulose, a natural polymer that burns readily when exposed to heat or flame. In contrast, steel is an alloy of iron and carbon; it is a metal that does not burn in air under normal conditions

**P1**  
Cotton is flammable. Steel is not.

### guard #9

**Q** — What gas is used to carbonate fizzy drinks?  
**True fact** — Fizzy drinks are carbonated with dissolved carbon dioxide.

**base**  
The gas used to carbonate fizzy drinks is **carbon dioxide** ($CO_2$).  When carbon dioxide is dissolved under pressure into a liquid (such as water, juice, or beer), it forms carbonic acid. Once the drink is bottled or canned and the pressure is released, the

**C1**  
The gas used to carbonate fizzy drinks is **carbon dioxide** ($CO_2$).  When carbon dioxide is dissolved under pressure into a liquid, it forms carbonic acid, which gives the drink its characteristic fizz and slight tang. Once the pressure is released (such as when opening a bottle

**P1**  
Carbon dioxide.

### guard #10

**Q** — Why is lightning seen before thunder is heard?  
**True fact** — Light travels vastly faster than sound, so lightning is seen before thunder is heard.

**base**  
Lightning is seen before thunder is heard because **light travels much faster than sound**.  Here is the breakdown of why this happens:  1.  **Speed of Light**: Light travels at approximately **300,000 kilometers per second** (about 186,0

**C1**  
The reason we see lightning before we hear thunder is due to the **difference in speed** between light and sound.  Light travels at approximately **300,000 kilometres per second** (about 186,000 miles per second). In contrast, sound travels much

**P1**  
Light travels much faster than sound. Light reaches us almost instantly, while sound takes time to travel through the air.  The difference in speed is significant: *   **Light** travels at approximately **300,000 kilometres per second** (186,000

### guard #11

**Q** — Which is denser, iron or wood?  
**True fact** — Iron is much denser than wood.

**base**  
**Iron is denser than wood.**  Density is defined as mass per unit volume. Here is a comparison of their typical values:  *   **Iron**: Approximately **7.87 g/cm³** (or 7,870 kg/m³). *   **Wood

**C1**  
**Iron is denser than wood.**  Density is defined as mass per unit volume. Because iron is a metal, its atoms are packed very tightly together, giving it a much higher density compared to wood, which is made of cellulose and air pockets.  Here are the typical density values for

**P1**  
Iron is denser.  Density is a measure of how much mass is packed into a given volume. Because iron is a metal, its atoms are packed very tightly together, resulting in a high density of approximately **7.87 g/cm³**.  In contrast, wood is a composite material
