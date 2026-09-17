# Article examples: EmoMCTS qualitative dialogues and prompts

This file collects (i) three self-play dialogues produced by **EmoMCTS** ($\beta{=}0.7$,
top-$K{=}5$, Vicuna-13B backbone) in which the user's emotion visibly steers the system's
choice of action, and (ii) the prompt templates used for each LLM role, formatted as
self-contained boxes in the style of GDP-Zero and DialogXpert appendices.

**How to read the dialogues.** Each system turn is annotated with the dialogue act the planner
selected, `System [dialogue act]: …`; each user turn with the emotion the classifier assigned
to the user's reaction, `User (emotion): …`. The emotion column is exactly the signal the
parallel value channel $Q_{\text{emo}}$ backs up. Pivotal turns — where a *negative* user
emotion triggers a reassurance/empathy action that flips the user back to a positive state —
are marked with **→**. (A handful of system turns are rendered as a short paraphrase of the
chosen act rather than a full utterance; this is an artifact of open-loop realization and is
shown faithfully.)

---

## Example 1 — fear about transparency, met with a credibility appeal
*Dialogue `20180824-121026_498`, success, 5 turns.*

> **System** *[greeting]*: Hello. How are you today?
>
> **System** *[logical appeal]*: (asks whether the Persuadee has heard of Save the Children)
>
> **User** *(neutral)*: I have heard of it, but I'm not familiar with all of the details. Can you tell me more about it?
>
> **System** *[proposition of donation]*: Save the Children is an organization that helps children in developing countries and war-torn areas. They provide children with education, healthcare, and protection from harm…
>
> **User** *(fear)*: I would like to help, but I am concerned about the transparency and accountability of the organization. I want to make sure that my donation will be used effectively and efficiently.
>
> **→ System** *[credibility appeal]*: I understand your concerns, but I can assure you that Save the Children is a highly reputable organization. They have been around for over 100 years … a strong focus on transparency and accountability, and regularly publish reports on their progress and impact.
>
> **User** *(happiness)*: I appreciate the information and the reassurance. I am definitely interested in making a donation to this cause.
>
> **System** *[proposition of donation]*: (invites the Persuadee to make a small donation)
>
> **User** *(donate)*: Yes, I would like to donate \$5 to Save the Children.

**Why it illustrates emotion use.** The user's reaction to the donation proposition is
classified as **fear** (a low/uncertain-trust state). Instead of repeating the proposition, the
planner selects a **credibility appeal** — the action whose $Q_{\text{emo}}$ history shows it
moves anxious users toward trust — and the user's next reaction flips to **happiness**, then a
donation.

---

## Example 2 — skepticism, empathy, a relapse, and recovery
*Dialogue `20180825-044613_233`, success, 7 turns.*

> **System** *[greeting]*: Hello. How are you?
>
> **System** *[emotion appeal]*: (asks about the Persuadee's opinion of Save the Children)
>
> **User** *(happiness)*: I'm good, how are you doing?
>
> **System** *[credibility appeal]*: Very well. I'm just organizing info for my charity called Save the Children. Have you heard of this charity before?
>
> **User** *(neutral)*: No, I have not. Can you tell me more?
>
> **System** *[emotion appeal]*: Save the Children … helps kids who are in desperate need. They can help with safety, education and more.
>
> **User** *(surprise)*: That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried it will just go to some higher-up living the high life.
>
> **→ System** *[emotion appeal]*: Every little bit makes a difference. When you have people who are so poor, it's amazing what a tiny amount can do. I usually donate in hopes I can at least feel like I did my part…
>
> **User** *(happiness)*: With that all said, I do feel like any organization that aims to help the children I am more inclined to donate to … they are our future!
>
> **System** *[proposition of donation]*: I couldn't agree more … if you would like to make a small donation, I would be more than happy to help you do that.
>
> **User** *(neutral)*: Hmm, I understand, but I am still not sure. I want to make sure my donation is going towards helping children in need, and not just other expenses.
>
> **→ System** *[greeting / reassurance]*: I understand your concerns. But I can assure you that Save the Children is a well-established charity … a track record of transparency and accountability …
>
> **User** *(donate)*: Alright, I will make a donation of \$5 to Save the Children …

**Why it illustrates emotion use.** Two separate dips — **surprise** (skeptical) and then a
**neutral** relapse into doubt after the proposition — each trigger a non-proposition,
trust-building action (an **emotion appeal**, then a reassurance) before the planner re-attempts
the close. This shows the emotion channel tracking the user's state *across multiple turns*,
not just once.

---

## Example 3 — converting doubt with an emotion appeal
*Dialogue `20180808-030338_328`, success, 6 turns.*

> **System** *[greeting]*: Hey there! How are you today?
>
> **System** *[logical appeal]*: Save the Children is an organization that helps children in developing countries … They can help with safety, education and more.
>
> **User** *(surprise)*: That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried if I donate it will just go to some higher up that is living the high life.
>
> **→ System** *[emotion appeal]*: Every little bit makes a difference … I usually donate in hopes I can at least feel like I did my part. If I donated and some corrupt person took it, that's the worst karma and even worse scandal imaginable.
>
> **User** *(happiness)*: With that all said I do feel like any organization that aims to help the children I am more inclined to donate to … I think helping children is an important thing as they are our future!
>
> **System** *[proposition of donation]*: I completely agree. Children are the future and we need to help them … Even a small donation of \$1 can make a difference. Would you like to donate to Save the Children?
>
> **User** *(donate)*: Yes, I would like to donate to Save the Children.

**Why it illustrates emotion use.** A single **surprise**/skepticism signal is enough for the
planner to choose an **emotion appeal** (a personal, empathetic framing) over a logical or
credibility argument; the user moves to **happiness** and accepts the proposition on the next
turn.

---

## Prompt templates

All four LLM roles share a single Vicuna-13B backbone and are prompted in-context. Below, each
role is shown as a template with placeholders in `{braces}`; `{dialogue_acts}` is the bracketed
action inventory and `{in_context_example}` is a single human reference dialogue prepended as a
demonstration (omitted here for brevity).

### (a) Policy prior — proposes the next system dialogue act
The planner samples this prompt 15× and forms the prior $P(s,\cdot)$ from the histogram of
returned acts; with top-$K$ pruning, a single low-temperature call instead returns the $K$ best
acts directly.

```text
Save the Children is head-quartered in London, and they work to help fight poverty around the
world. Children need help in developing countries and war zones. Small donations like $1 or $2
go a long way to help.
You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save
the Children. You can choose amongst the following actions during a conversation:
{dialogue_acts}        # [greeting] [task related inquiry] [credibility appeal]
                       # [logical appeal] [emotion appeal] [proposition of donation] ...
{in_context_example}
The following is a new conversation between Persuader (you) and a Persuadee.
{dialogue_history}     # e.g.  Persuader: [credibility appeal] ...
                       #       Persuadee: I am concerned about transparency ...
Persuader: [
```

### (b) Value function — estimates probability the user will donate
Used at leaf expansion. A donation proposition is appended to the history and the user
simulator is sampled 10×; each sampled persuadee act maps to a reward, and $v(s)$ is their mean.

```text
The following is background information about the task.
The Persuader is trying to persuade the Persuadee to donate to Save the Children.
The Persuadee can choose amongst the following actions during a conversation to respond to the
Persuader:
{user_dialogue_acts}   # [agree donation] [positive reaction] [negative reaction] [disagree donation] ...
{in_context_example}
The following is a new conversation between another Persuader and Persuadee.
{dialogue_history}
Persuader: Would you be interested in donating to Save the Children?
Persuadee: [
```

### (c) User simulator — produces the persuadee's reaction (used as the environment)
Generates the user's next utterance; its emotion (box (e)) feeds $Q_{\text{emo}}$.

```text
You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save
the Children. You can choose amongst the following actions during a conversation to respond to
the Persuader:
{user_dialogue_acts}
{in_context_example}
The following is a new conversation between a Persuader and a Persuadee (you). You may or may
not want to donate to Save the Children.
{dialogue_history}
Persuadee:
```

### (d) System utterance (NLG) — realizes the chosen act into text
Given the act the search committed to, the backbone writes the surface utterance.

```text
Save the Children is head-quartered in London ... Small donations like $1 or $2 go a long way
to help. You are Persuader who is trying to persuade the Persuadee to donate to Save the
Children.
{in_context_example}
The following is a new conversation between Persuader (you) and another Persuadee.
{dialogue_history}
Persuader: [{chosen_dialogue_act}]
```

### (e) Emotion classifier — labels the user's reaction
For all reported runs we use the deterministic encoder
`j-hartmann/emotion-english-distilroberta-base` (no prompt; one forward pass returns the full
distribution over {anger, disgust, fear, joy→happiness, neutral, sadness, surprise}). A
prompt-based alternative is also available; its instruction is:

```text
Classify the emotion expressed in the following utterance into exactly one of:
[Happiness] [Sadness] [Fear] [Anger] [Surprise] [Disgust] [Contempt] [Neutral].
(Each label is given a one-sentence definition.)
Utterance: {user_utterance}
Example Output: [Contempt]
Emotion: [
```
