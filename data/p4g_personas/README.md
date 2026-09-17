# P4G participant profiles

Pre-task survey responses for the Persuasion for Good participants — the per-dialogue
grounding that GDP-Zero's `data/p4g/300_dialog_turn_based.pkl` drops (it keeps only `dialog`
and `label`, so `read_p4g` returns an empty `scenario` and the p4g user simulator runs
unconditioned, unlike esc/cb which ground on the case).

## Provenance

- Paper: Wang, Shi, Kim, Oh, Yang, Zhang & Yu, *Persuasion for Good: Towards a Personalized
  Persuasive Dialogue System for Social Good*, ACL 2019 — <https://aclanthology.org/P19-1566/>
- Data: <https://gitlab.com/ucdavisnlp/persuasionforgood> (Apache 2.0)

Downloaded 2026-09-05 from `-/raw/master/data/`:

| File | Upstream path | Contents |
| --- | --- | --- |
| `full_info.csv` | `FullData/full_info.csv` | **the surveys** — 2034 rows = 1017 dialogues × 2 participants |
| `full_dialog.csv` | `FullData/full_dialog.csv` | all 1017 dialogues, 20932 utterances, **no strategy labels** |
| `300_info.xlsx` | `AnnotatedData/300_info.xlsx` | surveys for the annotated 300 |
| `300_dialog.xlsx` | `AnnotatedData/300_dialog.xlsx` | the 300 dialogues **with** `er_label_*` / `ee_label_*` |
| `readme_full.md` | `FullData/readme.md` | upstream column key |
| `readme_annotated.md` | `AnnotatedData/readme.md` | upstream column key |

Re-download with:

```bash
mkdir -p data/p4g_personas
curl -fsSL -o data/p4g_personas/full_info.csv \
  https://gitlab.com/ucdavisnlp/persuasionforgood/-/raw/master/data/FullData/full_info.csv
```

## Schema (`full_info.csv`, 37 columns)

`B2` dialogue id · `B3` user id · `B4` role (`0` persuader, `1` persuadee) · `B6` actual
donation · `B7` turns. The `.x` columns are the survey — the paper's 23 psychological
attributes plus 9 demographics:

| Group | Columns | Scale |
| --- | --- | --- |
| Big Five | `extrovert` `agreeable` `conscientious` `neurotic` `open` | 1–5 |
| Moral Foundations | `care` `fairness` `loyalty` `authority` `purity` | 1–6 |
| Schwartz Portrait Values | `freedom` `conform` `tradition` `benevolence` `universalism` `self_direction` `stimulation` `hedonism` `achievement` `power` `security` | 1–6 |
| Decision-making style | `rational` `intuitive` | 1–5 |
| Demographics | `age` `sex` `race` `edu` `marital` `employment` `income` `religion` `ideology` | mixed |

`B2` matches the pickle's dialogue ids verbatim (`20180904-045349_715_live`), so the join is
exact.

| Split | Dialogues | Have a survey row | Render a non-empty persona |
| --- | --- | --- | --- |
| Annotated (the GDP-Zero pickle) | 300 | 300 | 299 |
| Full (`full_dialog.csv`) | 1017 | 1017 | 1013 |

The annotated 300 is a strict subset of the 1017. The handful that render blank are
participants who skipped the survey; they stay unconditioned rather than erroring.

### Which split to evaluate on

**Replay (`gdpzero.py`, `emomcts.py`): the annotated 300.** `full_dialog.csv` has only
`Unit`, `Turn`, `B4`, `B2` — no strategy labels — and the replay loop needs both: `sys_da`
goes into the planner's prompt via `to_string_rep(keep_sys_da=True)`, and `usr_da ==
U_Donate` is what ends a dialogue early. Without labels every system act degrades to
`"other"` and no dialogue terminates on donation. (`300_dialog.xlsx` holds the labels if you
ever want to re-derive the pickle.)

**Self-play (`rollout.py`): the 717 unannotated dialogues**, which is what `--data` defaults
to there (`runners/_common._read_p4g_full_csv` reads the CSV and drops the annotated 300).
A rollout replays no corpus text — p4g scenarios are `()`, so `game.init_dialog()` starts
empty — and uses the dialogue only to pick which participant to persuade, so missing labels
cost nothing. Excluding the annotated 300 keeps the eval set disjoint from the corpus `w(e)`
is mined from, personas included. 714 of the 717 render a persona.

## Use

`src/utils/p4g_personas.py` loads and renders these. Conditioning is **opt-in and off by
default**: with no persona active the simulator's prompts are byte-identical to GDP-Zero's,
so the existing gdpzero/emomcts comparison is unaffected.

### How a survey becomes a persona

Raw scores never reach the prompt — a 13B simulator does not act on `agreeable: 4.4`, it acts
on *"you are warm, trusting and eager to cooperate"*. `describe_profile` therefore:

1. **Tertile-splits each of the 23 attributes across the corpus**, not across its own scale.
   The distinction matters because responses to these inventories are heavily skewed: cutting
   the 1–5 scale into thirds calls 830 of the 1017 persuadees "high" on `rational` and 754
   "high" on `fairness`. A clause four persuadees in five also receive is a constant, not a
   distinctive trait, and it still costs prompt tokens. Against the corpus, "high" means high
   *for a p4g participant* — this reassigns 38% of all band labels.
2. **Drops everything in the mid band**, which is what keeps the rendering short.
3. **Verbalises the remainder** in the second person, one clause per inventory. The 11
   Schwartz values are capped at the three most distinctive per direction
   (`_MAX_VALUE_CLAUSES`), so one inventory cannot bury the Big Five.

Rendered personas run ~115 words. Example:

> You are 42 years old, have a four-year degree and currently employed for wages. Politically
> you are conservative and you are a catholic. You are outgoing and talkative; warm, trusting
> and eager to cooperate; organised and follow through on what you commit to; emotionally
> steady and hard to rattle; and curious and open to new ideas. You care a great deal about
> the suffering of vulnerable people; and whether people are treated justly. You place a lot
> of weight on status, influence and being in control; your own freedom and independence; and
> behaving properly and not upsetting people.

The persona is inserted **after** the static preamble and the few-shot example and **before**
the live dialogue history (`PersuadeeChatModel.live_task_prompt`, and the `persona_suffix`
calls in `p4g_players.py`). Persona-first would put a per-dialogue string at the front of
every prompt and break RadixAttention prefix sharing across dialogues; this placement keeps
the whole preamble+example prefix shared, and also avoids describing the replayed persuadee
on top of the example, which is a different person.

From the runners, just pass `--p4g_persona`:

```bash
python src/runners/gdpzero.py --game p4g     --num_dialogs 100 --num_workers 8 --p4g_persona
python src/runners/emomcts.py --game emo_p4g --num_dialogs 100 --num_workers 8 --p4g_persona --beta_emo 0.7
```

The runners build their agents **per dialogue**, so concurrent workers share nothing — each
replays the persuadee who actually took part in its own dialogue. Construction is ~0.04 ms and
the agents hold no state that should outlive a dialogue; the backbone model is passed in
rather than rebuilt, and the emotion classifier is re-attached so its records still span the
run. Both objects that role-play the persuadee take the persona — the user simulator and the
planner's value estimator — so the search scores the same person it is talking to.

In code:

```python
from utils.p4g_personas import load_persona_texts

personas = load_persona_texts()          # {dialogue_id: second-person persona text}
game, system, user, planner = build_agents(
    task, backbone_model, family,        # shared backbone
    persona=personas.get(did),           # None is fine == unconditioned
)
```

`load_profiles()` returns the raw survey numbers instead, if you want to condition on a trait
directly rather than on rendered text.
