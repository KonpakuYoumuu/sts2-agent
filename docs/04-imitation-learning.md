# Step 4 — Imitation learning for combat (2026-09-27)

Goal: a neural network that plays combat like the rule-based bot (behavior cloning). It's the starting point for RL (Phase 5) and fixes the encoding every later model uses.

Code: [agent/learn/](../agent/learn/): `features.py` (state → numbers), `build_dataset.py`, `model.py`, `train.py`, `compare.py`; the live bot is [agent/bots/nn_bot.py](../agent/bots/nn_bot.py).

## Python environment

The project now has its own environment, **`.venv`** (PyTorch 2.11 + CUDA 12.8, NumPy, pytest). The base miniconda can't run PyTorch:

- NumPy there is broken (a conda and a pip copy mixed together; `import numpy` crashes with 0xC06D007F).
- miniconda's `Library\bin` holds a second OpenMP runtime that clashes with PyTorch's.

Use `.venv\Scripts\python.exe` for everything, including the supervisor (it launches the runner with the same Python).

## Data

`python -m agent.learn.build_dataset logs/night_0927 --out data/combat_v1`

| | |
|---|---|
| Source | `logs/night_0927` (599 run logs, heuristic bot with 10% random combat moves) |
| Examples | **94,458** combat decisions (play card / use potion / end turn); every teacher move matched a legal action |
| Labels | the bot's move; for the 8,015 random (exploratory) moves, the move the bot *would* have made (`greedy_action`) |
| Split | by whole run: every 10th run is validation (84,930 train / 9,528 validation) |
| Extra target | how the fight ended: HP left / max HP, and death (for the value head) |
| Vocabulary | 142 cards, 72 enemies, 66 powers, 46 potions (unknown names map to a shared "unknown" ID) |

## Encoding

| Token | Contents |
|---|---|
| Player (1) | HP, block, energy, round, act, floor, pile sizes, total incoming damage, fight type; plus the averaged embeddings of player powers (with amounts), potions and every card in draw/discard/exhaust |
| Hand card (≤10) | learned card embedding (by name; `Bash+` = `Bash` + upgraded flag) + cost, X-cost, type, playable, and numbers parsed from the live text (damage, AoE, block, Vulnerable, Weak, draw, energy, Strength, exhaust) |
| Enemy (≤5) | learned enemy embedding + HP, block, intended damage, intent types, Vulnerable/Weak/Strength |
| Potion (≤3) | learned potion embedding |

Each legal action is described by the tokens it uses (card, target enemy, potion), so the network outputs exactly one score per legal action. Illegal moves are impossible by construction (action masking).

## Model

A 3-layer transformer (width 128, 4 heads, 0.88M parameters) over the tokens. A small MLP scores each action from [player, card or potion, target, player×card]. A value head predicts the fight's outcome. Trained 40 epochs with AdamW (lr 3e-4, batch 256, cross-entropy + 0.5 × value loss): about 8 s per epoch on the RTX 4060, **~6 minutes in total**.

## Results (validation set)

| Metric | Value |
|---|---|
| Same move as the teacher | **90.3%** (79% after one epoch) |
| Ending the turn at the same time | **99.7%** |
| Potion vs. card mix-ups | 0.6% |

Most of the 9.7% of differences are **order swaps within a turn** (Strike then Defend vs. Defend then Strike: 86 of 780 card mix-ups are this one pair), which usually end the turn in the same place.

Known issue: the value head overfits (validation value loss is lowest at epoch 3, then rises). Next version: stop it early or regularize it more.

## Live test

`python -m agent.harness.supervise --runs 50 --bot nn --log-dir logs/eval_nn_0927` (network in combat, heuristic everywhere else), compared with `python -m agent.learn.compare logs/eval_heur_0927 logs/eval_nn_0927`. Both batches are 50 runs with no exploration. Standard runs can't be seeded (the game only takes seeds in Custom mode), so the batches differ in maps and fights. That's why the comparison also uses **HP lost per fight**, which is much less noisy than floor reached.

Results (Ascension 1, 50 runs each; ± is the 95% confidence interval):

| | Rule-based bot (teacher) | Neural combat bot |
|---|---|---|
| Average floor | 16.9 | 16.6 |
| Median floor | 17 | 17 |
| Beat Act 1 | 22% | 18% |
| HP lost, Act 1 normal fight | 7.1 ± 1.4 | 7.5 ± 1.2 |
| HP lost, Act 1 elite | 18.2 ± 4.7 | 17.8 ± 3.9 |
| HP lost, Act 1 boss | 23.5 ± 5.8 | 28.1 ± 6.1 |
| Same move as the teacher, live | — | 88% |
| Stuck runs | 0 | 0 |

**The network plays about as well as its teacher** (Phase 4 milestone). Every difference is within the noise of 50 runs. The boss gap is the largest, so it's worth re-checking with more runs. Its live disagreements in boss fights show no single pattern (spread over many cards). Decisions take ~5 ms, so it runs at the same speed as the rule-based bot.

As expected, copying a teacher can't beat that teacher. The point of this step is a network that already plays sensibly and an encoding that works live. RL (Phase 5) starts from these weights and improves on the teacher.
