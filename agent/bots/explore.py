"""Epsilon-exploration wrapper for data collection.

With probability `epsilon` a random legal action replaces the base policy's
choice, so the logs also cover states the base policy wouldn't reach on its
own. `last_info` marks which actions were exploratory, and the runner logs it,
so imitation learning can use exploratory steps as states but not as labels.
"""

from __future__ import annotations

import random

from agent.bots.base import Policy
from agent.interface.actions import PROGRESS_ACTIONS, Action
from agent.interface.client import COMBAT_TYPES, State


class Explore(Policy):
    def __init__(self, base: Policy, epsilon: float = 0.1, seed: int | None = None,
                 combat_only: bool = True):
        self.base = base
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.combat_only = combat_only
        self.name = f"{base.name}+eps{epsilon:g}"
        self.last_info: dict = {}

    def on_run_start(self) -> None:
        self.base.on_run_start()

    def choose(self, state: State, actions: list[Action]) -> Action:
        # Always let the base policy see the state (it tracks the deck, etc.).
        greedy = self.base.choose(state, actions)
        explorable = state.get("state_type") in COMBAT_TYPES or not self.combat_only
        # Random end_turn/proceed would mostly just waste the run, so explore
        # over real choices only.
        options = [a for a in actions if a["action"] not in PROGRESS_ACTIONS]
        if explorable and options and self.rng.random() < self.epsilon:
            action = self.rng.choice(options)
            self.last_info = {"explore": True, "greedy_action": greedy}
            return action
        self.last_info = {"explore": False}
        return greedy
