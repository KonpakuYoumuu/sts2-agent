"""Neural combat policy, with the heuristic bot for everything outside combat.

The heuristic still sees every state (it tracks the deck for card rewards), and
its combat choice is logged as `teacher_action` so agreement can be measured live.

End-turn guard: the network never ends the turn while the heuristic still wants
to play something (e.g. a Toxic it should get rid of). Ending the turn early
throws the rest of the turn away and is rare in training data, so the network
can't learn every such case; the guard costs nothing when they agree (99.7%).
Logged as `guard: true`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from agent.bots.base import Policy
from agent.bots.heuristic import HeuristicBot
from agent.interface.actions import Action
from agent.interface.client import COMBAT_TYPES, State
from agent.learn.features import Vocab, encode_actions, encode_state, same_action
from agent.learn.model import CombatNet, collate

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "combat_bc_v2"


class NNBot(Policy):
    def __init__(self, model_dir: Path = DEFAULT_MODEL, seed: int | None = None, end_turn_guard: bool = True):
        self.base = HeuristicBot(seed=seed)
        self.end_turn_guard = end_turn_guard
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(Path(model_dir) / "model.pt", map_location=self.device)
        cfg = ckpt["config"]
        self.model = CombatNet(cfg["sizes"], cfg["d"], cfg["layers"], cfg["heads"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.vocab = Vocab.load(Path(model_dir) / "vocab.json")
        self.name = f"nn:{Path(model_dir).name}"
        self.last_info: dict = {}

    def on_run_start(self) -> None:
        self.base.on_run_start()

    def choose(self, state: State, actions: list[Action]) -> Action:
        teacher = self.base.choose(state, actions)
        if state.get("state_type") not in COMBAT_TYPES:
            self.last_info = {}
            return teacher
        example = {"x": encode_state(state, self.vocab), "a": encode_actions(state, actions)}
        with torch.no_grad():
            logits, value = self.model(collate([example], self.device))
        action = actions[int(logits[0].argmax())]
        info = {"teacher_action": teacher, "agree": same_action(action, teacher),
                "value_hp": round(float(torch.sigmoid(value[0, 0])), 3)}
        if self.end_turn_guard and action["action"] == "end_turn" and teacher["action"] != "end_turn":
            action = teacher
            info["guard"] = True
        self.last_info = info
        return action
