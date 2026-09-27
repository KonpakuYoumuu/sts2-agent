"""Turn a combat state and its legal actions into arrays for the neural network.

The encoding is shared by imitation learning and later RL:

- tokens: one per hand card, one per enemy; each is an ID (embedded by the
  model) plus numeric features parsed from the text.
- globals: player numbers, plus bags (ID lists) of player powers, potions and
  the cards in draw/discard/exhaust piles.
- actions: each legal action becomes (kind, hand slot, target enemy slot,
  potion slot), so the model scores exactly the legal moves.

IDs come from a `Vocab` built from the training logs; unseen names map to 1
(UNK), padding is 0.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.bots.parsing import enemy_attack, parse_card
from agent.interface.actions import Action
from agent.interface.client import State

CARD_TYPES = ["Attack", "Skill", "Power", "Status", "Curse"]
INTENT_TYPES = ["Attack", "Buff", "StatusCard", "Debuff", "Defend", "Stun", "Summon",
                "DebuffStrong", "Sleep", "CardDebuff", "Heal", "Escape"]
KIND_END, KIND_CARD, KIND_POTION = 0, 1, 2
NO_TARGET = -1

N_CARD_NUM = 18
N_ENEMY_NUM = 8 + len(INTENT_TYPES)
N_GLOBAL_NUM = 13

def card_key(name: str) -> str:
    """Hand, pile and reward cards all have names; hand upgrades show as 'Bash+'."""
    return (name or "").rstrip("+").strip().lower()

def enemy_key(entity_id: str) -> str:
    return re.sub(r"_\d+$", "", entity_id or "")

def _log(x: float) -> float:
    return math.log1p(max(x, 0.0))

@dataclass
class Vocab:
    cards: dict[str, int] = field(default_factory=dict)
    enemies: dict[str, int] = field(default_factory=dict)
    powers: dict[str, int] = field(default_factory=dict)
    potions: dict[str, int] = field(default_factory=dict)
    frozen: bool = False

    def _id(self, table: dict[str, int], key: str) -> int:
        if key in table:
            return table[key]
        if self.frozen:
            return 1
        table[key] = len(table) + 2  # 0 = pad, 1 = unknown
        return table[key]

    def card(self, name: str) -> int:
        return self._id(self.cards, card_key(name))

    def enemy(self, entity_id: str) -> int:
        return self._id(self.enemies, enemy_key(entity_id))

    def power(self, pid: str) -> int:
        return self._id(self.powers, pid)

    def potion(self, pid: str) -> int:
        return self._id(self.potions, pid)

    def sizes(self) -> dict[str, int]:
        return {k: len(getattr(self, k)) + 2 for k in ("cards", "enemies", "powers", "potions")}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: getattr(self, k) for k in ("cards", "enemies", "powers", "potions")},
                                   indent=1))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        v = cls(**json.loads(path.read_text()))
        v.frozen = True
        return v

def _cost(card: dict) -> tuple[float, float]:
    """(cost, is_x). Unplayable cards have cost -1 or a non-number."""
    c = str(card.get("cost", ""))
    if c.upper() == "X":
        return 0.0, 1.0
    try:
        return float(c), 0.0
    except ValueError:
        return -1.0, 0.0

def _card_numeric(card: dict, energy: int) -> list[float]:
    cost, is_x = _cost(card)
    fx = parse_card(card.get("description", ""), x=energy if is_x else 1)
    ctype = card.get("type")
    return [
        cost / 3, is_x, float(bool(card.get("is_upgraded"))), float(bool(card.get("can_play", True))),
        *[float(ctype == t) for t in CARD_TYPES],
        _log(fx.damage) / 3, float(fx.aoe), _log(fx.block) / 3, fx.vulnerable / 3, fx.weak / 3,
        fx.draw / 3, fx.energy / 2, fx.strength / 3, float(fx.exhausts),
    ]

def _enemy_numeric(enemy: dict) -> list[float]:
    hp, mhp = enemy.get("hp", 0), max(enemy.get("max_hp", 1), 1)
    dmg = enemy_attack(enemy)
    types = {i.get("type") for i in enemy.get("intents", [])}
    status = {s.get("id"): s.get("amount") or 0 for s in enemy.get("status", [])}
    return [
        _log(hp) / 5, hp / mhp, _log(enemy.get("block", 0)) / 3, _log(dmg) / 3,
        float("VULNERABLE_POWER" in status), float("WEAK_POWER" in status),
        (status.get("STRENGTH_POWER") or 0) / 5, float(hp <= 0),
        *[float(t in types) for t in INTENT_TYPES],
    ]

def encode_state(state: State, vocab: Vocab) -> dict[str, Any]:
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    run = state.get("run") or {}
    energy = int(player.get("energy") or 0)
    hand = player.get("hand", [])
    enemies = battle.get("enemies", [])

    incoming = sum(enemy_attack(e) for e in enemies if e.get("hp", 0) > 0)
    hp, mhp = player.get("hp", 0), max(player.get("max_hp", 1), 1)
    glob = [
        _log(hp) / 5, hp / mhp, _log(player.get("block", 0)) / 3, energy / 3,
        (player.get("max_energy") or 3) / 3, _log(battle.get("round", 1)) / 3,
        (run.get("act") or 1) / 3, (run.get("floor") or 0) / 50,
        _log(player.get("draw_pile_count", 0)) / 3, _log(player.get("discard_pile_count", 0)) / 3,
        _log(player.get("exhaust_pile_count", 0)) / 3, _log(incoming) / 3,
        float(state.get("state_type") == "boss") + 0.5 * float(state.get("state_type") == "elite"),
    ]
    powers = [vocab.power(s.get("id", "")) for s in player.get("status", [])]
    power_amt = [_log(abs(s.get("amount") or 1)) * (1 if (s.get("amount") or 1) > 0 else -1)
                 for s in player.get("status", [])]
    potions = [vocab.potion(p.get("id", "")) for p in player.get("potions", [])]
    piles = []
    pile_src = []
    for src, pile in enumerate(("draw_pile", "discard_pile", "exhaust_pile")):
        for c in player.get(pile, []):
            piles.append(vocab.card(c.get("name", "")))
            pile_src.append(src)
    return {
        "hand_ids": [vocab.card(c.get("name", "")) for c in hand],
        "hand_num": [_card_numeric(c, energy) for c in hand],
        "enemy_ids": [vocab.enemy(e.get("entity_id", "")) for e in enemies],
        "enemy_num": [_enemy_numeric(e) for e in enemies],
        "glob": glob,
        "power_ids": powers, "power_amt": power_amt,
        "potion_ids": potions,
        "pile_ids": piles, "pile_src": pile_src,
    }

def encode_actions(state: State, actions: list[Action]) -> list[tuple[int, int, int, int]]:
    """(kind, hand slot, enemy slot, potion slot) per action; -1 where not used."""
    player = state.get("player") or {}
    hand_pos = {c["index"]: i for i, c in enumerate(player.get("hand", []))}
    pot_pos = {p["slot"]: i for i, p in enumerate(player.get("potions", []))}
    enemy_pos = {e["entity_id"]: i for i, e in enumerate((state.get("battle") or {}).get("enemies", []))}
    out = []
    for a in actions:
        target = enemy_pos.get(a.get("target"), NO_TARGET)
        if a["action"] == "play_card":
            out.append((KIND_CARD, hand_pos[a["card_index"]], target, -1))
        elif a["action"] == "use_potion":
            out.append((KIND_POTION, -1, target, pot_pos[a["slot"]]))
        else:
            out.append((KIND_END, -1, -1, -1))
    return out

def same_action(a: Action, b: Action) -> bool:
    return all(a.get(k) == b.get(k) for k in ("action", "card_index", "slot", "target"))
