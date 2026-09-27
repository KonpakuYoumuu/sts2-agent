"""Turn the mod's text (card descriptions, intent labels) into numbers.

Card descriptions are dynamic: they already include the player's Strength,
Dexterity, Weak, etc. They don't include the target's Vulnerable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ENERGY_ICON = re.compile(r"\[\w+_energy_icon\.png\]")


@dataclass
class CardEffects:
    damage: int = 0          # total damage per target (all hits)
    aoe: bool = False        # hits all enemies
    block: int = 0
    vulnerable: int = 0      # stacks applied
    weak: int = 0
    draw: int = 0
    energy: int = 0
    strength: int = 0
    hp_loss: int = 0
    exhausts: bool = False


def _num(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def parse_card(description: str, x: int = 1) -> CardEffects:
    """`x`: the value of X for X-cost cards (the energy that will be spent)."""
    text = ENERGY_ICON.sub("[E]", description or "")
    fx = CardEffects()

    per_hit = _num(r"Deal (\d+) damage", text)
    if not per_hit:
        per_hit = _num(r"\(Deals (\d+) damage\)", text)  # e.g. Body Slam's live value
    hits = 1
    if re.search(r"Deal \d+ damage (?:to [\w ]+ )?twice", text):
        hits = 2
    elif m := re.search(r"Deal \d+ damage (?:to [\w ]+ )?(\d+) times", text):
        hits = int(m.group(1))
    elif re.search(r"Deal \d+ damage (?:to [\w ]+ )?X times", text):
        hits = x
    fx.damage = per_hit * hits
    fx.aoe = "ALL enemies" in text

    fx.block = _num(r"Gain (\d+) Block", text)
    fx.vulnerable = _num(r"Apply (\d+) Vulnerable", text) or _num(r"apply (\d+) Vulnerable", text)
    fx.weak = _num(r"Apply (\d+) Weak", text)
    fx.draw = _num(r"Draw (\d+) cards?", text)
    fx.strength = _num(r"Gain (\d+) Strength", text)
    fx.hp_loss = _num(r"Lose (\d+) HP", text)
    fx.exhausts = bool(re.search(r"(?<!an )Exhaust\.", text))
    if m := re.search(r"Gain ((?:\[E\])+)", text):
        fx.energy = m.group(1).count("[E]")
    return fx


def intent_damage(label: str | None) -> int:
    """'11' -> 11, '6x2' -> 12, '' -> 0."""
    if not label:
        return 0
    m = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", label)
    if m:
        return int(m.group(1)) * int(m.group(2))
    m = re.fullmatch(r"\s*(\d+)\s*", label)
    return int(m.group(1)) if m else 0


def enemy_attack(enemy: dict) -> int:
    """Total damage this enemy intends to deal this turn."""
    return sum(intent_damage(i.get("label")) for i in enemy.get("intents", [])
               if i.get("type", "").startswith("Attack"))


def power_amount(entity: dict, power_id: str) -> int:
    for p in entity.get("status", []):
        if p.get("id") == power_id:
            return int(p.get("amount") or 0)
    return 0
