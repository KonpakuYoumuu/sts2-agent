"""Rule-based bot: the Phase 3 baseline and the teacher for imitation learning.

Combat is greedy, one card at a time: every legal play is scored by damage
dealt (with kill bonuses), block that's actually needed against the enemies'
intents, debuffs, draw and energy. The best play per energy spent is taken,
and the turn ends when nothing scores above zero. Everything outside combat
uses simple priority rules and the card values in data/ironclad_cards.json.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable

from agent.bots.base import Policy
from agent.bots.parsing import enemy_attack, parse_card, power_amount
from agent.interface.actions import Action
from agent.interface.client import COMBAT_TYPES, State

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Map node values; elites and rests depend on HP, shops on gold.
MAP_LOOKAHEAD = 4
MAP_DISCOUNT = 0.7

# Deck building: each existing copy of a card lowers the value of another one.
DUPLICATE_PENALTY = 1.5
BIG_DECK = 25

# Cards whose value the text parser can't see.
SPECIAL_CARD_BONUS = {
    # The Insatiable (Act 2 boss) adds these; playing one pushes back its Sandpit
    # countdown. The Sandpit counter isn't exposed, so just play them with spare energy.
    "FRANTIC_ESCAPE": 4.0,
}


def _norm(name: str) -> str:
    """'Fight Me!' -> 'FIGHTME', "Pact's End" -> 'PACTSEND' (upgraded '+' dropped too)."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())


class HeuristicBot(Policy):
    name = "heuristic"

    def __init__(self, seed: int | None = None, card_file: Path = DATA_DIR / "ironclad_cards.json"):
        self.rng = random.Random(seed)
        data = json.loads(card_file.read_text(encoding="utf-8"))
        self.card_values: dict[str, float] = {
            _norm(name): float(score) for score, names in data["tiers"].items() for name in names}
        self.rarity_default: dict[str, float] = data["rarity_default"]
        self.on_run_start()

    def on_run_start(self) -> None:
        self.skipped_card_reward_floors: set[int] = set()
        self.selected_in_screen: dict[str, set[int]] = {}
        self.removal_bought_floors: set[int] = set()
        # Deck card names, recorded at the start of each combat: the only time the
        # mod exposes the full deck.
        self.deck: list[str] = []

    # ---- dispatch ---------------------------------------------------------

    def choose(self, state: State, actions: list[Action]) -> Action:
        st = state.get("state_type")
        handler: Callable[[State, list[Action]], Action | None] | None
        handler = self._combat if st in COMBAT_TYPES else getattr(self, f"_{st}", None)
        choice = handler(state, actions) if handler else None
        return choice if choice is not None else self.rng.choice(actions)

    # ---- helpers ----------------------------------------------------------

    def card_value(self, card: dict) -> float:
        name = _norm(card.get("name") or card.get("card_name") or "")
        if name in self.card_values:
            value = self.card_values[name]
        else:
            rarity = card.get("rarity") or card.get("card_rarity") or "Common"
            value = self.rarity_default.get(rarity, 4)
        ctype = card.get("type") or card.get("card_type")
        if ctype == "Curse":
            value = min(value, self.rarity_default["Curse"])
        elif ctype == "Status":
            value = min(value, self.rarity_default["Status"])
        return value + (0.5 if card.get("is_upgraded") else 0)

    def pick_value(self, card: dict) -> float:
        """Value of adding `card` to the current deck: duplicates and bloat cost."""
        name = _norm((card.get("name") or card.get("card_name") or "").rstrip("+"))
        copies = sum(1 for n in self.deck if n == name)
        return self.card_value(card) - DUPLICATE_PENALTY * copies - (1.0 if len(self.deck) >= BIG_DECK else 0.0)

    @staticmethod
    def _of(actions: list[Action], name: str) -> list[Action]:
        return [a for a in actions if a["action"] == name]

    @staticmethod
    def _hp_ratio(state: State) -> float:
        p = state.get("player") or {}
        return (p.get("hp") or 0) / max(1, p.get("max_hp") or 1)

    # ---- combat -----------------------------------------------------------

    def _combat(self, state: State, actions: list[Action]) -> Action | None:
        player = state["player"]
        battle = state["battle"]
        enemies = {e["entity_id"]: e for e in battle["enemies"] if e.get("hp", 0) > 0}
        hand = {c["index"]: c for c in player.get("hand", [])}
        if battle.get("round") == 1:
            self.deck = [_norm(c["name"]) for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
                         for c in player.get(pile, [])]
        incoming = sum(enemy_attack(e) for e in enemies.values())
        need_block = max(0, incoming - (player.get("block") or 0))
        hp = player.get("hp") or 1
        if self._can_kill_all(list(hand.values()), list(enemies.values()), player.get("energy", 0)):
            need_block = 0  # the fight ends this turn: block is wasted energy
        lethal_threat = need_block >= hp
        # HP saved by blocking matters more the longer the fight will last.
        remaining_hp = sum(e["hp"] + (e.get("block") or 0) for e in enemies.values())
        self._block_weight = 2.5 if lethal_threat else 1.6 + min(remaining_hp / 100, 1.5)
        attacks_in_hand = sum(1 for c in hand.values() if c.get("type") == "Attack" and c.get("can_play"))
        is_hard_fight = state["state_type"] in ("elite", "boss")

        best, best_score = None, 0.0
        for a in actions:
            if a["action"] == "play_card":
                card = hand[a["card_index"]]
                cost = int(card["cost"]) if str(card.get("cost", "")).isdigit() else player.get("energy", 0)
                score = self._score_effects(card.get("description", ""), card.get("type"),
                                            enemies.get(a.get("target")), enemies, need_block,
                                            lethal_threat, attacks_in_hand, battle.get("round", 1), hp,
                                            x=player.get("energy", 0) if card.get("cost") == "X" else 1)
                if card.get("type") == "Power":
                    score += 8 if battle.get("round", 1) <= 2 else 4
                score += SPECIAL_CARD_BONUS.get(card.get("id", ""), 0.0)
                score += self._text_parser_misses(card, player, hand, need_block, lethal_threat,
                                                  attacks_in_hand, battle.get("round", 1))
            elif a["action"] == "use_potion":
                if not (is_hard_fight or lethal_threat or self._hp_ratio(state) < 0.35):
                    continue
                potion = next(p for p in player.get("potions", []) if p["slot"] == a["slot"])
                cost = 0
                score = 4 + self._score_effects(potion.get("description", ""), "Potion",
                                                enemies.get(a.get("target")), enemies, need_block,
                                                lethal_threat, attacks_in_hand, battle.get("round", 1), hp)
            else:
                continue
            efficiency = score / (cost + 0.5)
            if score > 0.5 and efficiency > best_score:
                best, best_score = a, efficiency
        return best or (self._of(actions, "end_turn") or [None])[0]

    def _text_parser_misses(self, card: dict, player: dict, hand: dict[int, dict], need_block: int,
                            lethal_threat: bool, attacks_in_hand: int, round_: int) -> float:
        """Value for effects parse_card can't see ("whenever...", "next turn...", heals...).

        Without this they score 0 and the bot ends its turn holding them.
        """
        cid = card.get("id", "")
        text = card.get("description", "")
        block_weight = self._block_weight
        other_attacks = attacks_in_hand - (1 if card.get("type") == "Attack" else 0)

        if cid == "RAGE" or re.search(r"Whenever you play an Attack this turn, gain (\d+) Block", text):
            per = int(re.search(r"gain (\d+) Block", text).group(1)) if re.search(r"gain (\d+) Block", text) else 3
            block = per * other_attacks
            useful = min(block, need_block)
            return useful * block_weight + (block - useful) * 0.15
        if m := re.search(r"Gain (\d+) Plating", text):
            return int(m.group(1)) * 2.0          # block every turn for several turns
        if m := re.search(r"Heal (\d+) HP", text):
            missing = (player.get("max_hp") or 0) - (player.get("hp") or 0)
            return min(int(m.group(1)), missing) * 1.2
        if "Next turn, gain Block equal to your current Block" in text:
            return (player.get("block") or 0) * 0.8
        if "your next Attack is played an extra time" in text:
            return 6.0 if other_attacks else 0.0
        if "At the end of" in text and re.search(r"deal \d+ damage to ALL enemies", text):
            return 12.0                            # The Bomb: big delayed AoE
        if "If you have no Attacks in your Hand, draw" in text:
            return 5.0 if not other_attacks else 0.0
        if re.search(r"Add .* into your Hand|Choose 1 of 3", text):
            return 5.0                             # card generation (Infernal Blade, Discovery...)
        if card.get("type") in ("Status", "Curse", "Power"):
            return 0.0                             # powers already get a bonus
        fx = parse_card(text)
        if any((fx.damage, fx.block, fx.vulnerable, fx.weak, fx.draw, fx.energy, fx.strength)):
            return 0.0
        # Playable but not understood: still better than wasting the energy.
        return 1.5

    @staticmethod
    def _can_kill_all(hand: list[dict], enemies: list[dict], energy: int) -> bool:
        """Can some set of playable attacks within `energy` kill every enemy this turn?

        Brute force over subsets (hands are <= 10 cards). With several enemies we
        compare total damage to total HP with a 20% margin for overkill waste.
        """
        attacks = []
        for c in hand:
            if c.get("type") != "Attack" or not c.get("can_play") or not str(c.get("cost", "")).isdigit():
                continue
            fx = parse_card(c.get("description", ""))
            if fx.damage:
                attacks.append((int(c["cost"]), fx.damage, fx.aoe))
        if not attacks or not enemies:
            return False
        vuln = any(power_amount(e, "VULNERABLE_POWER") for e in enemies)
        mult = 1.5 if vuln and len(enemies) == 1 else 1.0
        need = sum(e["hp"] + (e.get("block") or 0) for e in enemies)
        if len(enemies) > 1:
            need *= 1.2
        best = 0.0
        for mask in range(1, 1 << len(attacks)):
            cost = dmg = 0.0
            for i, (c, d, aoe) in enumerate(attacks):
                if mask >> i & 1:
                    cost += c
                    dmg += d * (len(enemies) if aoe else 1)
            if cost <= energy:
                best = max(best, dmg * mult)
        return best >= need

    def _score_effects(self, description: str, ctype: str | None, target: dict | None,
                       enemies: dict[str, dict], need_block: int, lethal_threat: bool,
                       attacks_in_hand: int, round_: int, hp: int, x: int = 1) -> float:
        fx = parse_card(description, x)
        score = 0.0

        if fx.damage:
            if fx.aoe:
                hit = list(enemies.values())
            elif target is not None:
                hit = [target]
            else:  # random target: assume the weakest
                hit = sorted(enemies.values(), key=lambda e: e["hp"])[:1]
            for e in hit:
                dmg = fx.damage * (1.5 if ctype == "Attack" and power_amount(e, "VULNERABLE_POWER") else 1)
                hp_dmg = max(0, min(dmg - (e.get("block") or 0), e["hp"]))
                # Breaking block is worth something too: it lets later hits through.
                score += hp_dmg + 0.4 * min(dmg, e.get("block") or 0)
                if dmg >= e["hp"] + (e.get("block") or 0):
                    score += 10 + 1.5 * enemy_attack(e)  # a kill also cancels its attack
                score += 0.01 * max(0, 100 - e["hp"])  # tie-break: focus the weakest

        if fx.block:
            useful = min(fx.block, need_block)
            score += useful * self._block_weight + (fx.block - useful) * 0.15

        debuff_targets = list(enemies.values()) if fx.aoe else [t for t in [target] if t]
        for e in debuff_targets:
            if fx.vulnerable and not power_amount(e, "VULNERABLE_POWER"):
                score += fx.vulnerable * (2 + 2 * min(attacks_in_hand, 3))
            if fx.weak and not power_amount(e, "WEAK_POWER"):
                score += fx.weak * (1 + 0.25 * min(enemy_attack(e), 12))

        score += 2.5 * fx.draw + 4 * fx.energy + 3 * fx.strength
        score -= fx.hp_loss * (3 if hp < 20 else 1)
        return score

    def _hand_select(self, state: State, actions: list[Action]) -> Action | None:
        hs = state["hand_select"]
        confirm = self._of(actions, "combat_confirm_selection")
        if confirm and hs.get("selected_cards"):
            return confirm[0]
        cards = {c["index"]: c for c in hs.get("cards", [])}
        picks = self._of(actions, "combat_select_card")
        if not picks:
            return confirm[0] if confirm else None
        want_best = "upgrade" in (hs.get("prompt") or "").lower() or hs.get("mode") == "upgrade_select"
        key = lambda a: self.card_value(cards[a["card_index"]])  # noqa: E731
        return max(picks, key=key) if want_best else min(picks, key=key)

    # ---- rewards ----------------------------------------------------------

    def _rewards(self, state: State, actions: list[Action]) -> Action | None:
        items = {it["index"]: it for it in state["rewards"].get("items", [])}
        player = state["player"]
        belt_full = len(player.get("potions", [])) >= player.get("max_potion_slots", 3)
        floor = (state.get("run") or {}).get("floor", 0)
        order = {"relic": 0, "gold": 1, "special_card": 2, "card_removal": 2, "potion": 3, "card": 4}
        claims = []
        for a in self._of(actions, "claim_reward"):
            kind = items.get(a["index"], {}).get("type")
            if kind == "potion" and belt_full:
                continue
            if kind == "card" and floor in self.skipped_card_reward_floors:
                continue
            claims.append((order.get(kind, 5), a))
        if claims:
            return min(claims, key=lambda t: t[0])[1]
        return (self._of(actions, "proceed") or [None])[0]

    def _card_reward(self, state: State, actions: list[Action]) -> Action | None:
        cards = {c["index"]: c for c in state["card_reward"].get("cards", [])}
        picks = self._of(actions, "select_card_reward")
        run = state.get("run") or {}
        if not picks:
            return None
        best = max(picks, key=lambda a: self.pick_value(cards[a["card_index"]]))
        value = self.pick_value(cards[best["card_index"]])
        # Early Act 1 needs damage: be less picky about attacks.
        if cards[best["card_index"]].get("type") == "Attack" and run.get("act") == 1 and run.get("floor", 99) < 8:
            value += 1
        skip = self._of(actions, "skip_card_reward")
        if value < 5 and skip:
            self.skipped_card_reward_floors.add(run.get("floor", 0))
            return skip[0]
        return best

    def _treasure(self, state: State, actions: list[Action]) -> Action | None:
        claim = self._of(actions, "claim_treasure_relic")
        return claim[0] if claim else (self._of(actions, "proceed") or [None])[0]

    def _relic_select(self, state: State, actions: list[Action]) -> Action | None:
        return (self._of(actions, "select_relic") or [None])[0]

    def _bundle_select(self, state: State, actions: list[Action]) -> Action | None:
        confirm = self._of(actions, "confirm_bundle_selection")
        if confirm:
            return confirm[0]
        bundles = {b["index"]: b for b in state["bundle_select"].get("bundles", [])}
        picks = self._of(actions, "select_bundle")
        return max(picks, key=lambda a: sum(self.card_value(c) for c in bundles[a["index"]].get("cards", [])),
                   default=None)

    # ---- card grids (upgrade, remove, transform, enchant, choose) ---------

    def _card_select(self, state: State, actions: list[Action]) -> Action | None:
        cs = state["card_select"]
        cards = {c["index"]: c for c in cs.get("cards", [])}
        screen_key = f"{(state.get('run') or {}).get('floor')}|{cs.get('prompt')}|{len(cards)}"
        chosen = self.selected_in_screen.setdefault(screen_key, set())
        picks = [a for a in self._of(actions, "select_card") if a["index"] not in chosen]
        # Some screens enable Confirm before anything is selected, and confirming an
        # empty selection does nothing, so pick the requested number of cards first.
        m = re.search(r"Choose (?:up to )?(\d+) cards?", cs.get("prompt") or "")
        required = int(m.group(1)) if m else 1
        confirm = self._of(actions, "confirm_selection")
        if confirm and (len(chosen) >= required or not picks):
            return confirm[0]
        if not picks:
            return (self._of(actions, "cancel_selection") or [None])[0]
        text = f"{cs.get('screen_type', '')} {cs.get('prompt', '')}".lower()
        want_worst = any(w in text for w in ("remove", "transform"))
        key = lambda a: self.card_value(cards[a["index"]]) - (2 if cards[a["index"]].get("is_upgraded") and not want_worst and "upgrade" in text else 0)  # noqa: E731
        pick = min(picks, key=key) if want_worst else max(picks, key=key)
        chosen.add(pick["index"])
        return pick

    # ---- map --------------------------------------------------------------

    def _node_value(self, node_type: str, hp_ratio: float, gold: int) -> float:
        return {
            "Monster": 2.0,
            "Elite": 4.0 if hp_ratio > 0.8 else (0.5 if hp_ratio > 0.6 else -6.0),
            "RestSite": 5.0 if hp_ratio < 0.5 else 1.5,
            "Shop": 3.0 if gold >= 120 else 0.5,
            "Treasure": 4.0,
            "Unknown": 2.5,
        }.get(node_type, 0.0)

    def _map(self, state: State, actions: list[Action]) -> Action | None:
        m = state["map"]
        nodes = {(n["col"], n["row"]): n for n in m.get("nodes", [])}
        hp_ratio = self._hp_ratio(state)
        gold = (state.get("player") or {}).get("gold", 0)

        def value(col: int, row: int, node_type: str, depth: int) -> float:
            v = self._node_value(node_type, hp_ratio, gold)
            node = nodes.get((col, row))
            if depth >= MAP_LOOKAHEAD or not node or not node.get("children"):
                return v
            kids = [nodes.get((c, r)) for c, r in node["children"]]
            return v + MAP_DISCOUNT * max((value(k["col"], k["row"], k["type"], depth + 1)
                                           for k in kids if k), default=0.0)

        options = {o["index"]: o for o in m.get("next_options", [])}
        picks = self._of(actions, "choose_map_node")
        return max(picks, key=lambda a: value(options[a["index"]]["col"], options[a["index"]]["row"],
                                              options[a["index"]]["type"], 0) + self.rng.random() * 0.01,
                   default=None)

    # ---- rooms ------------------------------------------------------------

    def _rest_site(self, state: State, actions: list[Action]) -> Action | None:
        options = {o["index"]: o for o in state["rest_site"].get("options", [])}
        by_id = {options[a["index"]]["id"].upper(): a for a in self._of(actions, "choose_rest_option")}
        if not by_id:
            return (self._of(actions, "proceed") or [None])[0]
        if self._hp_ratio(state) < 0.55 and "HEAL" in by_id:
            return by_id["HEAL"]
        for pref in ("SMITH", "HEAL"):
            if pref in by_id:
                return by_id[pref]
        return next(iter(by_id.values()))

    def _shop(self, state: State, actions: list[Action]) -> Action | None:
        return self._shop_common(state, state["shop"], actions)

    def _fake_merchant(self, state: State, actions: list[Action]) -> Action | None:
        return self._shop_common(state, state["fake_merchant"]["shop"], actions)

    def _shop_common(self, state: State, shop: dict, actions: list[Action]) -> Action | None:
        items = {it["index"]: it for it in shop.get("items", [])}
        player = state.get("player") or {}
        belt_full = len(player.get("potions", [])) >= player.get("max_potion_slots", 3)
        floor = (state.get("run") or {}).get("floor", 0)
        best, best_value = None, 0.0
        for a in self._of(actions, "shop_purchase"):
            it = items[a["index"]]
            cat = it.get("category")
            if cat == "card_removal":
                value = 0.0 if floor in self.removal_bought_floors else 9.0
            elif cat == "relic":
                value = 8.0
            elif cat == "card":
                v = self.pick_value({"name": it.get("card_name"), "rarity": it.get("card_rarity"),
                                     "type": it.get("card_type")})
                value = v if v >= 7 else 0.0
            elif cat == "potion":
                value = 3.0 if not belt_full and it.get("price", 999) <= 60 else 0.0
            else:
                value = 0.0
            if value > best_value:
                best, best_value = a, value
        if best is not None:
            if items[best["index"]].get("category") == "card_removal":
                self.removal_bought_floors.add(floor)
            return best
        return (self._of(actions, "proceed") or [None])[0]

    # ---- events -----------------------------------------------------------

    def _event(self, state: State, actions: list[Action]) -> Action | None:
        options = {o["index"]: o for o in state["event"].get("options", [])}
        picks = self._of(actions, "choose_event_option")
        if not picks:
            return None
        hp_ratio = self._hp_ratio(state)
        return max(picks, key=lambda a: self._event_option_value(options[a["index"]], hp_ratio)
                   + self.rng.random() * 0.1)

    @staticmethod
    def _event_option_value(option: dict, hp_ratio: float) -> float:
        text = f"{option.get('title', '')} {option.get('description', '')} {option.get('relic_description', '')}".lower()
        v = 0.0
        if option.get("relic_name"):
            v += 3
        for word, w in (("max hp", 3), ("upgrade", 2), ("remove", 2), ("rare", 2), ("relic", 2),
                        ("gold", 1), ("heal", 2 if hp_ratio < 0.5 else 0.5), ("card", 0.5)):
            if word in text:
                v += w
        for word, w in (("curse", 5), ("injury", 4), ("wound", 3), ("debt", 3)):
            if word in text:
                v -= w
        if m := re.search(r"lose (\d+) max hp", text):
            v -= int(m.group(1)) * 0.8
        elif m := re.search(r"lose (\d+) hp", text):
            v -= int(m.group(1)) * (0.6 if hp_ratio < 0.5 else 0.2)
        if "fight" in text and hp_ratio < 0.5:
            v -= 3
        return v

    def _crystal_sphere(self, state: State, actions: list[Action]) -> Action | None:
        clicks = self._of(actions, "crystal_sphere_click_cell")
        return self.rng.choice(clicks) if clicks else (self._of(actions, "crystal_sphere_proceed") or [None])[0]
