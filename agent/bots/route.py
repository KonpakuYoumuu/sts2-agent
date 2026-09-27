"""Route planning: pick map nodes (and heal vs. upgrade) to maximize the chance
of beating the act boss.

The model tracks HP (as a fraction of max HP) along every route to the boss.
Each room costs HP and carries some risk of dying; rewards (relics, cards,
upgrades) slightly raise the chance of winning later. The boss is worth the
estimated chance of beating it at the HP the route arrives with. Choices along
the way (which child next, heal or upgrade at a rest) are made optimally by
dynamic programming, so later rests are counted when judging an early elite.

All numbers come from 596 heuristic-bot runs (logs/night_0927, Act 1,
Ascension 1); HP losses are net of Burning Blood's heal. Acts 2-3 reuse them.
"""

from __future__ import annotations

REST_HEAL = 0.30              # "Heal for 30% of your Max HP"
EARLY_ELITE_ROW = 8           # elites up to here meet a near-starter deck
UPGRADE_BONUS = 1.05          # Smith instead of healing

# (HP lost as a fraction of max HP, reward multiplier)
ROOMS = {
    "Monster": (0.09, 1.01),   # 7 HP of 80 per hallway fight; card reward
    "Unknown": (0.04, 1.01),   # mix of events, fights and shops
    "Treasure": (0.0, 1.05),
    "Shop": (0.0, 1.0),        # set from gold in `plan`
}
ELITE_LOSS = {True: 0.38, False: 0.25}   # early / late: 31 vs 20 HP of 80
ELITE_REWARD = 1.08                      # relic + card

# Chance of beating the boss by HP fraction on arrival (40-60%: 15%,
# 60-80%: 40%, 80%+: 45%); linear in between.
BOSS_WIN = [(0.0, 0.0), (0.3, 0.03), (0.5, 0.15), (0.7, 0.40), (0.9, 0.45), (1.0, 0.47)]


def boss_win(hp: float) -> float:
    for (x0, y0), (x1, y1) in zip(BOSS_WIN, BOSS_WIN[1:]):
        if hp <= x1:
            return y0 + (y1 - y0) * (max(hp, x0) - x0) / (x1 - x0)
    return BOSS_WIN[-1][1]


def death_risk(room: str, hp: float, early: bool) -> float:
    """Chance a fight kills the player, by HP fraction entering it."""
    if room == "Elite":
        if hp < 0.4:
            return 0.45 if early else 0.40
        if hp < 0.7:
            return 0.35 if early else 0.22
        return 0.13 if early else 0.07
    if room == "Monster":
        return 0.15 if hp < 0.25 else (0.04 if hp < 0.5 else 0.01)
    return 0.0


class RoutePlanner:
    """Built from a map state; values routes from any node at any HP."""

    def __init__(self, map_state: dict, gold: int = 0):
        self.nodes = {(n["col"], n["row"]): n for n in map_state.get("nodes", [])}
        self.shop_reward = 1.03 if gold >= 120 else 1.0
        self._memo: dict[tuple, float] = {}

    def children(self, node: dict | None) -> list[dict]:
        if not node:
            return []
        return [self.nodes[(c, r)] for c, r in node.get("children", []) if (c, r) in self.nodes]

    def best_next(self, kids: list[dict], hp: float) -> float:
        """Value of continuing from a room's exit at `hp`; the boss if nothing follows."""
        if not kids:
            return boss_win(hp)
        return max(self.value(k, hp) for k in kids)

    def rest_choice(self, node: dict | None, hp: float) -> tuple[str, float]:
        """('HEAL' or 'SMITH', value) at a rest site `node`."""
        kids = self.children(node)
        heal = self.best_next(kids, min(1.0, hp + REST_HEAL))
        smith = UPGRADE_BONUS * self.best_next(kids, hp)
        return ("HEAL", heal) if heal >= smith else ("SMITH", smith)

    def value(self, node: dict, hp: float) -> float:
        key = (node["col"], node["row"], node.get("type", ""), round(hp, 2))
        if key not in self._memo:
            self._memo[key] = self._value(node, *key[2:])
        return self._memo[key]

    def _value(self, node: dict, room: str, hp: float) -> float:
        node = self.nodes.get((node["col"], node["row"]), node)
        row = node["row"]
        if room == "Boss":
            return boss_win(hp)
        if room == "RestSite":
            return self.rest_choice(node, hp)[1]
        early = row <= EARLY_ELITE_ROW
        if room == "Elite":
            loss, reward = ELITE_LOSS[early], ELITE_REWARD
        elif room == "Shop":
            loss, reward = 0.0, self.shop_reward
        else:
            loss, reward = ROOMS.get(room, (0.0, 1.0))
        survive = 1.0 - death_risk(room, hp, early)
        after = hp - loss
        if after <= 0.02:  # the expected loss alone would kill
            survive, after = min(survive, 0.4), 0.02
        return survive * reward * self.best_next(self.children(node), after)
