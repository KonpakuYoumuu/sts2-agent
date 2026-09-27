"""Enumerate the legal in-run actions for a game state.

Every action is a plain dict in the exact shape the mod's POST endpoint takes,
so it can be sent, logged, and later used as the RL action space (masking).
Menus and the game-over screen aren't decisions and are handled by the runner.
"""

from __future__ import annotations

from typing import Any

from agent.interface.client import COMBAT_TYPES, State

Action = dict[str, Any]

# Actions that move the game forward rather than make a choice. Bots use this
# to avoid ending turns or leaving screens too eagerly.
PROGRESS_ACTIONS = {
    "end_turn", "proceed", "skip_card_reward", "skip_relic_selection",
    "cancel_selection", "cancel_bundle_selection", "crystal_sphere_proceed",
}


def _alive_enemy_ids(state: State) -> list[str]:
    enemies = (state.get("battle") or {}).get("enemies", [])
    return [e["entity_id"] for e in enemies if e.get("hp", 0) > 0]


def _with_targets(base: Action, target_type: str | None, enemy_ids: list[str]) -> list[Action]:
    if target_type == "AnyEnemy":
        return [{**base, "target": eid} for eid in enemy_ids]
    return [base]


def _combat(state: State) -> list[Action]:
    player = state.get("player") or {}
    enemies = _alive_enemy_ids(state)
    actions: list[Action] = []
    for card in player.get("hand", []):
        if card.get("can_play"):
            base = {"action": "play_card", "card_index": card["index"]}
            actions += _with_targets(base, card.get("target_type"), enemies)
    for potion in player.get("potions", []):
        if potion.get("can_use_in_combat"):
            base = {"action": "use_potion", "slot": potion["slot"]}
            actions += _with_targets(base, potion.get("target_type"), enemies)
    actions.append({"action": "end_turn"})
    return actions


def _hand_select(state: State) -> list[Action]:
    hs = state["hand_select"]
    selected = {c["index"] for c in hs.get("selected_cards", [])}
    actions: list[Action] = [{"action": "combat_select_card", "card_index": c["index"]}
                             for c in hs.get("cards", []) if c["index"] not in selected]
    if hs.get("can_confirm"):
        actions.append({"action": "combat_confirm_selection"})
    return actions


def _rewards(state: State) -> list[Action]:
    r = state["rewards"]
    actions: list[Action] = [{"action": "claim_reward", "index": it["index"]}
                             for it in r.get("items", [])]
    if r.get("can_proceed"):
        actions.append({"action": "proceed"})
    return actions


def _card_reward(state: State) -> list[Action]:
    cr = state["card_reward"]
    actions: list[Action] = [{"action": "select_card_reward", "card_index": c["index"]}
                             for c in cr.get("cards", [])]
    if cr.get("can_skip"):
        actions.append({"action": "skip_card_reward"})
    return actions


def _map(state: State) -> list[Action]:
    return [{"action": "choose_map_node", "index": o["index"]}
            for o in state["map"].get("next_options", [])]


def _event(state: State) -> list[Action]:
    ev = state["event"]
    # Ancients (e.g. Neow) can keep in_dialogue=true while their options are
    # already clickable, so options take priority over advancing dialogue.
    options = [{"action": "choose_event_option", "index": o["index"]}
               for o in ev.get("options", []) if not o.get("is_locked")]
    if options:
        return options
    return [{"action": "advance_dialogue"}] if ev.get("in_dialogue") else []


def _rest_site(state: State) -> list[Action]:
    rs = state["rest_site"]
    actions: list[Action] = [{"action": "choose_rest_option", "index": o["index"]}
                             for o in rs.get("options", []) if o.get("is_enabled")]
    if rs.get("can_proceed"):
        actions.append({"action": "proceed"})
    return actions


def _shop_items(shop: dict[str, Any]) -> list[Action]:
    actions: list[Action] = [{"action": "shop_purchase", "index": it["index"]}
                             for it in shop.get("items", [])
                             if it.get("is_stocked") and it.get("can_afford")]
    # The mod reopens the inventory on every state read, which disables the
    # proceed button, so can_proceed is always false. `proceed` closes the
    # inventory first and works regardless.
    actions.append({"action": "proceed"})
    return actions


def _treasure(state: State) -> list[Action]:
    t = state["treasure"]
    actions: list[Action] = [{"action": "claim_treasure_relic", "index": r["index"]}
                             for r in t.get("relics", [])]
    if t.get("can_proceed"):
        actions.append({"action": "proceed"})
    return actions


def _card_select(state: State) -> list[Action]:
    cs = state["card_select"]
    actions: list[Action] = []
    if not cs.get("preview_showing"):
        actions += [{"action": "select_card", "index": c["index"]} for c in cs.get("cards", [])]
    if cs.get("can_confirm") or cs.get("preview_showing"):
        actions.append({"action": "confirm_selection"})
    if cs.get("can_cancel") or cs.get("can_skip"):
        actions.append({"action": "cancel_selection"})
    return actions


def _bundle_select(state: State) -> list[Action]:
    bs = state["bundle_select"]
    if bs.get("preview_showing"):
        actions: list[Action] = [{"action": "confirm_bundle_selection"}]
        if bs.get("can_cancel"):
            actions.append({"action": "cancel_bundle_selection"})
        return actions
    return [{"action": "select_bundle", "index": b["index"]} for b in bs.get("bundles", [])]


def _relic_select(state: State) -> list[Action]:
    rs = state["relic_select"]
    actions: list[Action] = [{"action": "select_relic", "index": r["index"]}
                             for r in rs.get("relics", [])]
    if rs.get("can_skip"):
        actions.append({"action": "skip_relic_selection"})
    return actions


def _crystal_sphere(state: State) -> list[Action]:
    cs = state["crystal_sphere"]
    actions: list[Action] = [{"action": "crystal_sphere_click_cell", "x": c["x"], "y": c["y"]}
                             for c in cs.get("clickable_cells", [])]
    if cs.get("can_proceed"):
        actions.append({"action": "crystal_sphere_proceed"})
    return actions


_HANDLERS = {
    "hand_select": _hand_select,
    "rewards": _rewards,
    "card_reward": _card_reward,
    "map": _map,
    "event": _event,
    "rest_site": _rest_site,
    "shop": lambda s: _shop_items(s["shop"]),
    "fake_merchant": lambda s: _shop_items(s["fake_merchant"]["shop"]),
    "treasure": _treasure,
    "card_select": _card_select,
    "bundle_select": _bundle_select,
    "relic_select": _relic_select,
    "crystal_sphere": _crystal_sphere,
}


def legal_actions(state: State) -> list[Action]:
    """All actions the player may take in this state ([] for menus/game over/unknown)."""
    st = state.get("state_type")
    if st in COMBAT_TYPES:
        return _combat(state)
    handler = _HANDLERS.get(st)
    return handler(state) if handler else []


def action_key(action: Action) -> str:
    """Stable string form of an action, for logging and bookkeeping."""
    return ",".join(f"{k}={action[k]}" for k in sorted(action))
