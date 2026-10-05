from agent.interface.actions import legal_actions
from agent.interface.client import is_settled


def combat_state(**battle_overrides):
    battle = {
        "round": 1, "turn": "player", "is_play_phase": True,
        "enemies": [
            {"entity_id": "NIBBIT_0", "hp": 44, "max_hp": 44},
            {"entity_id": "NIBBIT_1", "hp": 0, "max_hp": 44},
        ],
    }
    battle.update(battle_overrides)
    return {
        "state_type": "monster",
        "battle": battle,
        "player": {
            "hand": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "target_type": "AnyEnemy", "can_play": True},
                {"index": 1, "id": "DEFEND_IRONCLAD", "target_type": "Self", "can_play": True},
                {"index": 2, "id": "BASH", "target_type": "AnyEnemy", "can_play": False},
            ],
            "potions": [{"slot": 0, "target_type": "AnyEnemy", "can_use_in_combat": True}],
        },
    }


def test_combat_targets_only_living_enemies_and_skips_unplayable():
    actions = legal_actions(combat_state())
    assert {"action": "play_card", "card_index": 0, "target": "NIBBIT_0"} in actions
    assert {"action": "play_card", "card_index": 1} in actions
    assert not any(a.get("target") == "NIBBIT_1" for a in actions)
    assert not any(a.get("card_index") == 2 for a in actions)
    assert {"action": "use_potion", "slot": 0, "target": "NIBBIT_0"} in actions
    assert actions[-1] == {"action": "end_turn"}


def test_combat_settled_only_in_player_play_phase():
    assert is_settled(combat_state())
    assert not is_settled(combat_state(is_play_phase=False))
    assert not is_settled(combat_state(turn="enemy"))
    dead = [{"entity_id": "NIBBIT_0", "hp": 0}]
    assert not is_settled(combat_state(enemies=dead))


def test_combat_not_settled_while_actions_resolve():
    assert is_settled(combat_state(actions_idle=True, player_actions_disabled=False))
    assert not is_settled(combat_state(actions_idle=False))
    assert not is_settled(combat_state(player_actions_disabled=True))


def test_event_dialogue_then_unlocked_options():
    state = {"state_type": "event", "event": {"in_dialogue": True, "options": []}}
    assert legal_actions(state) == [{"action": "advance_dialogue"}]
    state["event"] = {"in_dialogue": False, "options": [
        {"index": 0, "is_locked": False}, {"index": 1, "is_locked": True}]}
    assert legal_actions(state) == [{"action": "choose_event_option", "index": 0}]


def test_ancient_options_clickable_during_dialogue():
    state = {"state_type": "event", "event": {"in_dialogue": True, "options": [
        {"index": 0, "is_locked": False}]}}
    assert legal_actions(state) == [{"action": "choose_event_option", "index": 0}]


def test_shop_only_affordable_stocked_items():
    state = {"state_type": "shop", "shop": {"can_proceed": False, "items": [
        {"index": 0, "is_stocked": True, "can_afford": True},
        {"index": 1, "is_stocked": True, "can_afford": False},
        {"index": 2, "is_stocked": False, "can_afford": True},
    ]}}
    assert legal_actions(state) == [{"action": "shop_purchase", "index": 0}, {"action": "proceed"}]


def test_shop_not_settled_while_inventory_loading():
    assert not is_settled({"state_type": "shop", "shop": {"error": "not ready"}})


def test_treasure_waits_for_chest():
    opening = {"state_type": "treasure", "treasure": {"message": "Opening chest..."}}
    assert not is_settled(opening)
    opened = {"state_type": "treasure", "treasure": {"relics": [{"index": 0}], "can_proceed": True}}
    assert is_settled(opened)
    assert legal_actions(opened) == [{"action": "claim_treasure_relic", "index": 0},
                                     {"action": "proceed"}]


def test_card_select_preview_only_confirms_or_cancels():
    state = {"state_type": "card_select", "card_select": {
        "cards": [{"index": 0}], "preview_showing": True, "can_confirm": False, "can_cancel": True}}
    assert legal_actions(state) == [{"action": "confirm_selection"}, {"action": "cancel_selection"}]


def test_hand_select_excludes_already_selected():
    state = {"state_type": "hand_select", "hand_select": {
        "cards": [{"index": 0}, {"index": 1}], "selected_cards": [{"index": 0}], "can_confirm": True}}
    assert legal_actions(state) == [{"action": "combat_select_card", "card_index": 1},
                                    {"action": "combat_confirm_selection"}]


def test_menu_and_game_over_have_no_policy_actions():
    assert legal_actions({"state_type": "menu"}) == []
    assert legal_actions({"state_type": "game_over"}) == []


def test_combat_with_no_targetable_enemies_is_still_a_player_turn():
    # Seen in an Act 3 boss phase: enemies list empty, player's turn, cards playable.
    state = combat_state(enemies=[])
    assert is_settled(state)
    actions = legal_actions(state)
    assert {"action": "play_card", "card_index": 1} in actions      # self-target card
    assert not any(a.get("card_index") == 0 for a in actions)       # attack has no target
    assert actions[-1] == {"action": "end_turn"}


def test_victory_is_detected_by_the_ending_event_not_hp():
    from agent.interface.client import is_victory_event

    architect = {"state_type": "event", "event": {"event_id": "THE_ARCHITECT", "options": []}}
    assert is_victory_event(architect)
    assert not is_victory_event({"state_type": "event", "event": {"event_id": "NEOW"}})
    assert not is_victory_event({"state_type": "game_over", "player": {"hp": 0}})
