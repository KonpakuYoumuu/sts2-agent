from agent.bots.heuristic import HeuristicBot
from agent.bots.parsing import intent_damage, parse_card
from agent.interface.actions import legal_actions

STRIKE = {"id": "STRIKE_IRONCLAD", "type": "Attack", "cost": "1", "target_type": "AnyEnemy",
          "description": "Deal 6 damage.", "can_play": True}
DEFEND = {"id": "DEFEND_IRONCLAD", "type": "Skill", "cost": "1", "target_type": "Self",
          "description": "Gain 5 Block.", "can_play": True}


def combat(hand, enemies, energy=3, hp=80, block=0, state_type="monster"):
    return {
        "state_type": state_type,
        "battle": {"round": 3, "turn": "player", "is_play_phase": True, "enemies": enemies},
        "player": {"hp": hp, "max_hp": 80, "block": block, "energy": energy, "potions": [],
                   "hand": [dict(c, index=i) for i, c in enumerate(hand)]},
    }


def enemy(eid="E_0", hp=40, attack="", status=()):
    intents = [{"type": "Attack", "label": attack}] if attack else [{"type": "Buff", "label": ""}]
    return {"entity_id": eid, "hp": hp, "max_hp": hp, "block": 0, "intents": intents,
            "status": list(status)}


def choose(state):
    return HeuristicBot(seed=0).choose(state, legal_actions(state))


def test_parse_multi_hit_and_energy():
    assert parse_card("Deal 5 damage twice.").damage == 10
    assert parse_card("Deal 3 damage to a random enemy 3 times.").damage == 9
    assert parse_card("Lose 3 HP. Gain [ironclad_energy_icon.png][ironclad_energy_icon.png].").energy == 2
    assert intent_damage("6x2") == 12 and intent_damage("") == 0


def test_blocks_when_big_attack_incoming():
    state = combat([STRIKE, DEFEND], [enemy(hp=40, attack="14")], energy=1)
    assert choose(state)["card_index"] == 1


def test_attacks_when_enemy_not_attacking():
    state = combat([STRIKE, DEFEND], [enemy(hp=40)])
    assert choose(state)["card_index"] == 0


def test_no_blocking_when_attacks_in_hand_are_lethal():
    # 12 incoming, but two Strikes (2 energy) kill the 12-HP enemy: attack, don't Defend.
    state = combat([DEFEND, STRIKE, STRIKE], [enemy(hp=12, attack="12")], energy=2)
    assert choose(state)["action"] == "play_card"
    assert state["player"]["hand"][choose(state)["card_index"]]["id"] == "STRIKE_IRONCLAD"


def test_still_blocks_when_attacks_are_not_lethal():
    state = combat([DEFEND, STRIKE, STRIKE], [enemy(hp=30, attack="12")], energy=1)
    assert choose(state)["card_index"] == 0


def test_takes_the_kill_over_blocking():
    state = combat([STRIKE, DEFEND], [enemy("A_0", hp=5, attack="8")])
    a = choose(state)
    assert a["card_index"] == 0 and a["target"] == "A_0"


def test_focuses_killable_enemy():
    state = combat([STRIKE], [enemy("BIG_0", hp=40, attack="5"), enemy("SMALL_0", hp=6, attack="5")])
    assert choose(state)["target"] == "SMALL_0"


def test_ends_turn_with_no_energy_left():
    strike = dict(STRIKE, can_play=False)
    state = combat([strike], [enemy(hp=40)], energy=0)
    assert choose(state) == {"action": "end_turn"}


def test_map_avoids_elite_at_low_hp():
    state = {"state_type": "map", "player": {"hp": 30, "max_hp": 80, "gold": 50},
             "map": {"nodes": [], "next_options": [
                 {"index": 0, "col": 0, "row": 1, "type": "Elite"},
                 {"index": 1, "col": 1, "row": 1, "type": "Monster"}]}}
    assert choose(state)["index"] == 1


def test_rest_heals_when_low_and_smiths_when_healthy():
    def rest(hp):
        return {"state_type": "rest_site", "player": {"hp": hp, "max_hp": 80},
                "rest_site": {"can_proceed": False, "options": [
                    {"index": 0, "id": "HEAL", "is_enabled": True},
                    {"index": 1, "id": "SMITH", "is_enabled": True}]}}
    assert choose(rest(30))["index"] == 0
    assert choose(rest(70))["index"] == 1


def test_card_select_removes_worst_card():
    state = {"state_type": "card_select", "run": {"floor": 5}, "card_select": {
        "screen_type": "select", "prompt": "Choose a card to Remove.", "can_confirm": False,
        "preview_showing": False, "cards": [
            {"index": 0, "name": "Inflame", "type": "Power", "rarity": "Uncommon"},
            {"index": 1, "name": "Strike", "type": "Attack", "rarity": "Basic"},
            {"index": 2, "name": "Injury", "type": "Curse", "rarity": "Curse"}]}}
    assert choose(state) == {"action": "select_card", "index": 2}


def test_duplicate_copies_lower_pick_value():
    bot = HeuristicBot(seed=0)
    card = {"name": "Thunderclap", "type": "Attack", "rarity": "Common"}
    fresh = bot.pick_value(card)
    bot.deck = ["THUNDERCLAP", "THUNDERCLAP"]
    assert bot.pick_value(card) == fresh - 3.0


def test_explore_wrapper_marks_random_moves():
    from agent.bots.explore import Explore
    state = combat([STRIKE, DEFEND], [enemy(hp=40)])
    always = Explore(HeuristicBot(seed=0), epsilon=1.0, seed=0)
    always.choose(state, legal_actions(state))
    assert always.last_info["explore"] is True and "greedy_action" in always.last_info
    never = Explore(HeuristicBot(seed=0), epsilon=0.0, seed=0)
    assert never.choose(state, legal_actions(state)) == HeuristicBot(seed=0).choose(state, legal_actions(state))
    assert never.last_info == {"explore": False}


RAGE = {"id": "RAGE", "type": "Skill", "cost": "0", "target_type": "Self", "can_play": True,
        "description": "Whenever you play an Attack this turn, gain 3 Block."}


def test_plays_rage_before_attacks_when_attack_incoming():
    state = combat([STRIKE, STRIKE, RAGE], [enemy(hp=40, attack="12")], energy=2)
    assert choose(state)["card_index"] == 2


def test_plays_unknown_skill_instead_of_wasting_energy():
    odd = {"id": "MYSTERY", "type": "Skill", "cost": "1", "target_type": "Self", "can_play": True,
           "description": "Something the parser doesn't understand."}
    state = combat([odd], [enemy(hp=40)], energy=1)
    assert choose(state) == {"action": "play_card", "card_index": 0}


def test_blocks_big_hits_in_long_fights():
    # Bygone Effigy case: 127-HP elite hitting for 23. Two Defends beat one 32-damage Bludgeon.
    bludgeon = {"id": "BLUDGEON", "type": "Attack", "cost": "3", "target_type": "AnyEnemy",
                "description": "Deal 32 damage.", "can_play": True}
    state = combat([bludgeon, DEFEND, DEFEND], [enemy(hp=127, attack="23")], energy=3, state_type="elite")
    assert state["player"]["hand"][choose(state)["card_index"]]["id"] == "DEFEND_IRONCLAD"


def test_x_cost_damage_scales_with_energy():
    assert parse_card("Deal 5 damage to ALL enemies X times.", x=3).damage == 15


def test_attacks_through_block_instead_of_useless_defend():
    whirlwind = {"id": "WHIRLWIND", "type": "Attack", "cost": "X", "target_type": "AllEnemies",
                 "description": "Deal 5 damage to ALL enemies X times.", "can_play": True}
    e = enemy(hp=22)
    e["block"] = 5
    state = combat([DEFEND, whirlwind], [e], energy=2)
    assert state["player"]["hand"][choose(state)["card_index"]]["id"] == "WHIRLWIND"


def test_card_select_picks_required_cards_before_confirming():
    # Enchant screens enable Confirm with nothing selected; confirming then does nothing.
    bot = HeuristicBot(seed=0)
    cards = [{"index": i, "name": "Strike", "type": "Attack", "rarity": "Basic"} for i in range(5)]
    state = {"state_type": "card_select", "run": {"floor": 21}, "card_select": {
        "screen_type": "NDeckEnchantSelectScreen", "prompt": "Choose 3 cards to Enchant.",
        "cards": cards, "preview_showing": False, "can_confirm": True, "can_cancel": False}}
    picked = [bot.choose(state, legal_actions(state))["action"] for _ in range(4)]
    assert picked == ["select_card", "select_card", "select_card", "confirm_selection"]


TOXIC = {"id": "TOXIC", "name": "Toxic", "type": "Status", "cost": "1", "target_type": "Self", "can_play": True,
         "description": "At the end of your turn, if this is in your Hand, take 5 damage. Exhaust."}


def test_plays_toxic_instead_of_ending_turn_with_energy_left():
    # The logged mistake: 2 energy, only Toxics in hand, enemy not attacking.
    state = combat([TOXIC, TOXIC, TOXIC], [enemy(hp=2)], energy=2)
    assert choose(state)["action"] == "play_card"


def _map_state(hp, nodes, options):
    return {"state_type": "map", "player": {"hp": hp, "max_hp": 80, "gold": 50},
            "map": {"nodes": nodes, "next_options": options}}


def _node(col, row, type_, children=()):
    return {"col": col, "row": row, "type": type_, "children": [list(c) for c in children]}


def test_route_skips_early_elite_even_at_full_hp():
    nodes = [_node(0, 2, "Elite", [(0, 16)]), _node(1, 2, "Monster", [(0, 16)]), _node(0, 16, "Boss")]
    state = _map_state(80, nodes, [{"index": 0, "col": 0, "row": 2, "type": "Elite"},
                                   {"index": 1, "col": 1, "row": 2, "type": "Monster"}])
    assert choose(state)["index"] == 1


def test_route_takes_late_elite_when_a_rest_follows():
    nodes = [_node(0, 13, "Elite", [(0, 15)]), _node(1, 13, "Monster", [(1, 15)]),
             _node(0, 15, "RestSite", [(0, 16)]), _node(1, 15, "Monster", [(0, 16)]), _node(0, 16, "Boss")]
    state = _map_state(80, nodes, [{"index": 0, "col": 0, "row": 13, "type": "Elite"},
                                   {"index": 1, "col": 1, "row": 13, "type": "Monster"}])
    assert choose(state)["index"] == 0


def test_rest_before_boss_heals_unless_nearly_full():
    nodes = [_node(0, 15, "RestSite", [(0, 16)]), _node(0, 16, "Boss")]
    for hp, expected in ((60, "HEAL"), (76, "SMITH")):  # 75% and 95% of 80
        bot = HeuristicBot(seed=0)
        m = _map_state(hp, nodes, [{"index": 0, "col": 0, "row": 15, "type": "RestSite"}])
        bot.choose(m, legal_actions(m))
        rest = {"state_type": "rest_site", "player": {"hp": hp, "max_hp": 80},
                "rest_site": {"can_proceed": False, "options": [
                    {"index": 0, "id": "HEAL", "is_enabled": True}, {"index": 1, "id": "SMITH", "is_enabled": True}]}}
        chosen = bot.choose(rest, legal_actions(rest))["index"]
        assert ("HEAL", "SMITH")[chosen] == expected


def test_upgrade_screen_picks_once_then_confirms():
    # The upgrade screen never lists the picked card, and clicking again un-picks it.
    state = {"state_type": "hand_select", "hand_select": {
        "mode": "upgrade_select", "prompt": "Confirm Card to Upgrade", "can_confirm": True,
        "cards": [dict(STRIKE, index=0, name="Strike"), dict(DEFEND, index=1, name="Defend")]}}
    bot = HeuristicBot(seed=0)
    assert bot.choose(state, legal_actions(state))["action"] == "combat_select_card"
    assert bot.choose(state, legal_actions(state))["action"] == "combat_confirm_selection"
    bot.choose({"state_type": "monster", "battle": {"enemies": []}, "player": {"hand": []}}, [{"action": "end_turn"}])
    assert bot.choose(state, legal_actions(state))["action"] == "combat_select_card"  # a new screen
