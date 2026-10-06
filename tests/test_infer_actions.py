from agent.learn.infer_actions import label_records, merge_move_log, split_runs


def card(i, cid, target="None", up=False):
    return {"index": i, "id": cid, "name": cid.title() + ("+" if up else ""), "is_upgraded": up,
            "target_type": target, "can_play": True}


def enemy(eid, hp, block=0):
    return {"entity_id": eid, "hp": hp, "block": block, "status": []}


def fight(hand, enemies, rnd=1, st="monster", potions=()):
    return {"state": {"state_type": st, "player": {"hand": hand, "potions": list(potions), "energy": 3},
                      "battle": {"round": rnd, "enemies": enemies}}}


def actions(records):
    labeled, _ = label_records(records)
    return [r.get("action") for r in labeled]


STRIKE = card(0, "STRIKE", "AnyEnemy")
DEFEND = card(1, "DEFEND")


def test_targeted_play_is_read_from_the_enemy_that_lost_hp():
    recs = [fight([STRIKE, DEFEND], [enemy("A", 20), enemy("B", 20)]),
            fight([{**DEFEND, "index": 0}], [enemy("A", 20), enemy("B", 14)])]
    assert actions(recs)[0] == {"action": "play_card", "card_index": 0, "target": "B"}


def test_effect_landing_a_state_later_still_finds_the_target():
    recs = [fight([STRIKE, DEFEND], [enemy("A", 20), enemy("B", 20)]),
            fight([{**DEFEND, "index": 0}], [enemy("A", 20), enemy("B", 20)]),
            fight([{**DEFEND, "index": 0}], [enemy("A", 14), enemy("B", 20)])]
    assert actions(recs)[0]["target"] == "A"


def test_end_turn_through_a_card_choice_screen():
    recs = [fight([DEFEND], [enemy("A", 20)], rnd=1),
            fight([], [enemy("A", 20)], rnd=2, st="hand_select"),
            fight([STRIKE], [enemy("A", 20)], rnd=2)]
    assert actions(recs)[0] == {"action": "end_turn"}


def test_in_hand_upgrade_is_not_a_second_play():
    armaments = card(0, "ARMAMENTS")
    recs = [fight([armaments, card(1, "STRIKE", "AnyEnemy")], [enemy("A", 20)]),
            fight([card(0, "STRIKE", "AnyEnemy", up=True)], [enemy("A", 20)])]
    assert actions(recs)[0] == {"action": "play_card", "card_index": 0}


def test_two_cards_gone_at_once_is_left_unlabeled():
    recs = [fight([STRIKE, DEFEND], [enemy("A", 20)]),
            fight([], [enemy("A", 14)])]
    assert actions(recs)[0] is None


def test_potion_use():
    potion = {"slot": 1, "id": "BLOCK_POTION", "target_type": "None", "can_use_in_combat": True}
    recs = [fight([DEFEND], [enemy("A", 20)], potions=[potion]),
            fight([DEFEND], [enemy("A", 20)])]
    assert actions(recs)[0] == {"action": "use_potion", "slot": 1}


def test_move_log_merges_with_recorded_screens_by_time():
    play = {"action": "play_card", "card_index": 0}
    moves = [{"t": 10.0, "state": fight([STRIKE], [enemy("A", 20)])["state"], "action": play},
             {"t": 11.0, "state": fight([], [enemy("A", 14)])["state"], "action": {"action": "end_turn"}}]
    recorded = [{"ts": 9.0, "state": {"state_type": "map"}},
                {"ts": 10.5, "state": fight([], [enemy("A", 14)])["state"]},  # recorded fight state: dropped
                {"ts": 12.0, "state": {"state_type": "rewards", "player": {"hp": 50, "max_hp": 80}}},
                {"ts": 20.0, "state": {"state_type": "game_over"}},
                {"ts": 30.0, "state": {"state_type": "map"}}]
    timeline = merge_move_log(moves, recorded)
    assert [r["state"]["state_type"] for r in timeline] == ["map", "monster", "monster", "rewards", "game_over", "map"]
    assert timeline[1]["action"] == play
    assert len(split_runs(timeline)) == 1  # the trailing map has no fights yet


def test_new_run_without_game_over_is_split_by_floor():
    def at(floor, st="map"):
        return {"state": {"state_type": st, "run": {"floor": floor}, "player": {"hand": []},
                          "battle": {"round": 1, "enemies": []}}}
    runs = split_runs([at(40, "monster"), at(41), at(1), at(2, "monster")])
    assert [[r["state"]["run"]["floor"] for r in run] for run in runs] == [[40, 41], [1, 2]]
