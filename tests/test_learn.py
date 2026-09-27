import pytest

from agent.interface.actions import legal_actions
from agent.learn.features import KIND_CARD, KIND_END, Vocab, encode_actions, encode_state

torch = pytest.importorskip("torch")

STATE = {
    "state_type": "monster",
    "battle": {"round": 1, "turn": "player", "is_play_phase": True, "enemies": [
        {"entity_id": "NIBBIT_0", "name": "Nibbit", "hp": 20, "max_hp": 20, "block": 0, "status": [],
         "intents": [{"type": "Attack", "label": "6x2"}]}]},
    "run": {"act": 1, "floor": 2},
    "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "status": [], "potions": [],
               "draw_pile": [{"name": "Strike+"}], "discard_pile": [], "exhaust_pile": [],
               "hand": [
                   {"index": 0, "id": "STRIKE_IRONCLAD", "name": "Strike", "type": "Attack", "cost": "1",
                    "description": "Deal 6 damage.", "target_type": "AnyEnemy", "can_play": True},
                   {"index": 1, "id": "DEFEND_IRONCLAD", "name": "Defend+", "type": "Skill", "cost": "1",
                    "description": "Gain 8 Block.", "target_type": "Self", "can_play": True, "is_upgraded": True},
               ]},
}


def test_encoding_and_model_score_every_legal_action():
    from agent.learn.model import CombatNet, collate

    vocab = Vocab()
    actions = legal_actions(STATE)
    x = encode_state(STATE, vocab)
    a = encode_actions(STATE, actions)
    assert [k for k, *_ in a] == [KIND_CARD, KIND_CARD, KIND_END]
    assert a[0][2] == 0 and a[1][2] == -1  # Strike targets the Nibbit, Defend has no target
    assert vocab.card("Strike+") == vocab.card("Strike")
    model = CombatNet(vocab.sizes(), d=32, layers=1, heads=2)
    logits, value = model(collate([{"x": x, "a": a}, {"x": x, "a": a[:1]}]))
    assert logits.shape == (2, 3) and value.shape == (2, 2)
    assert torch.isinf(logits[1, 1:]).all() and torch.isfinite(logits[0]).all()


def test_nn_bot_end_turn_guard_follows_teacher(tmp_path):
    from agent.bots.nn_bot import NNBot
    from agent.learn.model import CombatNet

    vocab = Vocab()
    encode_state(STATE, vocab)
    vocab.save(tmp_path / "vocab.json")
    model = CombatNet(vocab.sizes(), d=32, layers=1, heads=2)
    with torch.no_grad():  # every action scores 0, so argmax takes the first one
        model.scorer[-1].weight.zero_()
        model.scorer[-1].bias.zero_()
    torch.save({"state_dict": model.state_dict(), "config": {"sizes": vocab.sizes(), "d": 32, "layers": 1,
                                                             "heads": 2}}, tmp_path / "model.pt")
    actions = legal_actions(STATE)
    actions = actions[-1:] + actions[:-1]  # end_turn first: the network "chooses" it
    bot = NNBot(tmp_path, seed=0)
    action = bot.choose(STATE, actions)
    assert action["action"] == "play_card" and bot.last_info.get("guard")
    assert NNBot(tmp_path, seed=0, end_turn_guard=False).choose(STATE, actions)["action"] == "end_turn"
