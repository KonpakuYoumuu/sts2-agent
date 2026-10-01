"""Reinforcement learning for combat in the fight arena (PPO).

Starts from the imitation-learned network and improves it by playing arena
fights. Every recorded move gets an immediate reward:

    ENEMY_WEIGHT * (share of the fight's total enemy HP removed since the last move)
    - (share of own max HP lost since the last move)
    - POTION_COST if the move used a potion       (potions are shared across a run)

and the last move also gets -DEATH_PENALTY if the player died. The enemy term
is potential-based shaping (it sums to ENEMY_WEIGHT over a won fight), so it
speeds up learning without changing what's optimal. Credit reaches earlier
moves through GAE (gamma 1, lambda 0.95) against the network's value output
(head 0, read as "reward still to come"), which is first trained alone for
--value-warmup iterations. (v1 rewarded only the end of each fight and didn't
learn: one number per ~25 moves was too noisy.)

A KL penalty to the frozen imitation network keeps early updates from
wrecking a policy that already plays sensibly. Only combat moves are learned;
selection screens inside a fight stay with the rule-based bot.

    python -m agent.learn.ppo --init models/combat_bc_v2 --out models/combat_ppo_v2 --iterations 150
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from agent.arena.arena import Arena, ArenaError, load_scenarios
from agent.bots.base import Policy
from agent.bots.heuristic import HeuristicBot
from agent.interface.actions import Action
from agent.harness.supervise import DEFAULT_GAME_DIR, restart_game
from agent.interface.client import COMBAT_TYPES, GameClient, GameNotRunning, State, StateTimeout
from agent.learn.features import KIND_POTION, Vocab, encode_actions, encode_state
from agent.learn.model import CombatNet, collate

DEATH_PENALTY = 0.5
POTION_COST = 0.05
ENEMY_WEIGHT = 0.3
GAE_LAMBDA = 0.95


def value_of(v: torch.Tensor) -> torch.Tensor:
    """Expected reward still to come in this fight."""
    return v[..., 0]


def _fight_numbers(state: State) -> tuple[int, int, int]:
    """(player HP, player max HP, total HP of living enemies)."""
    p = state.get("player") or {}
    enemies = (state.get("battle") or {}).get("enemies", [])
    return p.get("hp") or 0, p.get("max_hp") or 1, sum(max(e.get("hp", 0), 0) for e in enemies)


class SamplingPolicy(Policy):
    """Samples combat moves from the network and records them for training."""

    name = "ppo"

    def __init__(self, model: CombatNet, vocab: Vocab, device: str, seed: int = 0, greedy: bool = False):
        self.model, self.vocab, self.device, self.greedy = model, vocab, device, greedy
        self.base = HeuristicBot(seed=seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.steps: list[dict] = []
        self.last_info: dict = {}

    def choose(self, state: State, actions: list[Action]) -> Action:
        teacher = self.base.choose(state, actions)
        if state.get("state_type") not in COMBAT_TYPES:
            self.last_info = {}
            return teacher
        example = {"x": encode_state(state, self.vocab), "a": encode_actions(state, actions)}
        with torch.no_grad():
            logits, value = self.model(collate([example], self.device))
        probs = torch.softmax(logits[0].float().cpu(), -1)
        idx = int(probs.argmax()) if self.greedy else int(torch.multinomial(probs, 1, generator=self.gen))
        hp, max_hp, enemy_hp = _fight_numbers(state)
        enemies = (state.get("battle") or {}).get("enemies", [])
        self.steps.append({**example, "y": idx, "logp": float(torch.log(probs[idx] + 1e-12)),
                           "v_old": float(value_of(value[0].cpu())), "hp": hp, "max_hp": max_hp,
                           "enemy_hp": enemy_hp, "enemy_max": sum(e.get("max_hp", 0) for e in enemies),
                           "potion": actions[idx]["action"] == "use_potion"})
        self.last_info = {"p": round(float(probs[idx]), 3)}
        return actions[idx]


def assign_rewards(steps: list[dict], result) -> float:
    """Per-move rewards, GAE advantages ("adv") and value targets ("ret"); returns the fight's total reward."""
    max_hp = max(steps[0]["max_hp"], 1)
    enemy_total = max(steps[0]["enemy_max"], steps[0]["enemy_hp"], 1)
    end_hp = result.hp_end
    end_enemy = steps[-1]["enemy_hp"] if result.died else 0
    rewards = []
    for t, s in enumerate(steps):
        nxt_hp, nxt_enemy = (steps[t + 1]["hp"], steps[t + 1]["enemy_hp"]) if t + 1 < len(steps) else (end_hp, end_enemy)
        r = ENEMY_WEIGHT * (s["enemy_hp"] - nxt_enemy) / enemy_total - (s["hp"] - nxt_hp) / max_hp
        r -= POTION_COST * s["potion"]
        rewards.append(r)
    rewards[-1] -= DEATH_PENALTY * result.died
    adv, next_v = 0.0, 0.0
    for t in reversed(range(len(steps))):
        delta = rewards[t] + next_v - steps[t]["v_old"]
        adv = delta + GAE_LAMBDA * adv
        steps[t]["adv"] = adv
        steps[t]["ret"] = adv + steps[t]["v_old"]
        next_v = steps[t]["v_old"]
    return sum(rewards)


def ppo_update(model: CombatNet, ref: CombatNet, steps: list[dict], opt, device: str, epochs: int = 4,
               bs: int = 256, clip: float = 0.2, kl_coef: float = 0.1, ent_coef: float = 0.003,
               policy_coef: float = 1.0) -> dict:
    # No dropout: the old log-probabilities were computed without it, so it would
    # make the policy look changed (ratio != 1) before any update.
    model.eval()
    adv = torch.tensor([s["adv"] for s in steps])
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    for s, a in zip(steps, adv_norm.tolist()):
        s["adv"] = a
    stats = {"policy": 0.0, "value": 0.0, "kl_ref": 0.0, "entropy": 0.0, "clipfrac": 0.0, "n": 0}
    for _ in range(epochs):
        random.shuffle(steps)
        for i in range(0, len(steps), bs):
            chunk = steps[i:i + bs]
            b = collate(chunk, device)
            y = torch.tensor([s["y"] for s in chunk], device=device)
            old_logp = torch.tensor([s["logp"] for s in chunk], device=device)
            advt = torch.tensor([s["adv"] for s in chunk], device=device)
            ret = torch.tensor([s["ret"] for s in chunk], device=device)
            logits, v = model(b)
            valid = torch.isfinite(logits)
            # A finite mask value keeps 0 * log(0) terms (and their gradients) at 0 instead of NaN.
            logp_all = torch.log_softmax(logits.masked_fill(~valid, -1e9), -1)
            logp = logp_all.gather(1, y[:, None]).squeeze(1)
            ratio = torch.exp(logp - old_logp)
            pg = -torch.min(ratio * advt, torch.clamp(ratio, 1 - clip, 1 + clip) * advt).mean()
            v_loss = F.mse_loss(value_of(v), ret)
            probs = logp_all.exp()
            entropy = -(probs * logp_all).sum(-1).mean()
            with torch.no_grad():
                ref_logp = torch.log_softmax(ref(b)[0].masked_fill(~valid, -1e9), -1)
            kl = (probs * (logp_all - ref_logp)).sum(-1).mean()
            loss = policy_coef * (pg + kl_coef * kl - ent_coef * entropy) + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = len(chunk)
            stats["policy"] += pg.item() * n
            stats["value"] += v_loss.item() * n
            stats["kl_ref"] += kl.item() * n
            stats["entropy"] += entropy.item() * n
            stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item() * n
            stats["n"] += n
    n = max(stats.pop("n"), 1)
    return {k: round(v / n, 4) for k, v in stats.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, default=Path("models/combat_bc_v2"))
    ap.add_argument("--ref", type=Path, default=Path("models/combat_bc_v2"),
                    help="imitation network the KL penalty anchors to")
    ap.add_argument("--out", type=Path, default=Path("models/combat_ppo_v1"))
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios_v1.jsonl"))
    ap.add_argument("--act", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--fights", type=int, default=40, help="fights per iteration")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--value-warmup", type=int, default=5,
                    help="first N iterations train only the value output (new reward = new value scale)")
    ap.add_argument("--mix", default="monster=0.4,elite=0.3,boss=0.3",
                    help="share of fights per kind (elites and bosses decide most runs)")
    ap.add_argument("--restart-every", type=int, default=10,
                    help="restart the headless game every N iterations (it slows down after a few hours)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.init / "model.pt", map_location=device)
    cfg = ckpt["config"]
    model = CombatNet(cfg["sizes"], cfg["d"], cfg["layers"], cfg["heads"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ref = copy.deepcopy(model)
    ref.load_state_dict(torch.load(args.ref / "model.pt", map_location=device)["state_dict"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vocab = Vocab.load(args.init / "vocab.json")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # Warm-up trains only the value output's own layers, so its large early errors
    # can't leak into the shared layers (and the policy) before it's calibrated.
    warm_opt = torch.optim.AdamW(model.value.parameters(), lr=1e-3, weight_decay=0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    vocab.save(args.out / "vocab.json")
    scenarios = [s for s in load_scenarios(args.scenarios) if s["act"] == args.act]
    by_kind = {k: [s for s in scenarios if s["kind"] == k] for k in ("monster", "elite", "boss")}
    mix = {k: float(v) for k, v in (part.split("=") for part in args.mix.split(","))}
    kinds, weights = zip(*[(k, w) for k, w in mix.items() if by_kind.get(k)])
    rng = random.Random(args.seed)
    client = GameClient()
    if not client.ping():
        print("Can't reach the STS2MCP mod. Start the headless game first.")
        return 1
    policy = SamplingPolicy(model, vocab, device, seed=args.seed)
    arena = Arena(client, HeuristicBot(seed=args.seed))
    # Resuming (--init pointing at a PPO checkpoint): continue its iteration count and history.
    start = ckpt.get("iteration", 0) + 1
    history_file = args.out / "history.json"
    history = json.loads(history_file.read_text()) if start > 1 and history_file.exists() else []
    with gzip.open(args.out / "fights.jsonl.gz", "at", encoding="utf-8") as fight_log:
        def fresh_game(reason: str) -> None:
            print(f"  restarting the game ({reason})", flush=True)
            if not restart_game(DEFAULT_GAME_DIR, args.out / "game"):
                raise SystemExit("The game didn't come back up; stopping.")

        for it in range(start, args.iterations + 1):
            if it > start and (it - 1) % args.restart_every == 0:
                fresh_game("scheduled")
            t0 = time.monotonic()
            batch, results = [], []
            for _ in range(args.fights):
                sc = rng.choice(by_kind[rng.choices(kinds, weights)[0]])
                policy.steps = []
                try:
                    r = arena.play_fight(sc, policy)
                except ArenaError as e:
                    print(f"  skipped {sc['encounter']}: {e}", flush=True)
                    continue
                except (GameNotRunning, StateTimeout) as e:
                    print(f"  game not responding during {sc['encounter']}: {e}", flush=True)
                    fresh_game("not responding")
                    continue
                if not policy.steps:
                    continue
                potions = sum(1 for s in policy.steps if s["a"][s["y"]][0] == KIND_POTION)
                g = assign_rewards(policy.steps, r)
                batch += policy.steps
                results.append({"encounter": r.encounter, "kind": r.kind, "hp_start": r.hp_start,
                                "hp_end": r.hp_end, "max_hp": r.max_hp, "died": r.died, "return": round(g, 4),
                                "potions": potions, "iteration": it})
                fight_log.write(json.dumps(results[-1]) + "\n")
            if not batch:
                continue
            warmup = it <= args.value_warmup
            stats = ppo_update(model, ref, batch, warm_opt if warmup else opt, device, kl_coef=args.kl,
                               policy_coef=0.0 if warmup else 1.0)
            n = len(results)
            summary = {
                "iteration": it, "fights": n, "decisions": len(batch),
                "mean_return": round(sum(r["return"] for r in results) / n, 4),
                "hp_lost": round(sum(r["hp_start"] - r["hp_end"] for r in results) / n, 2),
                "deaths": sum(r["died"] for r in results), "seconds": round(time.monotonic() - t0, 1),
                "value_warmup": warmup, **stats}
            history.append(summary)
            print(json.dumps(summary), flush=True)
            ckpt_out = {"state_dict": model.state_dict(), "config": cfg, "iteration": it}
            torch.save(ckpt_out, args.out / "model.pt")
            if it % 25 == 0:  # snapshots to evaluate how training progressed
                snap = args.out / f"it{it:03d}"
                snap.mkdir(exist_ok=True)
                torch.save(ckpt_out, snap / "model.pt")
                vocab.save(snap / "vocab.json")
            history_file.write_text(json.dumps(history, indent=1))
            fight_log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
