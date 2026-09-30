"""Reinforcement learning for combat in the fight arena (PPO).

Starts from the imitation-learned network and improves it by playing arena
fights. Each fight is one episode; its reward comes at the end:

    return = HP left / max HP  -  DEATH_PENALTY * died  -  POTION_COST * potions used

(potions are shared across a whole run, so using one must cost something).
The value estimate reuses the network's fight-outcome head:
V = sigmoid(hp) - DEATH_PENALTY * sigmoid(death). Advantages are Monte Carlo
(return - V): fights are short and the reward is only at the end.

A KL penalty to the frozen imitation network keeps early updates from
wrecking a policy that already plays sensibly. Only combat moves are learned;
selection screens inside a fight stay with the rule-based bot.

    python -m agent.learn.ppo --init models/combat_bc_v2 --out models/combat_ppo_v1 --iterations 50
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
from agent.interface.client import COMBAT_TYPES, GameClient, State
from agent.learn.features import KIND_POTION, Vocab, encode_actions, encode_state
from agent.learn.model import CombatNet, collate

DEATH_PENALTY = 0.5
POTION_COST = 0.05


def value_of(v: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(v[..., 0]) - DEATH_PENALTY * torch.sigmoid(v[..., 1])


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
        self.steps.append({**example, "y": idx, "logp": float(torch.log(probs[idx] + 1e-12)),
                           "v_old": float(value_of(value[0].cpu()))})
        self.last_info = {"p": round(float(probs[idx]), 3)}
        return actions[idx]


def fight_return(result, potions_used: int) -> float:
    frac = result.hp_end / max(result.max_hp, 1)
    return frac - DEATH_PENALTY * result.died - POTION_COST * potions_used


def ppo_update(model: CombatNet, ref: CombatNet, steps: list[dict], opt, device: str, epochs: int = 4,
               bs: int = 256, clip: float = 0.2, kl_coef: float = 0.1, ent_coef: float = 0.003) -> dict:
    # No dropout: the old log-probabilities were computed without it, so it would
    # make the policy look changed (ratio != 1) before any update.
    model.eval()
    adv = torch.tensor([s["ret"] - s["v_old"] for s in steps])
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
            loss = pg + 0.5 * v_loss + kl_coef * kl - ent_coef * entropy
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
    ap.add_argument("--out", type=Path, default=Path("models/combat_ppo_v1"))
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios_v1.jsonl"))
    ap.add_argument("--act", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--fights", type=int, default=40, help="fights per iteration")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.init / "model.pt", map_location=device)
    cfg = ckpt["config"]
    model = CombatNet(cfg["sizes"], cfg["d"], cfg["layers"], cfg["heads"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vocab = Vocab.load(args.init / "vocab.json")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    vocab.save(args.out / "vocab.json")
    scenarios = [s for s in load_scenarios(args.scenarios) if s["act"] == args.act]
    rng = random.Random(args.seed)
    client = GameClient()
    if not client.ping():
        print("Can't reach the STS2MCP mod. Start the headless game first.")
        return 1
    policy = SamplingPolicy(model, vocab, device, seed=args.seed)
    arena = Arena(client, HeuristicBot(seed=args.seed))
    history = []
    with gzip.open(args.out / "fights.jsonl.gz", "at", encoding="utf-8") as fight_log:
        for it in range(1, args.iterations + 1):
            t0 = time.monotonic()
            batch, results = [], []
            for _ in range(args.fights):
                sc = rng.choice(scenarios)
                policy.steps = []
                try:
                    r = arena.play_fight(sc, policy)
                except ArenaError as e:
                    print(f"  skipped {sc['encounter']}: {e}", flush=True)
                    continue
                if not policy.steps:
                    continue
                potions = sum(1 for s in policy.steps if s["a"][s["y"]][0] == KIND_POTION)
                g = fight_return(r, potions)
                for s in policy.steps:
                    s["ret"] = g
                batch += policy.steps
                results.append({"encounter": r.encounter, "kind": r.kind, "hp_start": r.hp_start,
                                "hp_end": r.hp_end, "max_hp": r.max_hp, "died": r.died, "return": round(g, 4),
                                "potions": potions, "iteration": it})
                fight_log.write(json.dumps(results[-1]) + "\n")
            if not batch:
                continue
            stats = ppo_update(model, ref, batch, opt, device, kl_coef=args.kl)
            n = len(results)
            summary = {
                "iteration": it, "fights": n, "decisions": len(batch),
                "mean_return": round(sum(r["return"] for r in results) / n, 4),
                "hp_lost": round(sum(r["hp_start"] - r["hp_end"] for r in results) / n, 2),
                "deaths": sum(r["died"] for r in results), "seconds": round(time.monotonic() - t0, 1), **stats}
            history.append(summary)
            print(json.dumps(summary), flush=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "iteration": it}, args.out / "model.pt")
            (args.out / "history.json").write_text(json.dumps(history, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
