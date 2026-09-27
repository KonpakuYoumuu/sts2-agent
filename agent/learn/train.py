"""Train the combat network by behavior cloning (plus the fight-outcome value head).

    python -m agent.learn.train --data data/combat_v1 --out models/combat_bc_v1
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from agent.learn.features import KIND_END, Vocab
from agent.learn.model import CombatNet, collate


def losses(model: CombatNet, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits, value = model(batch)
    policy_loss = F.cross_entropy(logits, batch["y"])
    known = batch["v"][:, 0] >= 0
    v_loss = torch.tensor(0.0, device=logits.device)
    if known.any():
        v = value[known]
        target = batch["v"][known]
        v_loss = F.mse_loss(torch.sigmoid(v[:, 0]), target[:, 0]) + \
            F.binary_cross_entropy_with_logits(v[:, 1], target[:, 1])
    pred = logits.argmax(-1)
    correct = pred == batch["y"]
    # Playing an identical copy of the card (same card ID, same numbers) on the
    # same target is the same move.
    same_card = (batch["act_card_key"].gather(1, pred.unsqueeze(1)) ==
                 batch["act_card_key"].gather(1, batch["y"].unsqueeze(1))).squeeze(1)
    same_target = (batch["act_target"].gather(1, pred.unsqueeze(1)) ==
                   batch["act_target"].gather(1, batch["y"].unsqueeze(1))).squeeze(1)
    same_kind = (batch["act_kind"].gather(1, pred.unsqueeze(1)) ==
                 batch["act_kind"].gather(1, batch["y"].unsqueeze(1))).squeeze(1)
    equivalent = correct | (same_card & same_target & same_kind)
    end_label = batch["act_kind"].gather(1, batch["y"].unsqueeze(1)).squeeze(1) == KIND_END
    stats = {"n": len(correct), "correct": correct.sum().item(), "equiv": equivalent.sum().item(),
             "end_n": end_label.sum().item(), "end_correct": (correct & end_label).sum().item()}
    return policy_loss + 0.5 * v_loss, {**stats, "policy_loss": policy_loss.item(), "value_loss": v_loss.item()}


def evaluate(model: CombatNet, data: list[dict], device: str, bs: int = 512) -> dict[str, float]:
    model.eval()
    tot = {"n": 0, "correct": 0, "equiv": 0, "end_n": 0, "end_correct": 0, "policy_loss": 0.0, "value_loss": 0.0}
    with torch.no_grad():
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            _, s = losses(model, collate(chunk, device))
            for k in ("n", "correct", "equiv", "end_n", "end_correct"):
                tot[k] += s[k]
            tot["policy_loss"] += s["policy_loss"] * len(chunk)
            tot["value_loss"] += s["value_loss"] * len(chunk)
    model.train()
    return {"acc": tot["correct"] / tot["n"], "equiv_acc": tot["equiv"] / tot["n"], "end_turn_acc": tot["end_correct"] / max(tot["end_n"], 1),
            "policy_loss": tot["policy_loss"] / tot["n"], "value_loss": tot["value_loss"] / tot["n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/combat_v1")
    ap.add_argument("--out", default="models/combat_bc_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "dataset.pkl", "rb") as f:
        data = pickle.load(f)
    vocab = Vocab.load(data_dir / "vocab.json")
    vocab.save(out / "vocab.json")
    config = {"sizes": vocab.sizes(), "d": 128, "layers": 3, "heads": 4}
    model = CombatNet(config["sizes"], config["d"], config["layers"], config["heads"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(data["train"]) // args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f"device={device} train={len(data['train'])} val={len(data['val'])} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    best, history = -1.0, []
    train = data["train"]
    for epoch in range(args.epochs):
        t0 = time.time()
        random.shuffle(train)
        run_loss = 0.0
        for i in range(0, len(train), args.bs):
            loss, _ = losses(model, collate(train[i:i + args.bs], device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
        val = evaluate(model, data["val"], device)
        val.update(epoch=epoch + 1, train_loss=run_loss / (len(train) / args.bs), seconds=round(time.time() - t0, 1))
        history.append(val)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in val.items()}), flush=True)
        if val["acc"] > best:
            best = val["acc"]
            torch.save({"state_dict": model.state_dict(), "config": config}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val acc {best:.4f}; saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
