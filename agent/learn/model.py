"""Combat policy/value network.

A small transformer reads one token for the player (numbers + pooled powers and
piles), one per hand card, one per enemy and one per potion. Every legal action
is then scored from the tokens it involves (card, target, potion) plus the
player token, so the output is exactly one logit per legal action. The value
head predicts how the fight ends (HP fraction left, death), which later RL uses
as a baseline.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from agent.learn.features import KIND_CARD, KIND_POTION, N_CARD_NUM, N_ENEMY_NUM, N_GLOBAL_NUM


class CombatNet(nn.Module):
    def __init__(self, sizes: dict[str, int], d: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.d = d
        self.card_emb = nn.Embedding(sizes["cards"], d, padding_idx=0)
        self.enemy_emb = nn.Embedding(sizes["enemies"], d, padding_idx=0)
        self.power_emb = nn.Embedding(sizes["powers"], d, padding_idx=0)
        self.potion_emb = nn.Embedding(sizes["potions"], d, padding_idx=0)
        self.card_num = nn.Linear(N_CARD_NUM, d)
        self.enemy_num = nn.Linear(N_ENEMY_NUM, d)
        self.glob_num = nn.Linear(N_GLOBAL_NUM, d)
        self.power_amt = nn.Linear(1, d)
        self.pile_src = nn.Embedding(3, d)
        self.pool = nn.Linear(3 * d, d)
        self.type_emb = nn.Embedding(4, d)  # player, card, enemy, potion
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.no_target = nn.Parameter(torch.zeros(d))
        self.kind_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(),
                                    nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action logits [B, A] with -inf on padding, value [B, 2])."""
        powers = self.power_emb(b["power_ids"]) + self.power_amt(b["power_amt"].unsqueeze(-1))
        piles = self.card_emb(b["pile_ids"]) + self.pile_src(b["pile_src"])
        pots = self.potion_emb(b["potion_ids"])
        pooled = self.pool(torch.cat([self._masked_mean(powers, b["power_ids"] > 0),
                                      self._masked_mean(piles, b["pile_ids"] > 0),
                                      self._masked_mean(pots, b["potion_ids"] > 0)], -1))
        g = self.glob_num(b["glob"]) + pooled + self.type_emb.weight[0]
        hand = self.card_emb(b["hand_ids"]) + self.card_num(b["hand_num"]) + self.type_emb.weight[1]
        enemies = self.enemy_emb(b["enemy_ids"]) + self.enemy_num(b["enemy_num"]) + self.type_emb.weight[2]
        pots = pots + self.type_emb.weight[3]
        tokens = torch.cat([g.unsqueeze(1), hand, enemies, pots], 1)
        pad = torch.cat([torch.zeros_like(b["hand_ids"][:, :1], dtype=torch.bool),
                         b["hand_ids"] == 0, b["enemy_ids"] == 0, b["potion_ids"] == 0], 1)
        h = self.encoder(tokens, src_key_padding_mask=pad)

        B, H, E = b["hand_ids"].shape[0], b["hand_ids"].shape[1], b["enemy_ids"].shape[1]
        kind, card, target, potion = b["act_kind"], b["act_card"], b["act_target"], b["act_potion"]

        def gather(offset: int, idx: torch.Tensor) -> torch.Tensor:
            pos = (offset + idx.clamp(min=0)).unsqueeze(-1).expand(-1, -1, self.d)
            return torch.gather(h, 1, pos)

        g_out = h[:, :1].expand(-1, kind.shape[1], -1)
        card_h = gather(1, card) * (kind == KIND_CARD).unsqueeze(-1)
        pot_h = gather(1 + H + E, potion) * (kind == KIND_POTION).unsqueeze(-1)
        target_h = torch.where((target >= 0).unsqueeze(-1), gather(1 + H, target), self.no_target)
        feats = torch.cat([g_out + self.kind_emb(kind.clamp(min=0)), card_h + pot_h, target_h,
                           g_out * (card_h + pot_h)], -1)
        logits = self.scorer(feats).squeeze(-1)
        logits = logits.masked_fill(kind < 0, float("-inf"))
        return logits, self.value(h[:, 0])


def _pad(rows: list[list], fill: Any, width: int | None = None) -> list[list]:
    n = max(1, max((len(r) for r in rows), default=0))
    return [list(r) + [fill if width is None else [fill] * width] * (n - len(r)) for r in rows]


def collate(examples: list[dict], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    xs = [e["x"] for e in examples]
    t = lambda v, dt=torch.long: torch.tensor(v, dtype=dt, device=device)  # noqa: E731
    acts = [e["a"] for e in examples]
    batch = {
        "hand_ids": t(_pad([x["hand_ids"] for x in xs], 0)),
        "hand_num": t(_pad([x["hand_num"] for x in xs], 0.0, N_CARD_NUM), torch.float),
        "enemy_ids": t(_pad([x["enemy_ids"] for x in xs], 0)),
        "enemy_num": t(_pad([x["enemy_num"] for x in xs], 0.0, N_ENEMY_NUM), torch.float),
        "glob": t([x["glob"] for x in xs], torch.float),
        "power_ids": t(_pad([x["power_ids"] for x in xs], 0)),
        "power_amt": t(_pad([x["power_amt"] for x in xs], 0.0), torch.float),
        "potion_ids": t(_pad([x["potion_ids"] for x in xs], 0)),
        "pile_ids": t(_pad([x["pile_ids"] for x in xs], 0)),
        "pile_src": t(_pad([x["pile_src"] for x in xs], 0)),
        "act_kind": t(_pad([[a[0] for a in r] for r in acts], -1)),
        "act_card": t(_pad([[a[1] for a in r] for r in acts], -1)),
        "act_target": t(_pad([[a[2] for a in r] for r in acts], -1)),
        "act_potion": t(_pad([[a[3] for a in r] for r in acts], -1)),
    }
    # Identifies identical card copies (for metrics): hash of the card ID and numbers.
    keys = [[hash((x["hand_ids"][a[1]], tuple(x["hand_num"][a[1]]))) % (1 << 62) if a[1] >= 0 else -1
             for a in r] for x, r in zip(xs, acts)]
    batch["act_card_key"] = t(_pad(keys, -2))
    if "y" in examples[0]:
        batch["y"] = t([e["y"] for e in examples])
    if "v" in examples[0]:
        batch["v"] = t([list(e["v"]) for e in examples], torch.float)
    return batch
