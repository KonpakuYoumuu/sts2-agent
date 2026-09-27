"""Common interface for every decision-maker (random, heuristic, neural)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.interface.actions import Action
from agent.interface.client import State


class Policy(ABC):
    name = "policy"

    @abstractmethod
    def choose(self, state: State, actions: list[Action]) -> Action:
        """Pick one of `actions` (non-empty, all legal) for `state`."""

    def on_run_start(self) -> None:
        """Hook called when a new run begins."""
