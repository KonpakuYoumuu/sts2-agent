"""HTTP client for the STS2MCP mod (localhost REST API).

Wraps GET/POST on /api/v1/singleplayer with retries, and knows when the game
is "settled" (waiting for player input) versus mid-animation / loading.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = "http://localhost:15526"

State = dict[str, Any]

TIMING_RETRY_ACTIONS = {"proceed", "end_turn"}
# Mod errors meaning "the screen hasn't finished opening", not "this move is illegal".
# E.g. "Rest site room is not open", "Shop inventory not ready yet".
TRANSIENT_ERROR = re.compile(r"is not open|not ready|currently disabled", re.IGNORECASE)


class GameNotRunning(RuntimeError):
    """The mod's HTTP server can't be reached."""


class StateTimeout(RuntimeError):
    """The game didn't reach an actionable state in time."""

    def __init__(self, message: str, last_state: State | None):
        super().__init__(message)
        self.last_state = last_state


@dataclass
class ActionResult:
    ok: bool
    message: str
    raw: dict[str, Any]


class GameClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 10.0,
                 poll_interval: float = 0.03):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    # ---- raw HTTP ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None,
                 retries: int = 3) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        last_err: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(self.base_url + path, data=data,
                                         headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # The mod reports action errors with a JSON body and non-200 code.
                try:
                    return json.loads(e.read().decode("utf-8"))
                except (ValueError, OSError):
                    last_err = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
            time.sleep(0.5 * (attempt + 1))
        raise GameNotRunning(
            f"Can't reach the STS2MCP mod at {self.base_url} ({last_err}). "
            "Is the game running with the mod enabled?")

    def ping(self) -> bool:
        try:
            return self._request("GET", "/", retries=1).get("status") == "ok"
        except GameNotRunning:
            return False

    def get_state(self) -> State:
        return self._request("GET", "/api/v1/singleplayer?format=json")

    def act(self, action: dict[str, Any]) -> ActionResult:
        result = self._post_action(action)
        # Timing-sensitive actions: e.g. a shop's first `proceed` only starts closing
        # the inventory, and the proceed button enables when the animation ends.
        # Don't GET state between attempts: reading shop state reopens the inventory.
        attempts = 0
        while not result.ok and attempts < 10 and (
                action.get("action") in TIMING_RETRY_ACTIONS or TRANSIENT_ERROR.search(result.message)):
            time.sleep(0.3)
            result = self._post_action(action)
            attempts += 1
        return result

    def _post_action(self, action: dict[str, Any]) -> ActionResult:
        raw = self._request("POST", "/api/v1/singleplayer", action, retries=1)
        ok = raw.get("status") == "ok"
        return ActionResult(ok=ok, message=raw.get("message") or raw.get("error") or "", raw=raw)

    def profiles(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/profiles")

    # ---- waiting ----------------------------------------------------------

    def wait_for_input(self, timeout: float = 30.0) -> State:
        """Poll until the game is waiting for a player decision."""
        deadline = time.monotonic() + timeout
        state: State | None = None
        while time.monotonic() < deadline:
            state = self.get_state()
            if is_settled(state):
                return state
            time.sleep(self.poll_interval)
        raise StateTimeout(f"Game not actionable after {timeout:.0f}s "
                           f"(state_type={state and state.get('state_type')})", state)


COMBAT_TYPES = {"monster", "elite", "boss"}
# After the final boss the game shows this event, then the game-over screen.
VICTORY_EVENT = "THE_ARCHITECT"


def is_victory_event(state: State) -> bool:
    return state.get("state_type") == "event" and (state.get("event") or {}).get("event_id") == VICTORY_EVENT


def is_settled(state: State) -> bool:
    """True when the game is waiting on the player, not animating/loading."""
    st = state.get("state_type")
    if st in (None, "unknown"):
        return False
    if st in COMBAT_TYPES:
        battle = state.get("battle") or {}
        if not battle.get("is_play_phase") or battle.get("turn") != "player":
            return False
        enemies = battle.get("enemies", [])
        # Between the last enemy dying and the rewards screen there is no input...
        if enemies and not any(e.get("hp", 0) > 0 for e in enemies):
            return False
        # ...but an empty list can also be a boss phase with nothing targetable,
        # where the player still has a normal turn (skills, end turn).
        return True
    if st == "treasure":
        t = state.get("treasure") or {}
        return "relics" in t or bool(t.get("can_proceed"))
    if st == "shop":
        return "error" not in (state.get("shop") or {})
    if st == "event":
        ev = state.get("event") or {}
        # Options are briefly empty while the event loads.
        return bool(ev.get("in_dialogue") or ev.get("options"))
    if st == "menu":
        m = state.get("menu_screen")
        # Right after launch the main menu briefly has no options while it loads.
        if m == "main" and not state.get("options") and not state.get("blocked_options"):
            return False
        return not state.get("loading") and m not in (None, "loading")
    return True
