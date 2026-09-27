# Setting up on a new PC

This gets the project from a fresh clone to a bot playing Slay the Spire 2. Windows only. Budget about an hour, most of it downloads.

## 0. Check the game version first

You need **Slay the Spire 2 on Steam, version v0.107.1**. The version is shown on the main menu and in `<game folder>\release_info.json`.

The game mod this project uses is only tested on v0.107.1, and upstream reports that **v0.111 breaks it**. If your game is newer, the mod may not load; tell the project owner before going further.

To stop Steam updating the game behind your back: Steam → Library → right-click Slay the Spire 2 → Properties → Updates → *Only update this game when I launch it*. Then always start the game through this project or with Steam in offline mode.

## 1. Tools

| Tool | How |
|---|---|
| Git | https://git-scm.com/download/win |
| Python 3.11+ (3.13 tested) | https://www.python.org/downloads/ (tick "Add to PATH") |
| .NET 9 SDK (to build the mod) | `winget install Microsoft.DotNet.SDK.9`, or the installer from https://dotnet.microsoft.com/download/dotnet/9.0 |

## 2. Get the code and the Python environment

```powershell
git clone <repository URL>
cd <repository folder>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q        # should print "31 passed"
```

Without an NVIDIA GPU, use `pip install torch` instead of the `cu128` line (CPU only; training takes minutes longer, playing is unaffected).

Always use `.venv\Scripts\python.exe` (or run `.\.venv\Scripts\Activate.ps1` once per terminal).

## 3. Build and install the game mod

The mod ([STS2MCP](https://github.com/Gennadiyev/STS2MCP), MIT) lets Python read the game state and send moves over `http://localhost:15526`. `external/STS2MCP` is upstream commit `55e0648` **plus our fixes** (listed in `patches/sts2mcp-changes.patch`); the upstream release doesn't work, so build this copy.

Close the game first, then (adjust the path if your Steam library is elsewhere):

```powershell
$env:STS2_GAME_DIR = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
cd external\STS2MCP
.\build.ps1
New-Item -ItemType Directory -Force "$env:STS2_GAME_DIR\mods\STS2_MCP"
Copy-Item out\STS2_MCP\STS2_MCP.dll "$env:STS2_GAME_DIR\mods\STS2_MCP\" -Force
Copy-Item mod_manifest.json "$env:STS2_GAME_DIR\mods\STS2_MCP\STS2_MCP.json" -Force
cd ..\..
```

For headless (no-window) runs, the game folder also needs a `steam_appid.txt` containing the game's Steam app ID:

```powershell
Set-Content "$env:STS2_GAME_DIR\steam_appid.txt" "2868840"
```

Writing into `Program Files` may need an Administrator terminal.

## 4. Set up the game

1. Start the game normally from Steam. Accept loading mods if it asks.
2. Settings: set the language to **English** (the bot reads card text).
3. Create a separate **profile** for the bot (e.g. profile 2) so it doesn't touch your own save, and switch to it.
4. New profiles must reveal **Timeline** unlocks by hand before Singleplayer is available: open Timeline on the main menu and click through the new unlocks. The bot stops with "manual action needed" whenever a new unlock appears; do the same then.
5. Leave the game on the main menu.

Check the connection:

```powershell
.\.venv\Scripts\python.exe -c "from agent.interface.client import GameClient; print(GameClient().ping())"
```

It should print `True`.

## 5. Play

```powershell
# One run in the open game window
.\.venv\Scripts\python.exe -m agent.harness.run --runs 1 --bot heuristic

# Many runs headless (close the game window first; ~75 runs/hour; Steam must be running)
.\.venv\Scripts\python.exe -m agent.harness.supervise --runs 50 --bot nn --log-dir logs/my_test
```

If the game isn't in the default Steam folder, keep `$env:STS2_GAME_DIR` set or pass `--game-dir`.

Bots: `random`, `heuristic` (rule-based), `nn` (the trained network in `models/combat_bc_v2`, included). Logs go to `logs/` and are not shared through git; collect your own or ask for a copy.

## Working together

Both of us push to the same GitHub repository.

- Before starting work: `git pull`.
- Do each piece of work on its own branch: `git switch -c my-change`, commit, `git push -u origin my-change`, then open a **pull request** on GitHub. The other person reviews and merges it.
- Don't commit `logs/`, `data/`, `.venv/` or `.tools/` (`.gitignore` already skips them).
- Run `python -m pytest -q` before pushing.
- Never commit decompiled game code. Reading the game's code for research is fine locally, but it's the developer's copyright.
