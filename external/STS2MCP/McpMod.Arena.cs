using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace STS2_MCP;

// The "fight arena": lets an agent set up a player (deck, relics, HP) and start
// any fight directly, using the game's own developer-console commands and model
// functions, so combat can be trained without playing whole runs.
public static partial class McpMod
{
    private static DevConsole? _devConsole;
    private static Task? _arenaSetupTask;

    /// <summary>`console {command}`: run a developer-console command, e.g. "fight SHRINKER_BEETLE_WEAK".</summary>
    private static Dictionary<string, object?> ExecuteConsoleCommand(Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("command", out var cmdElem) || cmdElem.ValueKind != JsonValueKind.String)
            return Error("Missing 'command' (a developer-console command line, e.g. 'fight SHRINKER_BEETLE_WEAK')");
        _devConsole ??= new DevConsole(shouldAllowDebugCommands: true);
        var result = _devConsole.ProcessCommand(cmdElem.GetString()!);
        if (!result.success)
            return Error(result.msg);
        return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = result.msg };
    }

    /// <summary>
    /// `arena_setup {deck?: [{id, upgrades?}], relics?: [id], max_hp?, hp?}`: replace the deck,
    /// relics and HP outside combat. Relics are added without their pickup effects (no card
    /// choices or max-HP gains). Runs asynchronously; poll `arena_status`.
    /// </summary>
    private static Dictionary<string, object?> ExecuteArenaSetup(Player player, Dictionary<string, JsonElement> data)
    {
        if (CombatManager.Instance.IsInProgress)
            return Error("arena_setup only works outside combat");
        if (_arenaSetupTask is { IsCompleted: false })
            return Error("arena_setup already running");

        List<(CardModel model, int upgrades)>? deck = null;
        if (data.TryGetValue("deck", out var deckElem))
        {
            deck = new();
            foreach (var c in deckElem.EnumerateArray())
            {
                var id = c.GetProperty("id").GetString()!.ToUpperInvariant();
                var model = ModelDb.AllCards.FirstOrDefault(m => m.Id.Entry == id);
                if (model == null)
                    return Error($"Card '{id}' not found");
                var upgrades = c.TryGetProperty("upgrades", out var u) ? u.GetInt32() : 0;
                deck.Add((model, upgrades));
            }
        }
        List<RelicModel>? relics = null;
        if (data.TryGetValue("relics", out var relicsElem))
        {
            relics = new();
            foreach (var r in relicsElem.EnumerateArray())
            {
                var id = r.GetString()!.ToUpperInvariant();
                var model = ModelDb.AllRelics.FirstOrDefault(m => m.Id.Entry == id);
                if (model == null)
                    return Error($"Relic '{id}' not found");
                relics.Add(model);
            }
        }
        int? maxHp = data.TryGetValue("max_hp", out var mh) ? mh.GetInt32() : null;
        int? hp = data.TryGetValue("hp", out var h) ? h.GetInt32() : null;

        _arenaSetupTask = TaskHelper.RunSafely(ArenaSetupAsync(player, deck, relics, maxHp, hp));
        return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Arena setup started" };
    }

    private static async Task ArenaSetupAsync(Player player, List<(CardModel model, int upgrades)>? deck,
                                              List<RelicModel>? relics, int? maxHp, int? hp)
    {
        var runState = RunManager.Instance.DebugOnlyGetState()!;
        if (deck != null)
        {
            await CardPileCmd.RemoveFromDeck(player.Deck.Cards.ToList(), showPreview: false);
            foreach (var (model, upgrades) in deck)
            {
                var card = runState.CreateCard(model, player);
                for (var i = 0; i < upgrades && card.IsUpgradable; i++)
                {
                    card.UpgradeInternal();
                    card.FinalizeUpgradeInternal();
                }
                await CardPileCmd.Add(card, PileType.Deck, skipVisuals: true);
            }
        }
        if (relics != null)
        {
            foreach (var relic in player.Relics.ToList())
                player.RemoveRelicInternal(relic, silent: true);
            foreach (var relic in relics)
                player.AddRelicInternal(relic.ToMutable(), -1, silent: true);
        }
        if (maxHp != null)
            await CreatureCmd.SetMaxHp(player.Creature, maxHp.Value);
        if (hp != null)
            await CreatureCmd.SetCurrentHp(player.Creature, hp.Value);
    }

    /// <summary>`arena_status`: whether the last arena_setup has finished.</summary>
    private static Dictionary<string, object?> ExecuteArenaStatus()
    {
        var task = _arenaSetupTask;
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["setup_done"] = task == null || task.IsCompleted,
            ["setup_error"] = task?.Exception?.GetBaseException().Message,
        };
    }
}
