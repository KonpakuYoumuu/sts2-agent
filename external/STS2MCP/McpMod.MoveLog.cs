using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;

namespace STS2_MCP;

// Move log: with STS2MCP_MOVE_LOG=<file>, every combat move that starts executing is
// appended as one JSON line {t, state, action}, whoever made it (our bots, another mod,
// a human). `state` is the API's state at that moment; `action` is in the API's POST
// format. This records exact moves at any game speed, where polling the API can't.
public static partial class McpMod
{
    private static readonly string? MoveLogPath = System.Environment.GetEnvironmentVariable("STS2MCP_MOVE_LOG");
    private static readonly object MoveLogLock = new();
    private static readonly JsonSerializerOptions MoveLogJson = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    private static void LogMove(Func<Dictionary<string, object?>, Dictionary<string, object?>?> describe)
    {
        if (string.IsNullOrEmpty(MoveLogPath) || !CombatManager.Instance.IsInProgress)
            return;
        try
        {
            var state = BuildGameState();
            var action = describe(state);
            if (action == null)
                return;
            var line = JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["t"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                ["state"] = state,
                ["action"] = action,
            }, MoveLogJson);
            lock (MoveLogLock)
                File.AppendAllText(MoveLogPath, line + "\n");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP] Move log failed: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static List<Dictionary<string, object?>> StateList(Dictionary<string, object?> state, string section, string key)
    {
        return (state.GetValueOrDefault(section) as Dictionary<string, object?>)?.GetValueOrDefault(key)
            as List<Dictionary<string, object?>> ?? new List<Dictionary<string, object?>>();
    }

    /// <summary>The state's entity_id for a creature combat id, or null.</summary>
    private static string? EntityIdForCombatId(Dictionary<string, object?> state, object? combatId)
    {
        if (combatId == null)
            return null;
        string wanted = combatId.ToString()!;
        return StateList(state, "battle", "enemies")
            .FirstOrDefault(e => e.GetValueOrDefault("combat_id")?.ToString() == wanted)
            ?.GetValueOrDefault("entity_id") as string;
    }

    [HarmonyPatch(typeof(PlayCardAction), "ExecuteAction")]
    private static class LogPlayCard
    {
        private static void Prefix(PlayCardAction __instance)
        {
            LogMove(state =>
            {
                // Actions built from the network form (the usual path) resolve the card lazily.
                var card = Traverse.Create(__instance).Field("_card").GetValue<CardModel>()
                           ?? __instance.NetCombatCard.ToCardModelOrNull();
                var hand = __instance.Player?.PlayerCombatState?.Hand?.Cards?.ToList();
                int index = card != null && hand != null ? hand.IndexOf(card) : -1;
                var action = new Dictionary<string, object?>
                {
                    ["action"] = "play_card",
                    ["card_index"] = index,
                    ["card_id"] = card?.Id.Entry,
                };
                var handState = StateList(state, "player", "hand");
                if (index >= 0 && index < handState.Count
                    && handState[index].GetValueOrDefault("target_type") as string == "AnyEnemy")
                    action["target"] = EntityIdForCombatId(state, (object?)__instance.Target?.CombatId ?? __instance.TargetId);
                return action;
            });
        }
    }

    [HarmonyPatch(typeof(UsePotionAction), "ExecuteAction")]
    private static class LogUsePotion
    {
        private static void Prefix(UsePotionAction __instance)
        {
            LogMove(state =>
            {
                var action = new Dictionary<string, object?>
                {
                    ["action"] = "use_potion",
                    ["slot"] = __instance.PotionIndex,
                };
                var potion = StateList(state, "player", "potions")
                    .FirstOrDefault(p => p.GetValueOrDefault("slot")?.ToString() == __instance.PotionIndex.ToString());
                if (potion?.GetValueOrDefault("target_type") as string == "AnyEnemy")
                    action["target"] = EntityIdForCombatId(state, __instance.TargetId);
                return action;
            });
        }
    }

    [HarmonyPatch(typeof(EndPlayerTurnAction), "ExecuteAction")]
    private static class LogEndTurn
    {
        private static void Prefix() =>
            LogMove(_ => new Dictionary<string, object?> { ["action"] = "end_turn" });
    }
}
