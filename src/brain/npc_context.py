"""NPC roster cross-referencing for the mission's keyNPCs (or equivalent) field.

Two independent consumers, both needing the same roster lookup to avoid the
model treating an established NPC as a fresh, disconnected identity:
- the compression pipeline (entity/scene extraction) — get_npc_roster()
- the live story prompt (active-NPC gear/appearance override) — flag_active_key_npcs()
"""

import json

from src.brain.system_registry import NPC_CONFIGS


def get_npc_roster(mission_json: str, game_type) -> list[dict]:
    """Extracts the adventure's named NPC roster (`keyNPCs`, or the system's
    equivalent field) as lightweight identity anchors — name/role/affiliation
    only, no secrets — so the entity and scene compressors can recognize that
    a name or title mentioned in play refers to an NPC the adventure already
    established, instead of inventing a duplicate, disconnected identity for
    them. Mirrors the lookup `flag_active_key_npcs` does for the story
    prompt, but for the compression pipeline rather than the live turn.

    Systems without an `NPC_CONFIGS` entry (e.g. Custom) or without a roster
    field present return an empty list.
    """
    npc_cfg = NPC_CONFIGS.get(game_type)
    if npc_cfg is None:
        return []
    try:
        parsed = json.loads(mission_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    roster = parsed.get(npc_cfg.npc_roster_key)
    if not isinstance(roster, list):
        return []

    anchors = []
    for entry in roster:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        anchors.append(
            {
                "name": entry["name"],
                "role": entry.get("role", ""),
                "affiliation": entry.get("affiliation", ""),
            }
        )
    return anchors


def flag_active_key_npcs(
    mission_json: str, game_type, active_npcs: list[tuple[str, str]]
) -> str:
    """Annotates `keyNPCs` entries (or the system's equivalent roster field)
    that correspond to a currently-active NPC, so the model isn't handed two
    silently conflicting descriptions of the same character — the adventure's
    static roster entry, and the live, scene-accurate Active NPCs entry.

    `active_npcs` is a list of `(name, matched_key_npc)` pairs for active
    NPCs that have a recorded match — the roster entry name the NPC
    Profiler matched at generation time (see `Gamemaster.generate_npc` /
    `_validate_matched_key_npc`), already confirmed against this same
    roster. This is a direct, trusted lookup with no fuzziness: there is no
    name-based heuristic here, deliberately — a "Temple Guard" sharing a word
    with an unrelated "Temple Priest" roster entry is exactly the kind of
    false positive that kind of matching invites. An active NPC with no
    recorded match (manually-created sheets, or ones generated before this
    field existed) is simply not flagged; the story prompt instructs the
    model to recognize a name correspondence itself in that case.

    Two active NPCs recording the same `matched_key_npc` (e.g. the same
    character generated twice under different sheet names) makes that
    match ambiguous too — dropped rather than guessed at.

    The flag is scoped to gear/equipment/appearance only, not personality —
    the Active NPC's `description` is a thin profiler-generated blurb, while
    `keyNPCs` carries the richer motivations/secrets/relationships that
    should keep driving roleplay regardless of this flag.

    Systems without an `NPC_CONFIGS` entry (e.g. Custom) or without a roster
    field present are left untouched.
    """
    if not active_npcs:
        return mission_json
    npc_cfg = NPC_CONFIGS.get(game_type)
    if npc_cfg is None:
        return mission_json
    try:
        parsed = json.loads(mission_json)
    except json.JSONDecodeError:
        return mission_json
    if not isinstance(parsed, dict):
        return mission_json
    roster = parsed.get(npc_cfg.npc_roster_key)
    if not isinstance(roster, list):
        return mission_json

    matched_groups: dict[str, list[str]] = {}
    for name, matched in active_npcs:
        matched_groups.setdefault(matched.strip().lower(), []).append(name)
    unique_matches = {key: names[0] for key, names in matched_groups.items() if len(names) == 1}
    if not unique_matches:
        return mission_json

    for entry in roster:
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("name", "")).strip().lower()
        if not entry_name or entry_name not in unique_matches:
            continue
        active_name = unique_matches[entry_name]
        entry["_activeNpcOverride"] = (
            f'This NPC is currently active in the scene as "{active_name}" in Active NPCs '
            "below. Active NPCs is authoritative there for current gear, equipment, and "
            "physical appearance. This keyNPCs entry remains the source for backstory, "
            "personality, motivations, and secrets."
        )

    return json.dumps(parsed, ensure_ascii=False, indent=2)
