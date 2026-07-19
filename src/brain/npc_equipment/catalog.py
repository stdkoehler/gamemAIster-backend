"""Loader and lookup helpers for the per-system equipment catalogs.

Each catalog is a flat ``{"metadata": ..., "items": [...], "npc_equipment_sets": [...]}``
JSON file normalized from the source TTRPG rulebooks. Field coverage varies per
system (e.g. Shadowrun items only have name/category/cost/source, Vampire items
additionally have subcategory/type/damage/availability) but ``name`` and
``category`` are always present. A few systems need special handling, all
addressed in `load_catalog`/`filter_equipment` below:
- Call of Cthulhu prices items per era (`cost_by_era`); this app pins to the
  1920s and drops items that don't exist yet in that era.
- The Expanse has no price at all, only an Availability TN; equipment-budget
  filtering uses that field in place of a price (see `_COST_FIELD`).
- Seventh Sea's Armor items store their defensive rating under
  `damage_reduction` instead of `damage` (every other system's convention for
  an armor rating); normalized onto `damage` so callers (and the NPC/PC
  equipment-suggestion UI, which reads `damage` uniformly across systems)
  don't need a per-system special case.
"""

import json
from functools import lru_cache
from pathlib import Path

from src.routers.schema.mission import EquipmentType, GameType

_DATA_DIR = Path(__file__).parent / "data"

# Call of Cthulhu items price by era (`cost_by_era: {"1890s": ..., "1920s": ...,
# "modern": ...}`) instead of a flat `cost`. This app only ever runs CoC
# missions in the 1920s (see npc_models._merge_cthulhu's hardcoded
# `"era": "1920s"`), so we pin to that era and drop items that didn't exist
# yet (null 1920s entry) rather than guess across eras.
_COC_ERA = "1920s"

# Some systems express equipment access as a non-currency numeric field. The
# Expanse catalog has no price at all, only an AGE-system Availability Test
# Number (rarer/more restricted gear = higher TN) — used here as the
# `max_cost`-filterable field in place of a price.
_COST_FIELD: dict[GameType, str] = {
    GameType.EXPANSE: "availability_tn",
}


@lru_cache
def load_catalog(game_type: GameType) -> list[dict]:
    """Returns the flat ``items`` list for a system's equipment catalog."""
    path = _DATA_DIR / f"{game_type.value}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    if game_type == GameType.CALL_OF_CTHULHU:
        items = _normalize_coc_items(items)
    elif game_type == GameType.SEVENTH_SEA:
        items = _normalize_seventh_sea_items(items)
    return items


def _normalize_coc_items(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        cost_by_era = item.get("cost_by_era")
        if not cost_by_era:
            normalized.append(item)
            continue
        cost_1920s = cost_by_era.get(_COC_ERA)
        if cost_1920s is None:
            continue  # didn't exist yet in the 1920s; not offerable to NPCs
        normalized.append({**item, "cost": cost_1920s})
    return normalized


def _normalize_seventh_sea_items(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        damage_reduction = item.get("damage_reduction")
        if damage_reduction is None:
            normalized.append(item)
            continue
        normalized.append({**item, "damage": damage_reduction})
    return normalized


def filter_equipment(
    game_type: GameType,
    category: str | None = None,
    subcategory: str | None = None,
    max_cost: float | None = None,
    keywords: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Filters a system's equipment catalog by category/subcategory/cost/keywords.

    `max_cost` filters against whichever field represents this system's
    equipment-access constraint (a price for most systems, an Availability TN
    for Expanse — see `_COST_FIELD`); it only filters items whose value
    parses as a plain number (many systems use non-numeric costs like "1 WP",
    which are left in regardless so the agent can inspect them via the
    returned notes).
    """
    items = load_catalog(game_type)
    cost_field = _COST_FIELD.get(game_type, "cost")
    results = []
    for item in items:
        if category and category.lower() not in str(item.get("category", "")).lower():
            continue
        if subcategory and subcategory.lower() not in str(
            item.get("subcategory", "")
        ).lower():
            continue
        if keywords:
            haystack = " ".join(
                str(item.get(k, ""))
                for k in ("name", "keywords", "notes", "type")
            ).lower()
            if keywords.lower() not in haystack:
                continue
        if max_cost is not None:
            cost_value = _parse_numeric_cost(item.get(cost_field))
            if cost_value is not None and cost_value > max_cost:
                continue
        results.append(item)
        if len(results) >= limit:
            break
    return results


def get_item(game_type: GameType, name: str) -> dict | None:
    """Looks up a single item by exact (case-insensitive) name match."""
    name_lower = name.lower()
    for item in load_catalog(game_type):
        if str(item.get("name", "")).lower() == name_lower:
            return item
    return None


# Combat/field-relevant category substrings per system, used to build a bounded
# candidate slice for the NPC equipment prompt (each catalog's real ``category``
# vocabulary, not a generic guess — see filter_equipment's substring matching).
_NPC_EQUIPMENT_CATEGORIES: dict[GameType, list[str]] = {
    GameType.SHADOWRUN: [
        "Pistols", "Rifles", "Shotguns", "Submachine Guns", "Machine Guns",
        "Assault Cannons", "Blades", "Clubs", "Bows", "Crossbows", "Unarmed",
        "Tasers", "Holdouts", "Exotic Melee Weapons", "Exotic Ranged Weapons",
        "Improvised Weapons", "Underbarrel Weapons", "Bio-Weapon", "Cyberweapon",
        "Armor", "Gear", "Survival Gear", "Tools of the Trade", "Commlinks",
        "Health", "Headware", "Bodyware", "Eyeware", "Earware", "Cyberlimb",
    ],
    GameType.VAMPIRE_THE_MASQUERADE: ["Melee", "Ranged", "Heavy", "Armor", "Gear", "Vampire Gear"],
    GameType.CALL_OF_CTHULHU: [
        "weapons", "investigative_and_field_gear", "medical_supplies_and_forensics",
    ],
    GameType.SEVENTH_SEA: ["Melee", "Ranged", "Firearms", "Offhand", "Explosives", "Shield", "Armor", "Gear"],
    GameType.EXPANSE: [
        "weapons", "ship_weapons_and_vehicle_equipment", "armor_and_apparel",
        "environment_suits_and_survival", "electronics_tools_and_scifi_tech",
        "medical_supplies_and_pharmaceuticals",
    ],
    GameType.SLAVIC: ["Melee", "Ranged", "Armor", "Helmet", "Shield", "Container", "Tools", "Trade"],
    GameType.DRAGONLANCE: ["Melee", "Ranged", "Armor", "Shield", "Adventuring Gear", "Tools", "Trade Goods"],
    GameType.DESOLATE_FRONTIER: ["Melee", "Ranged", "Armor", "Container", "Tools", "Trade"],
}

# Item fields worth showing the LLM when picking equipment; everything else
# (e.g. ``source``) is noise that just inflates the prompt.
_NPC_EQUIPMENT_ITEM_FIELDS = (
    "name", "category", "subcategory", "cost", "damage", "type", "availability",
    "availability_tn", "notes",
)

# Maps each system's real catalog categories (see the per-system lists above)
# onto the generic equipment slots a PC/NPC sheet actually has fields for
# (`ShadowrunCharacter.weapons`/`.armor`/`.cyberware`/`.gear`, etc. in the
# frontend's CharacterProps.tsx). A system that has no field for a slot (e.g.
# no system but Shadowrun has cyberware) simply omits that key — see
# `get_equipment_types`.
_EQUIPMENT_TYPE_CATEGORIES: dict[GameType, dict[EquipmentType, list[str]]] = {
    GameType.SHADOWRUN: {
        EquipmentType.WEAPONS: [
            "Pistols", "Rifles", "Shotguns", "Submachine Guns", "Machine Guns",
            "Assault Cannons", "Blades", "Clubs", "Bows", "Crossbows", "Unarmed",
            "Tasers", "Holdouts", "Exotic Melee Weapons", "Exotic Ranged Weapons",
            "Improvised Weapons", "Underbarrel Weapons", "Bio-Weapon", "Cyberweapon",
        ],
        EquipmentType.ARMOR: ["Armor"],
        EquipmentType.CYBERWARE: [
            "Headware", "Bodyware", "Eyeware", "Earware", "Cyberlimb", "Health",
        ],
        EquipmentType.GEAR: ["Gear", "Survival Gear", "Tools of the Trade", "Commlinks"],
    },
    GameType.VAMPIRE_THE_MASQUERADE: {
        EquipmentType.WEAPONS: ["Melee", "Ranged", "Heavy"],
        EquipmentType.ARMOR: ["Armor"],
        EquipmentType.GEAR: ["Gear", "Vampire Gear"],
    },
    GameType.CALL_OF_CTHULHU: {
        EquipmentType.WEAPONS: ["weapons"],
        EquipmentType.GEAR: [
            "investigative_and_field_gear", "medical_supplies_and_forensics",
        ],
    },
    GameType.SEVENTH_SEA: {
        EquipmentType.WEAPONS: ["Melee", "Ranged", "Firearms", "Offhand", "Explosives"],
        EquipmentType.ARMOR: ["Armor", "Shield"],
        EquipmentType.GEAR: ["Gear"],
    },
    GameType.EXPANSE: {
        EquipmentType.WEAPONS: ["weapons", "ship_weapons_and_vehicle_equipment"],
        EquipmentType.ARMOR: ["armor_and_apparel", "environment_suits_and_survival"],
        EquipmentType.GEAR: [
            "electronics_tools_and_scifi_tech", "medical_supplies_and_pharmaceuticals",
        ],
    },
    GameType.SLAVIC: {
        EquipmentType.WEAPONS: ["Melee", "Ranged"],
        EquipmentType.ARMOR: ["Armor", "Helmet", "Shield"],
        EquipmentType.GEAR: ["Container", "Tools", "Trade"],
    },
    GameType.DRAGONLANCE: {
        EquipmentType.WEAPONS: ["Melee", "Ranged"],
        EquipmentType.ARMOR: ["Armor", "Shield"],
        EquipmentType.GEAR: ["Adventuring Gear", "Tools", "Trade Goods"],
    },
    GameType.DESOLATE_FRONTIER: {
        EquipmentType.WEAPONS: ["Melee", "Ranged"],
        EquipmentType.ARMOR: ["Armor"],
        EquipmentType.GEAR: ["Container", "Tools", "Trade"],
    },
}


def get_equipment_types(game_type: GameType) -> list[EquipmentType]:
    """Returns the equipment slots this system's catalog can suggest items for."""
    return list(_EQUIPMENT_TYPE_CATEGORIES.get(game_type, {}).keys())


def equipment_by_type(
    game_type: GameType,
    equipment_type: EquipmentType,
    max_cost: float | None = None,
    keywords: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Returns catalog items for one equipment slot (weapons/armor/cyberware/gear),
    trimmed to the fields worth showing (see `_NPC_EQUIPMENT_ITEM_FIELDS`), for
    use as character-sheet equipment suggestions.

    Raises `ValueError` if `equipment_type` isn't offered for this system (see
    `get_equipment_types`) — e.g. asking Vampire for cyberware.
    """
    categories = _EQUIPMENT_TYPE_CATEGORIES.get(game_type, {}).get(equipment_type)
    if categories is None:
        raise ValueError(
            f"{game_type.value} has no {equipment_type.value} slot; "
            f"available: {[t.value for t in get_equipment_types(game_type)]}"
        )
    seen_names: set[str] = set()
    results: list[dict] = []
    for category in categories:
        for item in filter_equipment(
            game_type, category=category, max_cost=max_cost, keywords=keywords, limit=limit
        ):
            name = str(item.get("name", ""))
            if name in seen_names:
                continue
            seen_names.add(name)
            results.append({k: item[k] for k in _NPC_EQUIPMENT_ITEM_FIELDS if k in item})
            if len(results) >= limit:
                return results
    return results


def get_npc_equipment_categories(game_type: GameType) -> list[str]:
    """
    Returns the valid, exact equipment-category vocabulary for this system
    (the same categories `npc_equipment_candidates` samples from by default).
    Used to tell the Stats step which category strings it's allowed to pick
    when narrowing equipment to the NPC's archetype.
    """
    return list(_NPC_EQUIPMENT_CATEGORIES.get(game_type, []))


def npc_equipment_candidates(
    game_type: GameType,
    categories: list[str] | None = None,
    max_cost: float | None = None,
    limit_per_category: int = 8,
    total_limit: int = 60,
) -> list[dict]:
    """
    Builds a bounded, deterministic candidate list of catalog items relevant
    to NPC equipping (no LLM/tool call involved): walks this system's
    combat/field-relevant categories and samples a few items from each,
    deduped by name, trimmed to the fields worth showing the LLM.

    `categories`, if given, narrows the walk to the NPC-specific subset
    chosen by the Stats step (validated against the system's known
    vocabulary; unknown/empty selections fall back to the full vocabulary so
    a bad pick from the LLM never yields zero candidates).
    """
    valid_categories = _NPC_EQUIPMENT_CATEGORIES.get(game_type, [])
    if categories:
        valid_lower = {c.lower() for c in valid_categories}
        categories = [c for c in categories if c.lower() in valid_lower]
    categories = categories or valid_categories
    seen_names: set[str] = set()
    candidates: list[dict] = []
    for category in categories:
        for item in filter_equipment(
            game_type, category=category, max_cost=max_cost, limit=limit_per_category
        ):
            name = str(item.get("name", ""))
            if name in seen_names:
                continue
            seen_names.add(name)
            candidates.append({k: item[k] for k in _NPC_EQUIPMENT_ITEM_FIELDS if k in item})
            if len(candidates) >= total_limit:
                return candidates
    return candidates


def _parse_numeric_cost(cost: object) -> float | None:
    if isinstance(cost, (int, float)):
        return float(cost)
    if isinstance(cost, str):
        digits = "".join(ch for ch in cost if ch.isdigit() or ch == ".")
        if digits:
            try:
                return float(digits)
            except ValueError:
                return None
    return None
