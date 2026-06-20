"""Loader and lookup helpers for the per-system equipment catalogs.

Each catalog is a flat ``{"metadata": ..., "items": [...], "npc_equipment_sets": [...]}``
JSON file normalized from the source TTRPG rulebooks. Field coverage varies per
system (e.g. Shadowrun items only have name/category/cost/source, Vampire items
additionally have subcategory/type/damage/availability) but ``name`` and
``category`` are always present.
"""

import json
from functools import lru_cache
from pathlib import Path

from src.routers.schema.mission import GameType

_DATA_DIR = Path(__file__).parent / "data"


@lru_cache
def load_catalog(game_type: GameType) -> list[dict]:
    """Returns the flat ``items`` list for a system's equipment catalog."""
    path = _DATA_DIR / f"{game_type.value}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


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

    `max_cost` only filters items whose `cost` field parses as a plain number
    (many systems use non-numeric costs like "1 WP" or era-keyed costs, which
    are left in regardless so the agent can inspect them via the returned notes).
    """
    items = load_catalog(game_type)
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
            cost_value = _parse_numeric_cost(item.get("cost"))
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
}

# Item fields worth showing the LLM when picking equipment; everything else
# (e.g. ``source``) is noise that just inflates the prompt.
_NPC_EQUIPMENT_ITEM_FIELDS = (
    "name", "category", "subcategory", "cost", "damage", "type", "availability", "notes",
)


def npc_equipment_candidates(
    game_type: GameType,
    max_cost: float | None = None,
    limit_per_category: int = 8,
    total_limit: int = 60,
) -> list[dict]:
    """
    Builds a bounded, deterministic candidate list of catalog items relevant
    to NPC equipping (no LLM/tool call involved): walks this system's
    combat/field-relevant categories and samples a few items from each,
    deduped by name, trimmed to the fields worth showing the LLM.
    """
    categories = _NPC_EQUIPMENT_CATEGORIES.get(game_type, [])
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
