"""Pydantic models and merge helpers for the 3-step NPC generation pipeline
(Profiler -> Stats -> Equipment).

NPC stats/equipment are intentionally less detailed than full player
CharacterSheets: each ``<System>NpcStats``/``<System>NpcEquipment`` pair is
scoped to exactly the combat-relevant fields the frontend's ``NpcCard.tsx``
"NPC views" render (see ``SR_COMBAT_SKILLS``, ``VTM_COMBAT_SKILLS``, etc.).
The ``_merge_*`` functions backfill every other field required by the full
``CharacterProps`` TS shape with the same safe defaults the frontend's own
``createBlankCharacter()`` uses, so the resulting content dict still
type-checks on the frontend.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from src.routers.schema.mission import GameType


class NpcProfile(BaseModel):
    """Shared Profiler agent output."""

    character_description: str
    value: str  # equipment budget in system-native terms, e.g. "5000 nuyen", "30 silver"


# ───────────────────────── Shadowrun ─────────────────────────


class ShadowrunNpcStats(BaseModel):
    metatype: str
    archetype: str
    attributes: dict[str, float]
    skills: dict[str, int]
    initiative_base: int
    initiative_dice: int
    damage_physical_max: int
    damage_stun_max: int
    equipment_categories: list[str] = Field(default_factory=list)


class ShadowrunNpcEquipment(BaseModel):
    armor: int = 0
    weapons: list[str] = Field(default_factory=list)
    cyberware: list[str] = Field(default_factory=list)
    gear: list[str] = Field(default_factory=list)


def _merge_shadowrun(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: ShadowrunNpcStats,
    equipment: ShadowrunNpcEquipment,
) -> dict:
    return {
        "gameType": GameType.SHADOWRUN.value,
        "id": npc_id,
        "name": name,
        "metatype": stats.metatype,
        "archetype": stats.archetype,
        "description": profile.character_description,
        "attributes": stats.attributes,
        "derived": {
            "physicalLimit": 0,
            "mentalLimit": 0,
            "socialLimit": 0,
            "composure": 0,
            "judgeIntentions": 0,
            "memory": 0,
            "liftCarry": 0,
            "initiativeBase": stats.initiative_base,
            "initiativeDice": stats.initiative_dice,
        },
        "skills": stats.skills,
        "knowledgeSkills": {},
        "qualities": {"positive": [], "negative": []},
        "contacts": {},
        "armor": equipment.armor,
        "weapons": equipment.weapons,
        "cyberware": equipment.cyberware,
        "gear": equipment.gear,
        "nuyen": 0,
        "lifestyle": "Low",
        "streetCred": 0,
        "notoriety": 0,
        "publicAwareness": 0,
        "damage": {
            "physical": {"current": 0, "max": stats.damage_physical_max},
            "stun": {"current": 0, "max": stats.damage_stun_max},
        },
    }


# ───────────────────────── Vampire: The Masquerade ─────────────────────────

_VTM_BLANK_ATTRIBUTES = {
    "Strength": 1,
    "Dexterity": 1,
    "Stamina": 1,
    "Charisma": 1,
    "Manipulation": 1,
    "Composure": 1,
    "Intelligence": 1,
    "Wits": 1,
    "Resolve": 1,
}

_VTM_BLANK_SKILLS = {
    k: 0
    for k in (
        "Athletics",
        "Brawl",
        "Craft",
        "Drive",
        "Firearms",
        "Melee",
        "Larceny",
        "Stealth",
        "Survival",
        "Animal Ken",
        "Etiquette",
        "Insight",
        "Intimidation",
        "Leadership",
        "Performance",
        "Persuasion",
        "Streetwise",
        "Subterfuge",
        "Academics",
        "Awareness",
        "Finance",
        "Investigation",
        "Medicine",
        "Occult",
        "Politics",
        "Science",
        "Technology",
    )
}


class VampireNpcStats(BaseModel):
    nature: str
    clan: str | None = None
    generation: int | None = None
    predator_type: str | None = None
    attributes: dict[str, int]
    skills: dict[str, int]
    disciplines: dict[str, dict] = Field(default_factory=dict)
    hunger: int | None = None
    humanity: int | None = None
    blood_potency: int | None = None
    health_max: int
    willpower_max: int
    equipment_categories: list[str] = Field(default_factory=list)


class VampireNpcEquipment(BaseModel):
    weapons: list[dict] = Field(default_factory=list)  # V5Weapon-shaped


def _merge_vampire(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: VampireNpcStats,
    equipment: VampireNpcEquipment,
) -> dict:
    full_attributes = {**_VTM_BLANK_ATTRIBUTES, **stats.attributes}
    full_skills = {**_VTM_BLANK_SKILLS, **stats.skills}
    return {
        "gameType": GameType.VAMPIRE_THE_MASQUERADE.value,
        "id": npc_id,
        "name": name,
        "nature": stats.nature,
        "description": profile.character_description,
        "clan": stats.clan,
        "generation": stats.generation,
        "predatorType": stats.predator_type,
        "attributes": full_attributes,
        "skills": full_skills,
        "disciplines": stats.disciplines,
        "weapons": equipment.weapons,
        "hunger": stats.hunger,
        "humanity": stats.humanity,
        "bloodPotency": stats.blood_potency,
        "health": {"current": stats.health_max, "max": stats.health_max},
        "willpower": {"current": stats.willpower_max, "max": stats.willpower_max},
    }


# ───────────────────────── Call of Cthulhu ─────────────────────────

_COC_BLANK_CHARACTERISTICS = {
    "STR": 50,
    "CON": 50,
    "SIZ": 50,
    "DEX": 50,
    "APP": 50,
    "INT": 50,
    "POW": 50,
    "EDU": 50,
}


class CthulhuNpcStats(BaseModel):
    occupation: str
    characteristics: dict[str, int]
    skills: dict[str, int]
    build: int
    damage_bonus: str
    move_rate: int
    hit_points_max: int
    sanity_max: int
    equipment_categories: list[str] = Field(default_factory=list)


class CthulhuNpcEquipment(BaseModel):
    weapons: list[dict] = Field(default_factory=list)  # CocWeapon-shaped
    gear: list[str] = Field(default_factory=list)


def _merge_cthulhu(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: CthulhuNpcStats,
    equipment: CthulhuNpcEquipment,
) -> dict:
    full_chars = {**_COC_BLANK_CHARACTERISTICS, **stats.characteristics}
    half = {k: v // 2 for k, v in full_chars.items()}
    fifth = {k: v // 5 for k, v in full_chars.items()}
    magic_points = full_chars["POW"] // 5
    return {
        "gameType": GameType.CALL_OF_CTHULHU.value,
        "id": npc_id,
        "name": name,
        "occupation": stats.occupation,
        "era": "1920s",
        "age": 30,
        "description": profile.character_description,
        "characteristics": full_chars,
        "derived": {
            "hpMax": stats.hit_points_max,
            "mpMax": magic_points,
            "sanityMax": stats.sanity_max,
            "build": stats.build,
            "damageBonus": stats.damage_bonus,
            "moveRate": stats.move_rate,
            "half": half,
            "fifth": fifth,
        },
        "skills": stats.skills,
        "hitPoints": {"current": stats.hit_points_max, "max": stats.hit_points_max},
        "sanity": {"current": stats.sanity_max, "max": stats.sanity_max},
        "magicPoints": {"current": magic_points, "max": magic_points},
        "luck": 50,
        "cthulhuMythos": 0,
        "weapons": equipment.weapons,
        "gear": equipment.gear,
    }


# ───────────────────────── Seventh Sea ─────────────────────────

_SS_BLANK_TRAITS = {"Brawn": 2, "Finesse": 2, "Resolve": 2, "Wits": 2, "Panache": 2}

_SS_BLANK_SKILLS = {
    k: 0
    for k in (
        "Aim",
        "Athletics",
        "Brawl",
        "Convince",
        "Empathy",
        "Hide",
        "Intimidate",
        "Notice",
        "Perform",
        "Ride",
        "Sailing",
        "Tempt",
        "Theft",
        "Warfare",
        "Weaponry",
    )
}


class SeventhSeaNpcStats(BaseModel):
    nation: str
    arcana_virtue: str
    arcana_hubris: str
    traits: dict[str, int]
    skills: dict[str, int]
    dueling_style: str | None = None
    wounds_max: int
    equipment_categories: list[str] = Field(default_factory=list)


class SeventhSeaNpcEquipment(BaseModel):
    weapons: list[dict] = Field(default_factory=list)  # SeventhSeaWeapon-shaped
    gear: list[str] = Field(default_factory=list)


def _merge_seventh_sea(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: SeventhSeaNpcStats,
    equipment: SeventhSeaNpcEquipment,
) -> dict:
    full_traits = {**_SS_BLANK_TRAITS, **stats.traits}
    full_skills = {**_SS_BLANK_SKILLS, **stats.skills}
    return {
        "gameType": GameType.SEVENTH_SEA.value,
        "id": npc_id,
        "name": name,
        "nation": stats.nation,
        "arcana": {"virtue": stats.arcana_virtue, "hubris": stats.arcana_hubris},
        "description": profile.character_description,
        "traits": full_traits,
        "skills": full_skills,
        "advantages": [],
        "duelingStyle": stats.dueling_style,
        "weapons": equipment.weapons,
        "gear": equipment.gear,
        "wounds": {"current": 0, "max": stats.wounds_max},
        "heroPoints": 3,
    }


# ───────────────────────── The Expanse ─────────────────────────

_EXPANSE_BLANK_ABILITIES = {
    "Accuracy": 0,
    "Communication": 0,
    "Constitution": 0,
    "Dexterity": 0,
    "Fighting": 0,
    "Intelligence": 0,
    "Perception": 0,
    "Strength": 0,
    "Willpower": 0,
}


class ExpanseNpcStats(BaseModel):
    origin: str
    background: str
    faction: str
    abilities: dict[str, int]
    focuses: list[str] = Field(default_factory=list)
    speed: int
    defense: int
    health_max: int
    equipment_categories: list[str] = Field(default_factory=list)


class ExpanseNpcEquipment(BaseModel):
    weapons: list[dict] = Field(default_factory=list)  # AgeWeapon-shaped
    armor: str | None = None
    gear: list[str] = Field(default_factory=list)


def _merge_expanse(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: ExpanseNpcStats,
    equipment: ExpanseNpcEquipment,
) -> dict:
    full_abilities = {**_EXPANSE_BLANK_ABILITIES, **stats.abilities}
    return {
        "gameType": GameType.EXPANSE.value,
        "id": npc_id,
        "name": name,
        "origin": stats.origin,
        "background": stats.background,
        "faction": stats.faction,
        "description": profile.character_description,
        "abilities": full_abilities,
        "focuses": stats.focuses,
        "speed": stats.speed,
        "defense": stats.defense,
        "health": {"current": stats.health_max, "max": stats.health_max},
        "fortune": 3,
        "weapons": equipment.weapons,
        "armor": equipment.armor,
        "gear": equipment.gear,
    }


# ───────────────────────── Slavic ─────────────────────────

_SLAVIC_BLANK_ATTRIBUTES = {"Strength": 3, "Agility": 3, "Wits": 3, "Empathy": 3}

_SLAVIC_BLANK_SKILLS = {
    k: 0
    for k in (
        "Endurance",
        "Fight",
        "Sneak",
        "Move",
        "Marksmanship",
        "Scout",
        "Lore",
        "Survival",
        "Craft",
        "Insight",
        "Manipulation",
        "Healing",
        "Performance",
    )
}


class SlavicNpcStats(BaseModel):
    kin: str
    kin_ability: str
    calling: str
    attributes: dict[str, int]
    skills: dict[str, int]
    talents: list[str] = Field(default_factory=list)
    equipment_categories: list[str] = Field(default_factory=list)


class SlavicNpcEquipment(BaseModel):
    weapons: list[str] = Field(default_factory=list)
    armor_name: str | None = None
    armor_rating: int | None = None
    gear: list[str] = Field(default_factory=list)


def _merge_slavic(
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: SlavicNpcStats,
    equipment: SlavicNpcEquipment,
) -> dict:
    full_attributes = {**_SLAVIC_BLANK_ATTRIBUTES, **stats.attributes}
    full_skills = {**_SLAVIC_BLANK_SKILLS, **stats.skills}
    armor = (
        {"name": equipment.armor_name, "rating": equipment.armor_rating or 0}
        if equipment.armor_name
        else None
    )
    return {
        "gameType": GameType.SLAVIC.value,
        "id": npc_id,
        "name": name,
        "kin": stats.kin,
        "kinAbility": stats.kin_ability,
        "calling": stats.calling,
        "description": profile.character_description,
        "attributes": full_attributes,
        "attributeDamage": dict(full_attributes),
        "skills": full_skills,
        "talents": stats.talents,
        "weapons": equipment.weapons,
        "armor": armor,
        "gear": equipment.gear,
        "relationships": [],
        "experience": 0,
    }


# ───────────────────────── Dispatch ─────────────────────────

NPC_PIPELINE_CONFIG: dict[GameType, tuple[type[BaseModel], type[BaseModel], Callable]] = {
    GameType.SHADOWRUN: (ShadowrunNpcStats, ShadowrunNpcEquipment, _merge_shadowrun),
    GameType.VAMPIRE_THE_MASQUERADE: (VampireNpcStats, VampireNpcEquipment, _merge_vampire),
    GameType.CALL_OF_CTHULHU: (CthulhuNpcStats, CthulhuNpcEquipment, _merge_cthulhu),
    GameType.SEVENTH_SEA: (SeventhSeaNpcStats, SeventhSeaNpcEquipment, _merge_seventh_sea),
    GameType.EXPANSE: (ExpanseNpcStats, ExpanseNpcEquipment, _merge_expanse),
    GameType.SLAVIC: (SlavicNpcStats, SlavicNpcEquipment, _merge_slavic),
}


def merge_npc(
    game_type: GameType,
    npc_id: int,
    name: str,
    profile: NpcProfile,
    stats: BaseModel,
    equipment: BaseModel,
) -> dict:
    """Builds a full ``CharacterProps``-shaped dict from the pipeline's outputs,
    backfilling every field the system's TS interface requires but the NPC
    pipeline doesn't generate with the same defaults ``createBlankCharacter()``
    uses on the frontend.
    """
    _, _, merge_fn = NPC_PIPELINE_CONFIG[game_type]
    return merge_fn(npc_id, name, profile, stats, equipment)
