"""Single source of truth for per-system wiring.

Adding a new TTRPG system means adding one entry to ``GAME_CONFIGS`` (and,
if NPC generation is supported, one entry to ``NPC_CONFIGS``) — no other
dispatch lives elsewhere. The per-system *logic* these entries point at
(Oracle subclasses, NPC Stats/Equipment models, merge functions) still lives
in oracle.py / npc_models.py, where it's defined; this module only wires it
together. Irregular, non-uniform features that only apply to a subset of
systems (non-hero mode, reasoning warmstarts, equipment cost-field
overrides) intentionally stay as their own small lookups near the code that
uses them, rather than being forced into this registry as mostly-empty
fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

import src.routers.schema.mission as api_schema_mission
from src.brain.npc_models import (
    NpcProfile,
    ShadowrunNpcStats,
    ShadowrunNpcEquipment,
    _merge_shadowrun,
    VampireNpcStats,
    VampireNpcEquipment,
    _merge_vampire,
    CthulhuNpcStats,
    CthulhuNpcEquipment,
    _merge_cthulhu,
    SeventhSeaNpcStats,
    SeventhSeaNpcEquipment,
    _merge_seventh_sea,
    ExpanseNpcStats,
    ExpanseNpcEquipment,
    _merge_expanse,
    SlavicNpcStats,
    SlavicNpcEquipment,
    _merge_slavic,
    DragonlanceNpcStats,
    DragonlanceNpcEquipment,
    _merge_dragonlance,
)
from src.brain.oracle import (
    BaseOracle,
    CustomOracle,
    DragonlanceOracle,
    ExpanseNonHeroOracle,
    ExpanseOracle,
    SeventhSeaOracle,
    ShadowrunOracle,
    SlavicOracle,
    VampireOracle,
    CthulhuOracle,
)

_GT = api_schema_mission.GameType


@dataclass(frozen=True)
class GameConfig:
    game_name: str
    story_prompt: str               # relative to prompt_templates/ — GM persona governing the live narrative chat
    mission_prompt: str             # relative to prompt_templates/
    mission_prompt_non_oracle: str
    oracle_class: type[BaseOracle]


GAME_CONFIGS: dict[tuple[api_schema_mission.GameType, bool], GameConfig] = {
    (_GT.SHADOWRUN, False): GameConfig(
        game_name="Shadowrun 6th Edition",
        story_prompt="shadowrun/shadowrun_story_prompt.txt",
        mission_prompt="shadowrun/shadowrun_mission_prompt.txt",
        mission_prompt_non_oracle="shadowrun/shadowrun_mission_prompt.txt",
        oracle_class=ShadowrunOracle,
    ),
    (_GT.VAMPIRE_THE_MASQUERADE, False): GameConfig(
        game_name="Vampire the Masquerade 5th Edition",
        story_prompt="vampire/vampire_story_prompt.txt",
        mission_prompt="vampire/vampire_mission_prompt.txt",
        mission_prompt_non_oracle="vampire/vampire_non_oracle_mission_prompt.txt",
        oracle_class=VampireOracle,
    ),
    (_GT.CALL_OF_CTHULHU, False): GameConfig(
        game_name="Call of Cthulhu 7th Edition",
        story_prompt="cthulhu/cthulhu_story_prompt.txt",
        mission_prompt="cthulhu/cthulhu_mission_prompt.txt",
        mission_prompt_non_oracle="cthulhu/cthulhu_non_oracle_mission_prompt.txt",
        oracle_class=CthulhuOracle,
    ),
    (_GT.SEVENTH_SEA, False): GameConfig(
        game_name="Seventh Sea 2nd Edition",
        story_prompt="seventh_sea/seventh_sea_story_prompt.txt",
        mission_prompt="seventh_sea/seventh_sea_mission_prompt.txt",
        mission_prompt_non_oracle="seventh_sea/seventh_sea_non_oracle_mission_prompt.txt",
        oracle_class=SeventhSeaOracle,
    ),
    (_GT.EXPANSE, False): GameConfig(
        game_name="The Expanse RPG",
        story_prompt="expanse/expanse_story_prompt.txt",
        mission_prompt="expanse/expanse_mission_prompt.txt",
        mission_prompt_non_oracle="expanse/expanse_mission_prompt.txt",
        oracle_class=ExpanseOracle,
    ),
    (_GT.EXPANSE, True): GameConfig(
        game_name="The Expanse RPG",
        story_prompt="expanse/expanse_story_prompt_non_hero.txt",
        mission_prompt="expanse/expanse_mission_prompt_non_hero.txt",
        mission_prompt_non_oracle="expanse/expanse_mission_prompt_non_hero.txt",
        oracle_class=ExpanseNonHeroOracle,
    ),
    (_GT.SLAVIC, False): GameConfig(
        game_name="Baltic Slavic 800 A.D.",
        story_prompt="slavic/slavic_story_prompt.txt",
        mission_prompt="slavic/slavic_mission_prompt.txt",
        mission_prompt_non_oracle="slavic/slavic_mission_prompt.txt",
        oracle_class=SlavicOracle,
    ),
    (_GT.DRAGONLANCE, False): GameConfig(
        game_name="Dragonlance: Shadow of the Dragon Queen (D&D 5E)",
        story_prompt="dragonlance/dragonlance_story_prompt.txt",
        mission_prompt="dragonlance/dragonlance_mission_prompt.txt",
        mission_prompt_non_oracle="dragonlance/dragonlance_non_oracle_mission_prompt.txt",
        oracle_class=DragonlanceOracle,
    ),
    (_GT.CUSTOM, False): GameConfig(
        game_name="Custom RPG",
        story_prompt="custom/custom_story_prompt.txt",
        mission_prompt="custom/custom_mission_prompt.txt",
        mission_prompt_non_oracle="custom/custom_mission_prompt.txt",
        oracle_class=CustomOracle,
    ),
    (_GT.CUSTOM, True): GameConfig(
        game_name="Custom RPG",
        story_prompt="custom/custom_story_prompt.txt",
        mission_prompt="custom/custom_mission_prompt.txt",
        mission_prompt_non_oracle="custom/custom_mission_prompt.txt",
        oracle_class=CustomOracle,
    ),
}


@dataclass(frozen=True)
class NpcConfig:
    profile_prompt: str             # relative to prompt_templates/ — system-specific NPC tiers/budgets
    stats_prompt: str               # relative to prompt_templates/
    equipment_prompt: str           # relative to prompt_templates/
    stats_model: type[BaseModel]
    equipment_model: type[BaseModel]
    merge_fn: Callable[..., dict]


NPC_CONFIGS: dict[api_schema_mission.GameType, NpcConfig] = {
    _GT.SHADOWRUN: NpcConfig(
        profile_prompt="shadowrun/shadowrun_npc_profile_prompt.txt",
        stats_prompt="shadowrun/shadowrun_npc_stats_prompt.txt",
        equipment_prompt="shadowrun/shadowrun_npc_equipment_prompt.txt",
        stats_model=ShadowrunNpcStats,
        equipment_model=ShadowrunNpcEquipment,
        merge_fn=_merge_shadowrun,
    ),
    _GT.VAMPIRE_THE_MASQUERADE: NpcConfig(
        profile_prompt="vampire/vampire_npc_profile_prompt.txt",
        stats_prompt="vampire/vampire_npc_stats_prompt.txt",
        equipment_prompt="vampire/vampire_npc_equipment_prompt.txt",
        stats_model=VampireNpcStats,
        equipment_model=VampireNpcEquipment,
        merge_fn=_merge_vampire,
    ),
    _GT.CALL_OF_CTHULHU: NpcConfig(
        profile_prompt="cthulhu/cthulhu_npc_profile_prompt.txt",
        stats_prompt="cthulhu/cthulhu_npc_stats_prompt.txt",
        equipment_prompt="cthulhu/cthulhu_npc_equipment_prompt.txt",
        stats_model=CthulhuNpcStats,
        equipment_model=CthulhuNpcEquipment,
        merge_fn=_merge_cthulhu,
    ),
    _GT.SEVENTH_SEA: NpcConfig(
        profile_prompt="seventh_sea/seventh_sea_npc_profile_prompt.txt",
        stats_prompt="seventh_sea/seventh_sea_npc_stats_prompt.txt",
        equipment_prompt="seventh_sea/seventh_sea_npc_equipment_prompt.txt",
        stats_model=SeventhSeaNpcStats,
        equipment_model=SeventhSeaNpcEquipment,
        merge_fn=_merge_seventh_sea,
    ),
    _GT.EXPANSE: NpcConfig(
        profile_prompt="expanse/expanse_npc_profile_prompt.txt",
        stats_prompt="expanse/expanse_npc_stats_prompt.txt",
        equipment_prompt="expanse/expanse_npc_equipment_prompt.txt",
        stats_model=ExpanseNpcStats,
        equipment_model=ExpanseNpcEquipment,
        merge_fn=_merge_expanse,
    ),
    _GT.SLAVIC: NpcConfig(
        profile_prompt="slavic/slavic_npc_profile_prompt.txt",
        stats_prompt="slavic/slavic_npc_stats_prompt.txt",
        equipment_prompt="slavic/slavic_npc_equipment_prompt.txt",
        stats_model=SlavicNpcStats,
        equipment_model=SlavicNpcEquipment,
        merge_fn=_merge_slavic,
    ),
    _GT.DRAGONLANCE: NpcConfig(
        profile_prompt="dragonlance/dragonlance_npc_profile_prompt.txt",
        stats_prompt="dragonlance/dragonlance_npc_stats_prompt.txt",
        equipment_prompt="dragonlance/dragonlance_npc_equipment_prompt.txt",
        stats_model=DragonlanceNpcStats,
        equipment_model=DragonlanceNpcEquipment,
        merge_fn=_merge_dragonlance,
    ),
}


def merge_npc(
    game_type: api_schema_mission.GameType,
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
    return NPC_CONFIGS[game_type].merge_fn(npc_id, name, profile, stats, equipment)
