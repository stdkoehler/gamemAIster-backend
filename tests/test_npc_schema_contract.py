"""
Validates Gamemaster.generate_npc()'s merge step against the frontend's
*Character TS interfaces.

The frontend's CharacterProps interfaces and the hand-crafted NpcCard
components that render them are the design source of truth for the NPC
sheet shape — not the backend. This test guards the dependency direction:
each ``_merge_*`` function in npc_models.py must keep producing a dict that
satisfies the corresponding interface, or NpcCard.tsx will silently fail to
render fields (or TypeScript would reject the payload, if it were typed).

The JSON Schemas under tests/schemas/ are generated FROM those TS
interfaces (see gamemAIster-frontend/scripts/generate-npc-schemas.mjs,
`npm run gen:npc-schemas`) and must be regenerated whenever CharacterProps.tsx
changes.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from src.brain.npc_models import (
    CthulhuNpcEquipment,
    CthulhuNpcStats,
    ExpanseNpcEquipment,
    ExpanseNpcStats,
    NpcProfile,
    SeventhSeaNpcEquipment,
    SeventhSeaNpcStats,
    ShadowrunNpcEquipment,
    ShadowrunNpcStats,
    SlavicNpcEquipment,
    SlavicNpcStats,
    VampireNpcEquipment,
    VampireNpcStats,
    merge_npc,
)
from src.routers.schema.mission import GameType

_SCHEMA_DIR = Path(__file__).parent / "schemas"


def _load_schema(game_type: GameType) -> dict:
    with open(_SCHEMA_DIR / f"{game_type.value}.schema.json", encoding="utf-8") as f:
        return json.load(f)


_PROFILE = NpcProfile(character_description="A test NPC.", value="5000 nuyen")

_CASES = [
    (
        GameType.SHADOWRUN,
        ShadowrunNpcStats(
            metatype="Human",
            archetype="Street Samurai",
            attributes={"Body": 5, "Agility": 6, "Reaction": 5, "Strength": 4,
                        "Willpower": 3, "Logic": 3, "Intuition": 4, "Charisma": 2,
                        "Edge": 3, "Essence": 4.2},
            skills={"Automatics": 8, "Pistols": 6},
            initiative_base=9,
            initiative_dice=2,
            damage_physical_max=11,
            damage_stun_max=10,
            equipment_categories=["Pistols", "Armor"],
        ),
        ShadowrunNpcEquipment(
            armor=9, weapons=["Ares Predator VI"], cyberware=["Wired Reflexes 1"], gear=["Medkit"]
        ),
    ),
    (
        GameType.VAMPIRE_THE_MASQUERADE,
        VampireNpcStats(
            nature="kindred",
            clan="Brujah",
            generation=10,
            predator_type="Alleycat",
            attributes={"Strength": 3, "Dexterity": 3, "Stamina": 2},
            skills={"Brawl": 3, "Intimidation": 2},
            disciplines={"Potence": {"level": 2, "powers": ["Soaring Leap"]}},
            hunger=2,
            humanity=6,
            blood_potency=2,
            health_max=5,
            willpower_max=5,
            equipment_categories=["Melee"],
        ),
        VampireNpcEquipment(
            weapons=[{"name": "Switchblade", "damage": 1, "skill": "Melee", "range": 0, "properties": ["Concealable"]}]
        ),
    ),
    (
        GameType.CALL_OF_CTHULHU,
        CthulhuNpcStats(
            occupation="Detective",
            characteristics={"STR": 55, "CON": 60, "SIZ": 60, "DEX": 55, "APP": 50,
                              "INT": 65, "POW": 55, "EDU": 70},
            skills={"Fighting (Brawl)": 40, "Firearms (Handgun)": 45},
            build=0,
            damage_bonus="0",
            move_rate=8,
            hit_points_max=12,
            sanity_max=99,
            equipment_categories=["weapons"],
        ),
        CthulhuNpcEquipment(
            weapons=[{"name": ".38 Revolver", "skill": "Firearms (Handgun)", "damage": "1d10",
                      "range": "15 yards", "attacksPerRound": 1, "ammo": 6, "malfunction": 100}],
            gear=["Notebook"],
        ),
    ),
    (
        GameType.SEVENTH_SEA,
        SeventhSeaNpcStats(
            nation="Castille",
            arcana_virtue="Loyal",
            arcana_hubris="Stubborn",
            traits={"Brawn": 2, "Finesse": 3, "Resolve": 2, "Wits": 2, "Panache": 3},
            skills={"Weaponry": 3, "Athletics": 2},
            dueling_style="Valroux (Feint & Riposte)",
            wounds_max=2,
            equipment_categories=["Melee"],
        ),
        SeventhSeaNpcEquipment(
            weapons=[{"name": "Rapier", "trait": "Finesse", "type": "fencing", "properties": ["Dueling"]}],
            gear=["Fine Clothes"],
        ),
    ),
    (
        GameType.EXPANSE,
        ExpanseNpcStats(
            origin="Belter",
            background="Criminal",
            faction="OPA",
            abilities={"Accuracy": 2, "Communication": 0, "Constitution": 1, "Dexterity": 2,
                       "Fighting": 1, "Intelligence": 0, "Perception": 1, "Strength": 0, "Willpower": 1},
            focuses=["Accuracy (Pistols)"],
            speed=12,
            defense=12,
            health_max=35,
            equipment_categories=["weapons"],
        ),
        ExpanseNpcEquipment(
            weapons=[{"name": "PCA PDW", "damage": "2d6+2", "range": "Short/Long 10/30m", "qualities": ["Burst Fire"]}],
            armor="MCRN Infantry Battle Dress",
            gear=["Medpatch"],
        ),
    ),
    (
        GameType.SLAVIC,
        SlavicNpcStats(
            kin="Human",
            kin_ability="Stubborn: reroll one failed Endurance roll per day.",
            calling="Warrior",
            attributes={"Strength": 4, "Agility": 3, "Wits": 2, "Empathy": 2},
            skills={"Fight": 3, "Endurance": 2},
            talents=["Berserker"],
            equipment_categories=["Melee", "Armor"],
        ),
        SlavicNpcEquipment(
            weapons=["Hand Axe"], armor_name="Leather Jerkin", armor_rating=2, gear=["Waterskin"]
        ),
    ),
]


@pytest.mark.parametrize("game_type, stats, equipment", _CASES, ids=lambda v: v.value if isinstance(v, GameType) else "")
def test_merge_npc_matches_frontend_schema(game_type, stats, equipment):
    content = merge_npc(game_type, npc_id=1, name="Test NPC", profile=_PROFILE, stats=stats, equipment=equipment)
    schema = _load_schema(game_type)
    jsonschema.validate(instance=content, schema=schema)
