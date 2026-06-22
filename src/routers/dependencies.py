"""Dependencies for interaction router."""

from fastapi import Depends

from src.auth.auth import verify_user
from src.crud.crud import crud_instance

import src.routers.schema.interaction as api_schema_interaction
from src.routers.schema.mission import NewMissionPayload, CreateNpcPayload

from src.brain.gamemaster import build_gamemaster, MissionOptions, Gamemaster


def get_gamemaster_for_interaction(
    prompt: api_schema_interaction.InteractionPrompt,
    user: str = Depends(verify_user),
) -> Gamemaster:
    """Build a Gamemaster for interaction (existing mission)."""
    game_type = crud_instance.get_mission_game_type(prompt.mission_id)
    non_hero_mode = crud_instance.get_mission_non_hero_mode(prompt.mission_id)
    oracle = crud_instance.get_mission_oracle(prompt.mission_id)
    mission_options = MissionOptions(non_hero_mode=non_hero_mode, oracle=oracle)
    return build_gamemaster(user, game_type, mission_options)


def get_gamemaster_for_npc(
    payload: CreateNpcPayload,
    user: str = Depends(verify_user),
) -> Gamemaster:
    """Build a Gamemaster for NPC generation (existing mission)."""
    game_type = crud_instance.get_mission_game_type(payload.mission_id)
    non_hero_mode = crud_instance.get_mission_non_hero_mode(payload.mission_id)
    oracle = crud_instance.get_mission_oracle(payload.mission_id)
    mission_options = MissionOptions(non_hero_mode=non_hero_mode, oracle=oracle)
    return build_gamemaster(user, game_type, mission_options)


def get_gamemaster_for_mission(
    payload: NewMissionPayload,
    user: str = Depends(verify_user),
) -> Gamemaster:
    """Build a Gamemaster for mission creation (new mission)."""
    mission_options = MissionOptions(
        non_hero_mode=payload.non_hero_mode,
        oracle=payload.oracle,
    )
    return build_gamemaster(user, payload.game_type, mission_options)
