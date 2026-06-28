"""endpoints calling session"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.auth.auth import verify_user

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.routers.dependencies import get_gamemaster_for_mission, get_gamemaster_for_npc
from src.utils.logger import configure_logger

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction
from src.routers.schema.mission import (
    NewMissionPayload,
    UpsertCharacterSheet,
    DeleteCharacterSheet,
    CreateNpcPayload,
    SetNpcActivePayload,
)

log = configure_logger("mission")

router = APIRouter(
    prefix="/mission",
    tags=["mission"],
    dependencies=[Depends(verify_user)],
    responses={404: {"description": "Not found"}},
)


@router.post("/new-mission")
def new_mission(
    payload: NewMissionPayload,
    gamemaster: Gamemaster = Depends(get_gamemaster_for_mission),
) -> api_schema_mission.Mission:
    """
    Generate a new mission via LLM call.
    """
    log.info("new-mission | game_type=%s | non_hero_mode=%s | oracle=%s", payload.game_type, payload.non_hero_mode, payload.oracle)

    mission = gamemaster.generate_mission(
        background=payload.background, detailed_background=payload.detailed_background
    )
    mission = crud_instance.insert_mission(mission=mission)

    return mission


@router.post("/save-mission")
def save_mission(
    mission: api_schema_mission.SaveMission,
    user: str = Depends(verify_user),
) -> None:
    """
    Save mission to database.
    """
    try:
        crud_instance.verify_mission_user(mission_id=mission.mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    crud_instance.save_mission(mission)


@router.get("/missions")
async def missions(
    user: str = Depends(verify_user),
) -> list[api_schema_mission.Mission]:
    """
    List missons in database
    """
    return crud_instance.list_missions(user_id=user)


@router.get("/mission/{mission_id}")
async def get_mission(
    mission_id: int,
    user: str = Depends(verify_user),
) -> api_schema_mission.Mission | None:
    """
    Get the mission description for a given mission id
    """
    try:
        crud_instance.verify_mission_user(mission_id=mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    return crud_instance.get_mission_description(mission_id=mission_id)


@router.get("/load-mission/{mission_id}")
async def load_mission(
    mission_id: int,
    user: str = Depends(verify_user),
) -> api_schema_mission.LoadMission | None:
    """
    Load a mission from database
    """
    try:
        crud_instance.verify_mission_user(mission_id=mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    mission = crud_instance.get_mission_description(mission_id=mission_id)

    if mission is not None:
        interactions = crud_instance.get_interactions(mission_id=mission_id)
        character_sheets = crud_instance.get_character_sheets(mission_id=mission_id)
        return api_schema_mission.LoadMission(
            mission=mission,
            interactions=[
                api_schema_interaction.InteractionSchema(
                    user_input=interaction.user_input, llm_output=interaction.llm_output
                )
                for interaction in interactions
            ],
            character_sheets=character_sheets,
        )

    return None


@router.get("/character-sheets/{mission_id}")
async def get_character_sheets(
    mission_id: int,
    user: str = Depends(verify_user),
) -> list[api_schema_mission.CharacterSheetSchema]:
    """Fetch all character sheets for a mission."""
    try:
        crud_instance.verify_mission_user(mission_id=mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    return crud_instance.get_character_sheets(mission_id=mission_id)


@router.post("/upsert-character-sheet")
def upsert_character_sheet(
    sheet: UpsertCharacterSheet,
    user: str = Depends(verify_user),
) -> api_schema_mission.CharacterSheetSchema:
    """Create or update a character sheet for a mission."""
    try:
        crud_instance.verify_mission_user(mission_id=sheet.mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    try:
        return crud_instance.upsert_character_sheet(sheet=sheet)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/delete-character-sheet")
def delete_character_sheet(
    payload: DeleteCharacterSheet,
    user: str = Depends(verify_user),
) -> None:
    """Delete a character sheet."""
    try:
        crud_instance.verify_mission_user(mission_id=payload.mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    crud_instance.delete_character_sheet(
        character_sheet_id=payload.character_sheet_id,
        mission_id=payload.mission_id,
    )


@router.post("/set-npc-active")
def set_npc_active(
    payload: SetNpcActivePayload,
    user: str = Depends(verify_user),
) -> None:
    """Mark an NPC as active/inactive in the current scene, without rewriting its content."""
    try:
        crud_instance.verify_mission_user(mission_id=payload.mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc
    try:
        crud_instance.set_character_sheet_active(
            character_sheet_id=payload.character_sheet_id,
            mission_id=payload.mission_id,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/create-npc")
def create_npc(
    payload: CreateNpcPayload,
    gamemaster: Gamemaster = Depends(get_gamemaster_for_npc),
    user: str = Depends(verify_user),
) -> api_schema_mission.CharacterSheetSchema:
    """Generate an NPC via the 3-agent pipeline and persist it as a CharacterSheet."""
    try:
        crud_instance.verify_mission_user(mission_id=payload.mission_id, user_id=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
        ) from exc

    log.info("create-npc | mission_id=%d | name=%s", payload.mission_id, payload.name)

    try:
        result = gamemaster.generate_npc(name=payload.name, mission_id=payload.mission_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    sheet = UpsertCharacterSheet(
        character_sheet_id=None,
        mission_id=payload.mission_id,
        name=payload.name,
        game_type=result.content["gameType"],
        content=result.content,
        is_protagonist=False,
        is_npc=True,
        matched_key_npc=result.matched_key_npc,
    )
    return crud_instance.upsert_character_sheet(sheet=sheet)
