"""endpoints calling session"""

import os

from fastapi import APIRouter

from src.llmclient.llm_client import (
    LLMClientClaude,
    LLMClientGemini,
    LLMClientLocal,
    LLMClientDeepSeek,
)

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.utils.logger import configure_logger

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction
from src.routers.schema.mission import NewMissionPayload

log = configure_logger("mission")

router = APIRouter(
    prefix="/mission",
    tags=["mission"],
    responses={404: {"description": "Not found"}},
)


@router.post("/new-mission")
def new_mission(payload: NewMissionPayload) -> api_schema_mission.Mission:
    """
    Generate a new mission via LLM call.
    """
    print(payload.game_type)
    print("non_hero_mode:", payload.non_hero_mode)

    llm_type = os.getenv("LLM")
    if llm_type == "LOCAL":
        llm_client_local = LLMClientLocal(base_url="http://127.0.0.1:5000")
        gamemaster = Gamemaster(
            llm_client_chat=llm_client_local,
            llm_client_reasoning=llm_client_local,
            game_type=payload.game_type,
            non_hero_mode=payload.non_hero_mode,
        )
    elif llm_type == "DEEPSEEK":
        api_key = os.getenv("API_KEY_DEEPSEEK")
        if api_key is None:
            raise ValueError("OpenRouter API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientDeepSeek(api_key=api_key, model="deepseek-chat"),
            llm_client_reasoning=LLMClientDeepSeek(
                api_key=api_key, model="deepseek-reasoner"
            ),
            game_type=payload.game_type,
            non_hero_mode=payload.non_hero_mode,
        )
    elif llm_type == "GEMINI":
        api_key = os.getenv("API_KEY_GEMINI")
        if api_key is None:
            raise ValueError("Gemini API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientGemini(
                api_key=api_key,
                model="gemini-2.5-pro-exp-03-25",  # "gemini-2.5-flash-preview-04-17"
            ),
            llm_client_reasoning=LLMClientGemini(
                api_key=api_key,
                model="gemini-2.5-pro-exp-03-25",  # "gemini-2.5-flash-preview-04-17"
            ),
            game_type=payload.game_type,
            non_hero_mode=payload.non_hero_mode,
        )
    elif llm_type == "CLAUDE":
        api_key = os.getenv("API_KEY_CLAUDE")
        if api_key is None:
            raise ValueError("Claude API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientClaude(
                api_key=api_key,
                model="claude-3-7-sonnet-latest",
            ),
            llm_client_reasoning=LLMClientClaude(
                api_key=api_key,
                model="claude-3-7-sonnet-latest",
            ),
            game_type=payload.game_type,
            non_hero_mode=payload.non_hero_mode,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    mission = gamemaster.generate_mission(background=payload.background)
    mission = crud_instance.insert_mission(mission=mission)

    if mission.mission_id is not None:
        crud_instance.get_mission_description(mission_id=mission.mission_id)

    return mission


@router.post("/save-mission")
def save_mission(mission: api_schema_mission.SaveMission) -> None:
    """
    Save mission to database.
    """
    crud_instance.save_mission(mission)


@router.get("/missions")
async def missions() -> list[api_schema_mission.Mission]:
    """
    List missons in database
    """
    return crud_instance.list_missions()


@router.get("/mission/{mission_id}")
async def get_mission(mission_id: int) -> api_schema_mission.Mission | None:
    """
    Get the mission description for a given mission id
    """
    return crud_instance.get_mission_description(mission_id=mission_id)


@router.get("/load-mission/{mission_id}")
async def load_mission(mission_id: int) -> api_schema_mission.LoadMission | None:
    """
    Load a mission from database
    """
    mission = crud_instance.get_mission_description(mission_id=mission_id)

    if mission is not None:
        interactions = crud_instance.get_interactions(mission_id=mission_id)
        return api_schema_mission.LoadMission(
            mission=mission,
            interactions=[
                api_schema_interaction.InteractionSchema(
                    user_input=interaction.user_input, llm_output=interaction.llm_output
                )
                for interaction in interactions
            ],
        )

    return None
