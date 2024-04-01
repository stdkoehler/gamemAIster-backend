"""endpoints calling session"""

from fastapi import APIRouter

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.utils.logger import configure_logger

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction

log = configure_logger("mission")

router = APIRouter(
    prefix="/mission",
    tags=["mission"],
    responses={404: {"description": "Not found"}},
)


@router.post("/new-mission")
def new_mission() -> api_schema_mission.Mission:
    gamemaster = Gamemaster("http://127.0.0.1:5000")
    mission = gamemaster.generate_mission()
    mission = crud_instance.insert_mission(mission=mission)

    if mission.mission_id is not None:
        crud_instance.get_mission_description(mission_id=mission.mission_id)

    return mission


@router.post("/save-mission")
def save_mission(mission: api_schema_mission.SaveMission):
    crud_instance.save_mission(mission)


@router.get("/missions")
async def missions() -> list[api_schema_mission.Mission]:
    return crud_instance.list_missions()


@router.get("/mission/{mission_id}")
async def get_mission(mission_id: int) -> api_schema_mission.Mission | None:
    return crud_instance.get_mission_description(mission_id=mission_id)


@router.get("/load-mission/{mission_id}")
async def load_mission(mission_id: int) -> api_schema_mission.LoadMission | None:
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
