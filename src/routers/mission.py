"""endpoints calling session"""

from fastapi import APIRouter

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.utils.logger import configure_logger

import src.routers.schema.mission as api_schema_mission

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
    crud_instance.insert_mission(mission=mission)
    return mission


@router.post("/save-mission")
def save_mission(mission_id: int):
    crud_instance.save_mission(mission_id=mission_id)


@router.get("/missions")
async def missions() -> list[api_schema_mission.Mission]:
    return crud_instance.list_missions()
