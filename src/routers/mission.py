"""endpoints calling session"""

from fastapi import APIRouter

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.utils.logger import configure_logger

log = configure_logger("mission")

router = APIRouter(
    prefix="/mission",
    tags=["mission"],
    responses={404: {"description": "Not found"}},
)


@router.post("/new-mission")
def new_mission():
    gamemaster = Gamemaster("http://127.0.0.1:5000")
    mission_description = gamemaster.generate_mission()
    crud_instance.new_mission(
        name=mission_description["title"], description=mission_description["task"]
    )


@router.get("/sessions")
async def sessions():

    log.info("stream_data")
