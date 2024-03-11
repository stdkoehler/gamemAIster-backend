"""endpoints calling session"""

from fastapi import APIRouter

from src.crud.crud import crud_instance
from src.brain.gamemaster import Gamemaster

from src.utils.logger import configure_logger

log = configure_logger("session")

router = APIRouter(
    prefix="/session",
    tags=["session"],
    responses={404: {"description": "Not found"}},
)


@router.post("/new-session")
def new_session():
    gamemaster = Gamemaster("http://127.0.0.1:5000")
    session_description = gamemaster.generate_session()
    crud_instance.new_session(
        name=session_description["title"], description=session_description["task"]
    )


@router.get("/sessions")
async def sessions():

    log.info("stream_data")
