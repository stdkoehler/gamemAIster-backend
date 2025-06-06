from enum import StrEnum

from pydantic import BaseModel
from src.routers.schema.interaction import InteractionSchema


class GameType(StrEnum):
    SHADOWRUN = "shadowrun"
    VAMPIRE_THE_MASQUERADE = "vampire_the_masquerade"
    CALL_OF_CTHULHU = "call_of_cthulhu"
    SEVENTH_SEA = "seventh_sea"


class Mission(BaseModel):
    mission_id: int | None = None
    name_custom: str = ""
    name: str
    description: str
    game_type: GameType
    background: str


class SaveMission(BaseModel):
    mission_id: int
    name_custom: str


class LoadMission(BaseModel):
    mission: Mission
    interactions: list[InteractionSchema]


class NewMissionPayload(BaseModel):
    game_type: GameType
    background: str
