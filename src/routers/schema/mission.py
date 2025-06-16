from __future__ import annotations
from enum import StrEnum

from pydantic import BaseModel, model_validator
from src.routers.schema.interaction import InteractionSchema


class GameType(StrEnum):
    SHADOWRUN = "shadowrun"
    VAMPIRE_THE_MASQUERADE = "vampire_the_masquerade"
    CALL_OF_CTHULHU = "call_of_cthulhu"
    SEVENTH_SEA = "seventh_sea"
    EXPANSE = "expanse"


class Mission(BaseModel):
    mission_id: int | None = None
    name_custom: str = ""
    name: str
    description: str
    game_type: GameType
    background: str
    non_hero_mode: bool

    @model_validator(mode="after")
    def check_non_hero_mode(self) -> Mission:
        if self.non_hero_mode and self.game_type != GameType.EXPANSE:
            raise ValueError("Only GameType.EXPANSE may have non_hero_mode=True")
        return self


class SaveMission(BaseModel):
    mission_id: int
    name_custom: str


class LoadMission(BaseModel):
    mission: Mission
    interactions: list[InteractionSchema]


class NewMissionPayload(BaseModel):
    game_type: GameType
    background: str
    non_hero_mode: bool = False

    @model_validator(mode="after")
    def check_non_hero_mode(self) -> NewMissionPayload:
        if self.non_hero_mode and self.game_type != GameType.EXPANSE:
            raise ValueError("Only GameType.EXPANSE may have non_hero_mode=True")
        return self
