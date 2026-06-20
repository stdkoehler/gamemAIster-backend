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
    SLAVIC = "slavic"
    CUSTOM = "custom"


class Mission(BaseModel):
    mission_id: int | None = None
    user_id: str
    name_custom: str = ""
    name: str
    description: str
    game_type: GameType
    background: str
    detailed_background: str
    non_hero_mode: bool
    oracle: bool

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
    character_sheets: list[CharacterSheetSchema] = []


class CharacterSheetSchema(BaseModel):
    character_sheet_id: int
    mission_id: int
    name: str
    game_type: str
    content: dict
    is_protagonist: bool = False
    is_npc: bool = False


class UpsertCharacterSheet(BaseModel):
    character_sheet_id: int | None = None
    mission_id: int
    name: str
    game_type: str
    content: dict
    is_protagonist: bool = False
    is_npc: bool = False


class DeleteCharacterSheet(BaseModel):
    character_sheet_id: int
    mission_id: int


class CreateNpcPayload(BaseModel):
    mission_id: int
    name: str


class NewMissionPayload(BaseModel):
    game_type: GameType
    background: str
    detailed_background: str
    non_hero_mode: bool = False
    oracle: bool = True

    @model_validator(mode="after")
    def check_non_hero_mode(self) -> NewMissionPayload:
        if self.non_hero_mode and self.game_type != GameType.EXPANSE:
            raise ValueError("Only GameType.EXPANSE may have non_hero_mode=True")
        return self
