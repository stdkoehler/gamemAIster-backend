from pydantic import BaseModel

from src.routers.schema.interaction import InteractionSchema


class Mission(BaseModel):
    mission_id: int | None = None
    name_custom: str = ""
    name: str
    description: str


class SaveMission(BaseModel):
    mission_id: int
    name_custom: str


class LoadMission(BaseModel):
    mission: Mission
    interactions: list[InteractionSchema]
