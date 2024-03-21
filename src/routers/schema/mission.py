from pydantic import BaseModel


class Mission(BaseModel):
    mission_id: int | None = None
    name_custom: str = ""
    name: str
    description: str


class SaveMission(BaseModel):
    mission_id: int
    name_custom: str
