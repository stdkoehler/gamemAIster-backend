from pydantic import BaseModel


class Mission(BaseModel):
    mission_id: int
    name: str
