from pydantic import BaseModel


class Mission(BaseModel):
    mission_id: int | None = None
    name: str
    description: str
