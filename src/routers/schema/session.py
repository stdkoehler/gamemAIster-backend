from pydantic import BaseModel


class Session(BaseModel):
    session_id: int
    name: str
