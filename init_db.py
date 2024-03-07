""" Init Database """

import sqlalchemy
from src.crud.sqlmodel import Base

engine = sqlalchemy.create_engine("sqlite:///memory.db")
Base.metadata.create_all(engine)
