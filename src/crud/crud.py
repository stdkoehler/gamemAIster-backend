import sqlalchemy
from sqlalchemy import event, select, delete, and_
from sqlalchemy.orm import sessionmaker

from src.crud.sqlmodel import Mission, MissionDescription, ConversationMemory
from src.brain.types import Interaction

import src.routers.schema.mission as schema_mission


class CRUD:
    def __init__(self, dbase: str):
        self._engine = sqlalchemy.create_engine(dbase)

        def _fk_pragma_on_connect(dbapi_con, _):
            dbapi_con.execute("pragma foreign_keys=ON")

        event.listen(self._engine, "connect", _fk_pragma_on_connect)

        self._sessionmaker = sessionmaker(self._engine)
        self._cleanse_unpersisted()

    def _cleanse_unpersisted(self):
        with self._sessionmaker() as session:
            stmt = delete(Mission).where(Mission.persist.is_(False))
            session.execute(stmt)
            session.commit()

    def get_mission_id(self, mission_name: str) -> int:
        with self._sessionmaker() as session:
            stmt = select(Mission.mission_id).where(Mission.name == mission_name)
            result = session.execute(stmt).scalar()
            if result is None:
                raise ValueError(f"No mission with name {mission_name}")
            return result

    def new_mission(self, name, description):
        with self._sessionmaker() as session:
            new_mission = Mission(name=name, persist=False)
            session.add(new_mission)
            session.flush()
            mission_description = MissionDescription(
                mission_id=new_mission.mission_id, description=description
            )
            session.add(mission_description)
            session.commit()

    def list_missions(self) -> list[schema_mission.Mission]:

        with self._sessionmaker() as session:
            stmt = (
                select(Mission)
                .where(Mission.persist.is_(True))
                .order_by(Mission.mission_id)
            )
            results = session.execute(stmt).scalars().all()

            return [
                schema_mission.Mission(mission_id=result.mission_id, name=result.name)
                for result in results
            ]

    def insert_mission(self, mission_name: str) -> int:
        with self._sessionmaker() as session:
            new_mission = Mission(name=mission_name, persist=False)
            session.add(new_mission)
            session.commit()
            return new_mission.mission_id

    def get_interactions(self, mission_id: int) -> list[Interaction]:
        with self._sessionmaker() as session:
            stmt = (
                select(ConversationMemory)
                .join(
                    Mission,
                    and_(
                        Mission.mission_id == ConversationMemory.mission_id,
                        Mission.mission_id == mission_id,
                    ),
                )
                .order_by(ConversationMemory.conversation_memory_id.asc())
            )
            result = session.execute(stmt).scalars().all()
            return [
                Interaction(
                    id_=memory.conversation_memory_id,
                    user_input=memory.user_input,
                    llm_output=memory.llm_output,
                )
                for memory in result
            ]

    def insert_interaction(self, mission_id: int, interaction: Interaction):
        memory = ConversationMemory(
            mission_id=mission_id,
            user_input=interaction.user_input,
            llm_output=interaction.llm_output,
        )
        with self._sessionmaker() as session:
            session.add(memory)
            session.commit()


crud_instance = CRUD(dbase="sqlite:///memory.db")
