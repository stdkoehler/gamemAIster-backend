"""CRUD operations for the mission database."""

from typing import Any

import sqlalchemy
from sqlalchemy import event, select, update, delete, and_
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.crud.sqlmodel import (
    Mission,
    MissionDescription,
    ConversationMemory,
    SummaryMemory,
    EntityMemory,
)

from src.brain.data_types import Interaction, Entity, EntityResponse

import src.routers.schema.mission as api_schema_mission


class CRUD:
    def __init__(self, dbase: str):
        self._engine = sqlalchemy.create_engine(dbase)

        def _fk_pragma_on_connect(dbapi_con: Any, _: Any) -> None:
            dbapi_con.execute("pragma foreign_keys=ON")

        event.listen(self._engine, "connect", _fk_pragma_on_connect)

        self._sessionmaker = sessionmaker(self._engine)
        self._cleanse_unpersisted()

    def _cleanse_unpersisted(self) -> None:
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

    def get_mission_game_type(self, mission_id: int) -> api_schema_mission.GameType:
        with self._sessionmaker() as session:
            stmt = select(Mission.game_type).where(Mission.mission_id == mission_id)
            result = session.execute(stmt).scalar()
            if result is None:
                raise ValueError(f"No mission with id {mission_id}")
            return api_schema_mission.GameType(result)

    def insert_mission(
        self, mission: api_schema_mission.Mission
    ) -> api_schema_mission.Mission:
        self._cleanse_unpersisted()
        with self._sessionmaker() as session:
            db_mission = Mission(
                name=mission.name, game_type=mission.game_type.value, persist=False
            )
            session.add(db_mission)
            session.flush()
            db_mission_description = MissionDescription(
                mission_id=db_mission.mission_id,
                description=mission.description,
                background=mission.background,
            )
            session.add(db_mission_description)
            session.commit()
            mission.mission_id = db_mission.mission_id
            return mission

    def save_mission(self, mission: api_schema_mission.SaveMission) -> None:
        with self._sessionmaker() as session:
            stmt = (
                update(Mission)
                .where(Mission.mission_id == mission.mission_id)
                .values(persist=True, name_custom=mission.name_custom)
            )
            session.execute(stmt)
            session.commit()

    def get_mission_description(
        self, mission_id: int
    ) -> api_schema_mission.Mission | None:
        with self._sessionmaker() as session:
            stmt = select(Mission, MissionDescription).join(
                MissionDescription,
                and_(
                    MissionDescription.mission_id == Mission.mission_id,
                    Mission.mission_id == mission_id,
                ),
            )
            try:
                result = session.execute(stmt).one()
            except NoResultFound:
                return None

        return api_schema_mission.Mission(
            mission_id=result.Mission.mission_id,
            name=result.Mission.name,
            name_custom=result.Mission.name_custom,
            description=result.MissionDescription.description,
            game_type=api_schema_mission.GameType(result.Mission.game_type),
            background=result.MissionDescription.background,
        )

    def list_missions(self) -> list[api_schema_mission.Mission]:

        with self._sessionmaker() as session:
            stmt = (
                select(Mission, MissionDescription)
                .join(
                    MissionDescription,
                    Mission.mission_id == MissionDescription.mission_id,
                )
                .where(Mission.persist.is_(True))
                .order_by(Mission.mission_id)
            )
            results = session.execute(stmt).all()

            return [
                api_schema_mission.Mission(
                    mission_id=result.Mission.mission_id,
                    name=result.Mission.name,
                    name_custom=result.Mission.name_custom,
                    description=result.MissionDescription.description,
                    game_type=api_schema_mission.GameType(result.Mission.game_type),
                    background=result.MissionDescription.background,
                )
                for result in results
            ]

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

    def insert_interaction(self, mission_id: int, interaction: Interaction) -> None:
        memory = ConversationMemory(
            mission_id=mission_id,
            user_input=interaction.user_input,
            llm_output=interaction.llm_output,
        )
        with self._sessionmaker() as session:
            session.add(memory)
            session.commit()

    def update_last_interaction(
        self, mission_id: int, interaction: Interaction
    ) -> None:
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
                .order_by(ConversationMemory.conversation_memory_id.desc())
                .limit(1)
            )
            result = session.execute(stmt)
            conversation_memory = result.scalar_one_or_none()

            if conversation_memory is not None:
                conversation_memory.user_input = interaction.user_input
                conversation_memory.llm_output = interaction.llm_output
                session.commit()
            else:
                print("No ConversationMemory found for the given mission_id.")

    def get_summary(self, mission_id: int) -> tuple[str, int]:
        with self._sessionmaker() as session:
            stmt = select(SummaryMemory).where(SummaryMemory.mission_id == mission_id)
            existing_summary = session.execute(stmt).scalar_one_or_none()
            if existing_summary:
                return existing_summary.summary, existing_summary.n_summarized

            return "", 0

    def update_summary(self, mission_id: int, summary: str, n_summarized: int) -> None:
        with self._sessionmaker() as session:
            stmt = select(SummaryMemory).where(SummaryMemory.mission_id == mission_id)
            existing_summary = session.execute(stmt).scalar_one_or_none()

            if existing_summary:
                existing_summary.summary = summary
                existing_summary.n_summarized = n_summarized
            else:
                new_summary = SummaryMemory(
                    mission_id=mission_id, summary=summary, n_summarized=n_summarized
                )
                session.add(new_summary)

            try:
                session.commit()
            except IntegrityError:
                session.rollback()

    def get_entities(self, mission_id: int) -> list[Entity]:
        with self._sessionmaker() as session:
            stmt = (
                select(EntityMemory)
                .where(EntityMemory.mission_id == mission_id)
                .order_by(EntityMemory.type.asc())
            )
            result = session.execute(stmt).scalars().all()
            return [
                Entity(
                    name=memory.name,
                    type=memory.type,
                    summary=memory.summary,
                )
                for memory in result
            ]

    def update_entities(self, mission_id: int, entity_response: EntityResponse) -> None:
        with self._sessionmaker() as session:
            for entity in entity_response.updated_entities:
                stmt = select(EntityMemory).where(
                    EntityMemory.mission_id == mission_id,
                    EntityMemory.name == entity.name,
                )
                existing_entity = session.execute(stmt).scalar_one_or_none()

                if existing_entity is not None:
                    if entity.updated_name == "DELETE":
                        session.delete(existing_entity)
                    else:
                        existing_entity.summary = entity.summary
                        existing_entity.name = entity.updated_name
                else:
                    print(f"Updated Entity {entity.name} not found in the database.")
            session.commit()

            for entity in entity_response.entities:
                stmt = select(EntityMemory).where(
                    EntityMemory.mission_id == mission_id,
                    EntityMemory.name == entity.name,
                )
                existing_entity = session.execute(stmt).scalar_one_or_none()

                if existing_entity:
                    existing_entity.summary += "; " + entity.summary
                else:
                    new_entity = EntityMemory(
                        mission_id=mission_id,
                        name=entity.name,
                        type=entity.type,
                        summary=entity.summary,
                    )
                    session.add(new_entity)

                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()


crud_instance = CRUD(dbase="sqlite:///memory.db")
