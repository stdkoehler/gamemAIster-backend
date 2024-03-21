from sqlalchemy import String, Integer, Boolean, Text, UniqueConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class Mission(Base):
    __tablename__ = "Mission"
    mission_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name_custom: Mapped[str] = mapped_column(String(50), default="")
    name: Mapped[str] = mapped_column(String(50))
    persist: Mapped[bool] = mapped_column(Boolean)


class MissionDescription(Base):
    __tablename__ = "MissionDescription"
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    description: Mapped[str] = mapped_column(Text)


class MissionScene(Base):
    __tablename__ = "MissionScene"
    scene_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[str] = mapped_column(Text)


class ConversationMemory(Base):
    __tablename__ = "ConversationMemory"
    conversation_memory_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE")
    )
    user_input: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "conversation_memory_id", "mission_id", name="conversation_memory_mission"
        ),
    )
