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
    game_type: Mapped[str] = mapped_column(String(50))
    non_hero_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    persist: Mapped[bool] = mapped_column(Boolean)


class MissionDescription(Base):
    __tablename__ = "MissionDescription"
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    background: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)


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


class SummaryMemory(Base):
    __tablename__ = "SummaryMemory"
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[str] = mapped_column(Text)
    n_summarized: Mapped[int] = mapped_column(Integer)


class EntityMemory(Base):
    __tablename__ = "EntityMemory"
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    type: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)


class SceneMemory(Base):
    """
    Let the LLM create Summary of Scenes in the Mission.
    """

    __tablename__ = "SceneMemory"
    scene_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Mission.mission_id", ondelete="CASCADE"), primary_key=True
    )
    title: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(Text)
    characters: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class ConversationSummaryMemory(Base):
    """
    Let the LLM create a Summary of the Conversation Memory:
    Each user input and LLM output is to be stored as a one-senctence summary
    We can keep k complete interactions and n summerized interactions
    """

    __tablename__ = "ConversationSummaryMemory"
    conversation_memory_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ConversationMemory.conversation_memory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_input: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)
