from sqlalchemy import String, Integer, Boolean, Text, UniqueConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "Session"
    session_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String(50))
    persist: Mapped[bool] = mapped_column(Boolean)


class SessionDescription(Base):
    __tablename__ = "SessionDescription"
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Session.session_id", ondelete="CASCADE"), primary_key=True
    )
    description: Mapped[str] = mapped_column(Text)


class SessionScene(Base):
    __tablename__ = "SessionScene"
    scene_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Session.session_id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[str] = mapped_column(Text)


class ConversationMemory(Base):
    __tablename__ = "ConversationMemory"
    conversation_memory_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("Session.session_id", ondelete="CASCADE")
    )
    user_input: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "conversation_memory_id", "session_id", name="conversation_memory_session"
        ),
    )
