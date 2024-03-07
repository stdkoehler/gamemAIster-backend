from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "Memory"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session: Mapped[str] = mapped_column(String(50))
    user_input: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("id", "session", name="id_session"),)
