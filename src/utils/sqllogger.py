from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Callable, Any


Base = declarative_base()


class SummaryLog(Base):
    __tablename__ = "summary_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    llm_input = Column(Text)
    raw_output = Column(Text)
    processed_output = Column(Text)


class EntityLog(Base):
    __tablename__ = "entity_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    llm_input = Column(Text)
    raw_output = Column(Text)
    processed_output = Column(Text)


class SceneLog(Base):
    __tablename__ = "scene_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    llm_input = Column(Text)
    raw_output = Column(Text)
    processed_output = Column(Text)


class SQLLogger:
    def __init__(self, db_url: str = "sqlite:///logs/llm_logs.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _add_log(
        self,
        model_class: type[SummaryLog] | type[EntityLog] | type[SceneLog],
        llm_input: str,
        raw_output: str,
        processed_output: str = "",
    ) -> None:
        session = self.Session()
        try:
            if isinstance(llm_input, list):
                llm_input = str(llm_input)  # Convert message lists to string

            log_entry = model_class(
                llm_input=llm_input,
                raw_output=raw_output,
                processed_output=processed_output,
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging to database: {e}")
        finally:
            session.close()

    def log_summary(
        self, llm_input: str, raw_output: str, processed_output: str = ""
    ) -> None:
        self._add_log(SummaryLog, llm_input, raw_output, processed_output)

    def log_entity(
        self, llm_input: str, raw_output: str, processed_output: str = ""
    ) -> None:
        self._add_log(EntityLog, llm_input, raw_output, processed_output)

    def log_scene(
        self, llm_input: str, raw_output: str, processed_output: str = ""
    ) -> None:
        self._add_log(SceneLog, llm_input, raw_output, processed_output)
