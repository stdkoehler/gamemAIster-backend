import logging
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy import Column, DateTime, Index, Integer, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.logger import configure_logger

Base = declarative_base()
_log = configure_logger("sqllogger")


class LogType(StrEnum):
    SUMMARY = "summary"
    ENTITY = "entity"
    SCENE = "scene"


class LLMCallLog(Base):
    """Summary, entity, and scene extraction calls — same I/O shape, filtered by log_type."""

    __tablename__ = "llm_call_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(Text)
    log_type = Column(Text)       # LogType value
    llm_input = Column(Text)
    llm_output = Column(Text)
    extracted_json = Column(Text)
    processed_output = Column(Text)

    __table_args__ = (
        Index("ix_llm_call_logs_log_type", "log_type"),
        Index("ix_llm_call_logs_correlation_id", "correlation_id"),
    )


class MissionLog(Base):
    """Full mission-generation flow: oracle roll → alignment → LLM output."""

    __tablename__ = "mission_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(Text)
    game_type = Column(Text)
    oracle_used = Column(Text)
    oracle_background = Column(Text)   # character background fed into oracle
    oracle_roll = Column(Text)         # raw random draw from pools
    oracle_aligned = Column(Text)      # LLM-aligned oracle result
    llm_input = Column(Text)           # messages sent to mission generation LLM
    llm_output = Column(Text)          # parsed mission JSON

    __table_args__ = (Index("ix_mission_logs_correlation_id", "correlation_id"),)


class ReasoningLog(Base):
    """LLM thinking/reasoning blocks captured per provider."""

    __tablename__ = "reasoning_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(Text)
    provider = Column(Text)
    content = Column(Text)

    __table_args__ = (Index("ix_reasoning_logs_correlation_id", "correlation_id"),)


class ParseRepairLog(Base):
    """Records each JSON repair attempt made by parse_with_retry."""

    __tablename__ = "parse_repair_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(Text)
    model_type = Column(Text)       # Pydantic model class name
    attempt = Column(Integer)       # 1-indexed repair attempt number
    validation_error = Column(Text) # error that triggered the repair
    bad_response = Column(Text)     # the LLM output that failed to parse
    repair_output = Column(Text)    # what the repair LLM produced
    success = Column(Text)          # "true" / "false"

    __table_args__ = (Index("ix_parse_repair_logs_model_type", "model_type"),)


class SQLLogger:
    def __init__(self) -> None:
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        engine = create_engine(f"sqlite:///{logs_dir}/llm_logs.db")
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    def _write(self, model_class: type, label: str, **kwargs: object) -> str:
        cid = uuid.uuid4().hex[:8]
        session = self._Session()
        try:
            session.add(model_class(correlation_id=cid, **kwargs))
            session.commit()
            _log.info("DB write | table=%s | cid=%s", label, cid)
        except Exception as e:
            session.rollback()
            _log.error("DB write failed | table=%s | cid=%s | error=%s", label, cid, e)
        finally:
            session.close()
        return cid

    def log_llm_call(
        self,
        log_type: LogType,
        llm_input: str,
        llm_output: str,
        extracted_json: str = "",
        processed_output: str = "",
    ) -> str:
        return self._write(
            LLMCallLog,
            label=f"llm_call_logs/{log_type}",
            log_type=log_type,
            llm_input=llm_input if isinstance(llm_input, str) else str(llm_input),
            llm_output=llm_output,
            extracted_json=extracted_json,
            processed_output=processed_output,
        )

    def log_mission(
        self,
        game_type: str,
        oracle_used: bool,
        oracle_background: str,
        oracle_roll: str,
        oracle_aligned: str,
        llm_input: str,
        llm_output: str,
    ) -> str:
        return self._write(
            MissionLog,
            label="mission_logs",
            game_type=game_type,
            oracle_used=str(oracle_used),
            oracle_background=oracle_background,
            oracle_roll=oracle_roll,
            oracle_aligned=oracle_aligned,
            llm_input=llm_input if isinstance(llm_input, str) else str(llm_input),
            llm_output=llm_output,
        )

    def log_reasoning(self, provider: str, content: str) -> str:
        return self._write(
            ReasoningLog,
            label="reasoning_logs",
            provider=provider,
            content=content,
        )

    def log_parse_repair(
        self,
        model_type: str,
        attempt: int,
        validation_error: str,
        bad_response: str,
        repair_output: str,
        success: bool,
    ) -> str:
        return self._write(
            ParseRepairLog,
            label="parse_repair_logs",
            model_type=model_type,
            attempt=attempt,
            validation_error=validation_error,
            bad_response=bad_response,
            repair_output=repair_output,
            success=str(success).lower(),
        )
