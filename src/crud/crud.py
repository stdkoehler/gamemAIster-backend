"""CRUD operations for the mission database."""

import json
from dataclasses import dataclass
from typing import Any

from src.utils.logger import configure_logger

_log = configure_logger("crud")

import sqlalchemy
from sqlalchemy import event, select, update, delete, and_
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import sessionmaker

from src.crud.sqlmodel import (
    Mission,
    MissionDescription,
    ConversationMemory,
    SceneMemory,
    SummaryMemory,
    EntityMemory,
    CharacterSheet,
    UserLlmSettings,
)
from src.crud import secrets as secrets_crypto
from src.llmclient.known_local_models import resolve_local_mode

from src.brain.data_types import (
    Interaction,
    Entity,
    EntityResponse,
    Scene,
)

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.settings as api_schema_settings


@dataclass
class ResolvedLlmSettings:
    """The active provider's settings, with the API key decrypted — consumed
    by build_gamemaster."""

    provider: str
    model_name: str | None
    api_key: str | None
    local_host: str | None = None
    local_port: int | None = None
    local_interface: str | None = None
    local_mode: str | None = None


@dataclass
class LlmProviderOverview:
    """One provider's stored settings for display — never carries the
    decrypted key, only whether one exists."""

    provider: str
    model_name: str | None
    has_api_key: bool
    local_host: str | None
    local_port: int | None
    local_interface: str | None
    local_mode: str | None


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

    def verify_mission_user(self, mission_id: int, user_id: str) -> None:
        with self._sessionmaker() as session:
            stmt = select(Mission).where(
                Mission.mission_id == mission_id, Mission.user_id == user_id
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result is None:
                raise ValueError(f"No mission with id {mission_id} for user {user_id}")

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

    def get_mission_non_hero_mode(self, mission_id: int) -> bool:
        with self._sessionmaker() as session:
            stmt = select(Mission.non_hero_mode).where(Mission.mission_id == mission_id)
            result = session.execute(stmt).scalar()
            if result is None:
                raise ValueError(f"No mission with id {mission_id}")
            return result

    def get_mission_oracle(self, mission_id: int) -> bool:
        with self._sessionmaker() as session:
            stmt = select(Mission.oracle).where(Mission.mission_id == mission_id)
            result = session.execute(stmt).scalar()
            if result is None:
                raise ValueError(f"No mission with id {mission_id}")
            return result

    def insert_mission(
        self, mission: api_schema_mission.Mission
    ) -> api_schema_mission.Mission:
        self._cleanse_unpersisted()
        with self._sessionmaker() as session:
            db_mission = Mission(
                user_id=mission.user_id,
                name=mission.name,
                game_type=mission.game_type.value,
                non_hero_mode=mission.non_hero_mode,
                oracle=mission.oracle,
                persist=False,
            )
            session.add(db_mission)
            session.flush()
            db_mission_description = MissionDescription(
                mission_id=db_mission.mission_id,
                description=mission.description,
                background=mission.background,
                detailed_background=mission.detailed_background,
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
            user_id=result.Mission.user_id,
            name=result.Mission.name,
            name_custom=result.Mission.name_custom,
            description=result.MissionDescription.description,
            game_type=api_schema_mission.GameType(result.Mission.game_type),
            background=result.MissionDescription.background,
            detailed_background=result.MissionDescription.detailed_background,
            non_hero_mode=result.Mission.non_hero_mode,
            oracle=result.Mission.oracle,
        )

    def list_missions(self, user_id: str) -> list[api_schema_mission.Mission]:

        with self._sessionmaker() as session:
            stmt = (
                select(Mission, MissionDescription)
                .join(
                    MissionDescription,
                    Mission.mission_id == MissionDescription.mission_id,
                )
                .where(Mission.persist.is_(True), Mission.user_id == user_id)
                .order_by(Mission.mission_id)
            )
            results = session.execute(stmt).all()

            return [
                api_schema_mission.Mission(
                    user_id=result.Mission.user_id,
                    mission_id=result.Mission.mission_id,
                    name=result.Mission.name,
                    name_custom=result.Mission.name_custom,
                    description=result.MissionDescription.description,
                    game_type=api_schema_mission.GameType(result.Mission.game_type),
                    background=result.MissionDescription.background,
                    detailed_background=result.MissionDescription.detailed_background,
                    non_hero_mode=result.Mission.non_hero_mode,
                    oracle=result.Mission.oracle,
                )
                for result in results
            ]

    def get_interactions(
        self, mission_id: int, limit: int | None = None
    ) -> list[Interaction]:
        with self._sessionmaker() as session:
            stmt = select(ConversationMemory).join(
                Mission,
                and_(
                    Mission.mission_id == ConversationMemory.mission_id,
                    Mission.mission_id == mission_id,
                ),
            )
            if limit is not None:
                stmt = stmt.order_by(
                    ConversationMemory.conversation_memory_id.desc()
                ).limit(limit)
            else:
                stmt = stmt.order_by(ConversationMemory.conversation_memory_id.asc())
            result = session.execute(stmt).scalars().all()
            if limit is not None:
                result = list(reversed(result))
            return [
                Interaction(
                    id_=memory.conversation_memory_id,
                    user_input=memory.user_input,
                    llm_output=memory.llm_output,
                    llm_thinking=memory.llm_thinking,
                    llm_thinking_signature=memory.llm_thinking_signature,
                )
                for memory in result
            ]

    def insert_interaction(self, mission_id: int, interaction: Interaction) -> None:
        memory = ConversationMemory(
            mission_id=mission_id,
            user_input=interaction.user_input,
            llm_output=interaction.llm_output,
            llm_thinking=interaction.llm_thinking,
            llm_thinking_signature=interaction.llm_thinking_signature,
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
                # only update if llm_thinking and llm_signature is not none
                if interaction.llm_thinking is not None:
                    conversation_memory.llm_thinking = interaction.llm_thinking
                if interaction.llm_thinking_signature is not None:
                    conversation_memory.llm_thinking_signature = (
                        interaction.llm_thinking_signature
                    )
                session.commit()
            else:
                _log.warning("No ConversationMemory found | mission_id=%s", mission_id)

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

    def get_digest(self, mission_id: int) -> tuple[str, int]:
        """Returns (digest, digest_ledger_tokens) — the bounded "story so far"
        derived from the ledger, and the ledger's token count at the time it
        was last refreshed. Empty/0 if no digest has been computed yet (the
        ledger hasn't crossed digest_budget_tokens)."""
        with self._sessionmaker() as session:
            stmt = select(SummaryMemory).where(SummaryMemory.mission_id == mission_id)
            existing_summary = session.execute(stmt).scalar_one_or_none()
            if existing_summary:
                return existing_summary.digest, existing_summary.digest_ledger_tokens
            return "", 0

    def update_digest(
        self, mission_id: int, digest: str, digest_ledger_tokens: int
    ) -> None:
        with self._sessionmaker() as session:
            stmt = select(SummaryMemory).where(SummaryMemory.mission_id == mission_id)
            existing_summary = session.execute(stmt).scalar_one_or_none()
            if not existing_summary:
                # update_summary always runs first in the same cycle, so this
                # shouldn't happen in practice.
                _log.warning(
                    "update_digest called with no existing SummaryMemory row | mission_id=%s",
                    mission_id,
                )
                return

            existing_summary.digest = digest
            existing_summary.digest_ledger_tokens = digest_ledger_tokens

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
                    matched_key_npc=memory.matched_key_npc,
                )
                for memory in result
            ]

    def update_entities(self, mission_id: int, entity_response: EntityResponse) -> None:
        with self._sessionmaker() as session:
            try:
                # Fetch existing entities
                existing: dict[str, EntityMemory] = {
                    e.name: e
                    for e in session.execute(
                        select(EntityMemory).where(
                            EntityMemory.mission_id == mission_id
                        )
                    ).scalars()
                }
                # Secondary index on the roster link: a name-independent identity
                # for entities matched to a keyNPCs (or equivalent) roster entry.
                # Lets us resolve an LLM-named entity to the right DB row even if
                # the model used an inconsistent display name this call (e.g.
                # "the High Priestess" vs. the previously-stored "Devana") —
                # exact-name matching alone can't catch that, since storage is
                # keyed on name.
                existing_by_match: dict[str, EntityMemory] = {
                    e.matched_key_npc.strip().lower(): e
                    for e in existing.values()
                    if e.matched_key_npc
                }

                def resolve(name: str, matched_key_npc: str | None) -> EntityMemory | None:
                    if matched_key_npc:
                        match = existing_by_match.get(matched_key_npc.strip().lower())
                        if match:
                            return match
                    return existing.get(name)

                def reindex(db_entity: EntityMemory, new_name: str | None, new_match: str | None) -> None:
                    """Keep both lookup dicts in sync after a name/match change."""
                    if db_entity.matched_key_npc:
                        existing_by_match.pop(db_entity.matched_key_npc.strip().lower(), None)
                    if new_name is not None and new_name != db_entity.name:
                        existing.pop(db_entity.name, None)
                        existing[new_name] = db_entity
                        db_entity.name = new_name
                    if new_match is not None:
                        db_entity.matched_key_npc = new_match
                    if db_entity.matched_key_npc:
                        existing_by_match[db_entity.matched_key_npc.strip().lower()] = db_entity

                # 1. HANDLE DELETIONS FIRST (Standard and Overwrites)
                for updated_entity in entity_response.updated_entities:
                    db_entity = resolve(updated_entity.name, updated_entity.matched_key_npc)
                    if not db_entity:
                        continue

                    # Case A: Explicit deletion request
                    if updated_entity.updated_name == "DELETE":
                        session.delete(db_entity)
                        existing.pop(db_entity.name, None)
                        if db_entity.matched_key_npc:
                            existing_by_match.pop(db_entity.matched_key_npc.strip().lower(), None)

                    # Case B: Rename Collision (Delete the target to make room)
                    elif updated_entity.updated_name != db_entity.name:
                        collision_entity = existing.get(updated_entity.updated_name)
                        if collision_entity and collision_entity is not db_entity:
                            session.delete(collision_entity)
                            existing.pop(updated_entity.updated_name, None)
                            # We don't pop from existing yet, we'll overwrite it in step 2

                # IMPORTANT: Flush deletions to the DB so the names are "freed up"
                session.flush()

                # 2. HANDLE RENAMES AND UPDATES
                for updated_entity in entity_response.updated_entities:
                    if updated_entity.updated_name == "DELETE":
                        continue

                    db_entity = resolve(updated_entity.name, updated_entity.matched_key_npc)
                    if db_entity:
                        reindex(db_entity, updated_entity.updated_name, updated_entity.matched_key_npc)
                        db_entity.summary = updated_entity.summary

                # 3. PROCESS NEW ENTITIES
                for entity in entity_response.entities:
                    db_entity = resolve(entity.name, entity.matched_key_npc)
                    if db_entity:
                        db_entity.summary += "; " + entity.summary
                        if entity.matched_key_npc and not db_entity.matched_key_npc:
                            reindex(db_entity, None, entity.matched_key_npc)
                    else:
                        new_entity = EntityMemory(
                            mission_id=mission_id,
                            name=entity.name,
                            type=entity.type,
                            summary=entity.summary,
                            matched_key_npc=entity.matched_key_npc,
                        )
                        session.add(new_entity)
                        existing[entity.name] = new_entity
                        if entity.matched_key_npc:
                            existing_by_match[entity.matched_key_npc.strip().lower()] = new_entity

                session.commit()

            except Exception as e:
                session.rollback()
                _log.error("Entity update failed | mission_id=%s | error=%s", mission_id, e)
                raise

    def get_scenes(self, mission_id: int) -> list[Scene]:
        with self._sessionmaker() as session:
            stmt = (
                select(SceneMemory)
                .where(SceneMemory.mission_id == mission_id)
                .order_by(SceneMemory.scene_id.asc())
            )
            result = session.execute(stmt).scalars().all()
            return [
                Scene(
                    id=memory.scene_id,
                    title=memory.title,
                    location=memory.location,
                    characters=json.loads(memory.characters),
                    summary=memory.summary,
                    completed=memory.completed,
                )
                for memory in result
            ]

    def update_scenes(self, mission_id: int, scenes: list[Scene]) -> None:
        with self._sessionmaker() as session:
            for scene in scenes:
                stmt = select(SceneMemory).where(
                    SceneMemory.mission_id == mission_id,
                    SceneMemory.scene_id == scene.id,
                )
                existing_scene = session.execute(stmt).scalar_one_or_none()

                if existing_scene is not None:
                    existing_scene.title = scene.title
                    existing_scene.location = scene.location
                    existing_scene.characters = json.dumps(scene.characters)
                    existing_scene.summary = scene.summary
                    existing_scene.completed = scene.completed
                else:
                    new_scene = SceneMemory(
                        mission_id=mission_id,
                        scene_id=scene.id,
                        title=scene.title,
                        location=scene.location,
                        characters=json.dumps(scene.characters),
                        summary=scene.summary,
                        completed=scene.completed,
                    )
                    session.add(new_scene)

            session.commit()

    # ------------------------------------------------------------------
    # Character sheets
    # ------------------------------------------------------------------

    def get_character_sheets(
        self, mission_id: int
    ) -> list[api_schema_mission.CharacterSheetSchema]:
        with self._sessionmaker() as session:
            stmt = select(CharacterSheet).where(CharacterSheet.mission_id == mission_id)
            rows = session.execute(stmt).scalars().all()
            return [
                api_schema_mission.CharacterSheetSchema(
                    character_sheet_id=row.character_sheet_id,
                    mission_id=row.mission_id,
                    name=row.name,
                    game_type=row.game_type,
                    content=json.loads(row.content),
                    is_protagonist=row.is_protagonist,
                    is_npc=row.is_npc,
                    is_active=row.is_active,
                    matched_key_npc=row.matched_key_npc,
                )
                for row in rows
            ]

    def upsert_character_sheet(
        self, sheet: api_schema_mission.UpsertCharacterSheet
    ) -> api_schema_mission.CharacterSheetSchema:
        with self._sessionmaker() as session:
            if sheet.is_protagonist:
                session.execute(
                    update(CharacterSheet)
                    .where(CharacterSheet.mission_id == sheet.mission_id)
                    .values(is_protagonist=False)
                )

            if sheet.character_sheet_id is not None:
                stmt = select(CharacterSheet).where(
                    CharacterSheet.character_sheet_id == sheet.character_sheet_id,
                    CharacterSheet.mission_id == sheet.mission_id,
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row is None:
                    raise ValueError(
                        f"CharacterSheet {sheet.character_sheet_id} not found for mission {sheet.mission_id}"
                    )
                row.name = sheet.name
                row.game_type = sheet.game_type
                row.content = json.dumps(sheet.content)
                row.is_protagonist = sheet.is_protagonist
                row.is_npc = sheet.is_npc
                row.is_active = sheet.is_active
                row.matched_key_npc = sheet.matched_key_npc
            else:
                row = CharacterSheet(
                    mission_id=sheet.mission_id,
                    name=sheet.name,
                    game_type=sheet.game_type,
                    content=json.dumps(sheet.content),
                    is_protagonist=sheet.is_protagonist,
                    is_npc=sheet.is_npc,
                    is_active=sheet.is_active,
                    matched_key_npc=sheet.matched_key_npc,
                )
                session.add(row)
                session.flush()

            session.commit()
            return api_schema_mission.CharacterSheetSchema(
                character_sheet_id=row.character_sheet_id,
                mission_id=row.mission_id,
                name=row.name,
                game_type=row.game_type,
                content=json.loads(row.content),
                is_protagonist=row.is_protagonist,
                is_npc=row.is_npc,
                is_active=row.is_active,
                matched_key_npc=row.matched_key_npc,
            )

    def delete_character_sheet(self, character_sheet_id: int, mission_id: int) -> None:
        with self._sessionmaker() as session:
            stmt = delete(CharacterSheet).where(
                CharacterSheet.character_sheet_id == character_sheet_id,
                CharacterSheet.mission_id == mission_id,
            )
            session.execute(stmt)

    def set_character_sheet_active(
        self, character_sheet_id: int, mission_id: int, is_active: bool
    ) -> None:
        with self._sessionmaker() as session:
            stmt = (
                update(CharacterSheet)
                .where(
                    CharacterSheet.character_sheet_id == character_sheet_id,
                    CharacterSheet.mission_id == mission_id,
                )
                .values(is_active=is_active)
            )
            result = session.execute(stmt)
            if result.rowcount == 0:
                raise ValueError(
                    f"CharacterSheet {character_sheet_id} not found for mission {mission_id}"
                )
            session.commit()


    # ------------------------------------------------------------------
    # Per-user LLM settings
    # ------------------------------------------------------------------

    def get_llm_settings(self, user_id: str) -> ResolvedLlmSettings | None:
        """The user's *active* provider settings, key decrypted — what
        build_gamemaster uses. None if the user has stored nothing."""
        with self._sessionmaker() as session:
            stmt = select(UserLlmSettings).where(
                UserLlmSettings.user_id == user_id,
                UserLlmSettings.is_active.is_(True),
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                return None
            api_key: str | None = None
            if row.api_key_encrypted:
                try:
                    api_key = secrets_crypto.decrypt(row.api_key_encrypted)
                except Exception:
                    # A rotated/invalid SETTINGS_ENCRYPTION_KEY or corrupt
                    # ciphertext must not brick the user: providers that need no
                    # key (LOCAL) keep working, and key-based providers surface a
                    # clean "API key not set" instead of a 500. Re-saving the key
                    # recovers.
                    _log.warning(
                        "Failed to decrypt stored API key | user_id=%s", user_id
                    )
            return ResolvedLlmSettings(
                provider=row.provider,
                model_name=row.model_name,
                api_key=api_key,
                local_host=row.local_host,
                local_port=row.local_port,
                local_interface=row.local_interface,
                local_mode=row.local_mode,
            )

    def list_llm_settings(
        self, user_id: str
    ) -> tuple[str | None, list[LlmProviderOverview]]:
        """All of the user's saved per-provider settings (no decrypted keys)
        plus which provider is active — for populating the settings UI."""
        with self._sessionmaker() as session:
            stmt = select(UserLlmSettings).where(UserLlmSettings.user_id == user_id)
            rows = session.execute(stmt).scalars().all()
            active = next((r.provider for r in rows if r.is_active), None)
            overviews = [
                LlmProviderOverview(
                    provider=r.provider,
                    model_name=r.model_name,
                    has_api_key=r.api_key_encrypted is not None,
                    local_host=r.local_host,
                    local_port=r.local_port,
                    local_interface=r.local_interface,
                    local_mode=r.local_mode,
                )
                for r in rows
            ]
            return active, overviews

    def upsert_llm_settings(
        self, user_id: str, settings: api_schema_settings.SaveLlmSettings
    ) -> None:
        """Save one provider's settings and make it the active provider. Each
        provider keeps its own row/key, so a blank api_key keeps *that*
        provider's existing key and never touches another provider's."""
        provider = settings.provider.value
        with self._sessionmaker() as session:
            # Exactly one active provider per user.
            session.execute(
                update(UserLlmSettings)
                .where(
                    UserLlmSettings.user_id == user_id,
                    UserLlmSettings.provider != provider,
                )
                .values(is_active=False)
            )
            stmt = select(UserLlmSettings).where(
                UserLlmSettings.user_id == user_id,
                UserLlmSettings.provider == provider,
            )
            row = session.execute(stmt).scalar_one_or_none()
            api_key_encrypted = (
                secrets_crypto.encrypt(settings.api_key) if settings.api_key else None
            )
            local_interface = (
                settings.local_interface.value if settings.local_interface else None
            )
            # For a known preset, which local client it needs is a property of
            # the model, not a user choice — the registry overrides whatever
            # the client sent. For a custom model the registry can't derive
            # it, so the client's requested_mode is honored (defaulting to
            # NATIVE_COMPLETIONS). See resolve_local_mode's docstring.
            local_mode = (
                resolve_local_mode(settings.model_name, settings.local_mode).value
                if settings.provider == api_schema_settings.LlmProvider.LOCAL
                else None
            )
            if row is not None:
                row.is_active = True
                row.model_name = settings.model_name
                # A blank api_key on update keeps this provider's existing key
                # rather than clearing it.
                if settings.api_key:
                    row.api_key_encrypted = api_key_encrypted
                row.local_host = settings.local_host
                row.local_port = settings.local_port
                row.local_interface = local_interface
                row.local_mode = local_mode
            else:
                session.add(
                    UserLlmSettings(
                        user_id=user_id,
                        provider=provider,
                        is_active=True,
                        model_name=settings.model_name,
                        api_key_encrypted=api_key_encrypted,
                        local_host=settings.local_host,
                        local_port=settings.local_port,
                        local_interface=local_interface,
                        local_mode=local_mode,
                    )
                )
            session.commit()

    def delete_llm_settings(self, user_id: str) -> None:
        """Remove *all* of the user's stored provider settings so
        `build_gamemaster` falls back to the deployment's env-var config again.
        No-op if none exist."""
        with self._sessionmaker() as session:
            session.execute(
                delete(UserLlmSettings).where(UserLlmSettings.user_id == user_id)
            )
            session.commit()


crud_instance = CRUD(dbase="sqlite:///memory.db")
