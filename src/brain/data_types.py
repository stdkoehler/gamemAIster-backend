"""Types"""

import re
from dataclasses import dataclass
from pydantic import BaseModel, model_validator, ValidationInfo


@dataclass(frozen=True)
class Interaction:
    user_input: str
    llm_output: str
    llm_thinking: str | None = None
    llm_thinking_signature: str | None = None
    id_: int | None = None

    def format_interaction_summary(self) -> str:
        """
        Formats the interaction for use in summarization and entity extraction
        (strips OOC blocks, think tags, and "What do you do?" prompts).
        """

        def cleanse_text(text: str) -> str:
            ooc_pattern = r"(?si)\s*[\[\(]OOC:.*?[\)\]]"
            think_pattern = r"(?si)<think>.*?</think>"
            wdyd_pattern = r"(?si)---(?:\s+)?\**What do you do.*"
            text = re.sub(ooc_pattern, "", text)
            text = re.sub(think_pattern, "", text)
            return re.sub(wdyd_pattern, "", text)

        return (
            f"Player: {cleanse_text(self.user_input)}\n"
            f"Gamemaster: {cleanse_text(self.llm_output)}"
        )


class Entity(BaseModel):
    """
    A class representing an entity extracted from a text.
    """

    name: str
    type: str
    summary: str
    # Exact `name` of the mission's keyNPCs (or equivalent) roster entry this
    # entity was matched to, if any. A real link independent of `name` —
    # lets storage merge an entity onto the same roster NPC even if the LLM
    # used an inconsistent display name (title vs. given name) across calls.
    matched_key_npc: str | None = None


class Scene(BaseModel):
    """
    A class representing a scene in a mission.
    """

    id: int
    title: str
    location: str
    characters: list[str]
    summary: str
    completed: bool


class UpdatedEntity(Entity):
    """
    A class representing an entity extracted from a text.
    """

    updated_name: str


class EntityResponse(BaseModel):
    """
    A class representing a response containing a list of entities and updated entities.
    """

    entities: list[Entity]
    updated_entities: list[UpdatedEntity]

    @model_validator(mode="after")
    def _validate_references(self, info: ValidationInfo) -> "EntityResponse":
        """
        Enforces two reference-integrity invariants the prompt alone can't:
        every `matched_key_npc` must name a real entry in the mission's NPC
        roster (crud.update_entities() trusts this field completely, with no
        fuzzy fallback, so a hallucinated value would silently corrupt the
        roster cross-reference), and every `updated_entities[].name` must
        reference a previously known entity (otherwise that update silently
        no-ops today — the row it was supposed to touch doesn't exist, and
        nothing tells the model).

        Needs context the model itself doesn't carry (the roster, the
        previously known entity names) — passed in via pydantic_ai's
        `validation_context` on the Agent, which pydantic threads through to
        `info.context` here. See SummaryMemory.extract_entities().
        """
        context = info.context or {}
        known_names: set[str] = context.get("known_entity_names", set())
        roster_names: set[str] = context.get("roster_names", set())
        if not known_names and not roster_names:
            # No context supplied (e.g. validating outside extract_entities'
            # Agent) — nothing to check against, skip rather than reject everything.
            return self

        errors = []
        for entity in list(self.entities) + list(self.updated_entities):
            if entity.matched_key_npc and entity.matched_key_npc.strip().lower() not in roster_names:
                errors.append(
                    f"entity '{entity.name}' has matched_key_npc={entity.matched_key_npc!r}, "
                    "which does not match any name in known_npc_roster"
                )
        for updated in self.updated_entities:
            if updated.updated_name != "DELETE" and updated.name.strip().lower() not in known_names:
                errors.append(
                    f"updated_entities references '{updated.name}', which is not in the "
                    "previously known entities list — this update would silently do nothing"
                )
        if errors:
            raise ValueError(
                "Invalid entity references:\n"
                + "\n".join(f"- {e}" for e in errors)
                + f"\n\nValid known entity names: {sorted(known_names) or '(none yet)'}"
                + f"\nValid known_npc_roster names: {sorted(roster_names) or '(none)'}"
            )
        return self
