"""Types"""

import re
from dataclasses import dataclass
from pydantic import BaseModel


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
