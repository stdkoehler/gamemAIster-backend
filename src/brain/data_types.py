"""Types"""

import re
from enum import Enum
from pydantic import BaseModel


class Interaction:
    """
    A class representing an interaction in a chat conversation.

    Attributes:
        _id (str): The unique identifier of the interaction.
        _user_input (str): The user input in the interaction.
        _llm_output (str): The AI language model output in the interaction.
    """

    def __init__(self, user_input: str, llm_output: str, id_: int | None = None):
        self._id = id_
        self._user_input = user_input
        self._llm_output = llm_output

    def format_interaction_summary(self) -> str:
        """
        Formats the interaction by combining the formatted user input and the
        formatted AI language model output. Use this for summarizing the interaction
        and for entity extraction. (not using the LLM Intruct keywords)

        Returns:
            str: The formatted interaction.
        """

        def cleanse_text(text: str) -> str:
            """Cleanses the text by removing OOC and "What do you do" sections."""
            ooc_pattern = r"(?si)\s*[\[\(]OOC:.*?[\)\]]"
            think_pattern = r"(?si)<think>.*?</think>"
            wdyd_pattern = r"(?si)---(?:\s+)?\**What do you do.*"
            text = re.sub(ooc_pattern, "", text)
            text = re.sub(think_pattern, "", text)
            return re.sub(wdyd_pattern, "", text)

        clean_user_input = cleanse_text(self._user_input)
        clean_llm_output = cleanse_text(self._llm_output)
        return f"Player: {clean_user_input}\n" f"Gamemaster: {clean_llm_output}"

    @property
    def user_input(self) -> str:
        return self._user_input

    @property
    def llm_output(self) -> str:
        return self._llm_output


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
