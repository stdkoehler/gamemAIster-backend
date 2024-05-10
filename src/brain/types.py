""" Types """

from enum import Enum
from pydantic import BaseModel


class Actor(Enum):
    """
    An enumeration class representing the actors in a chat conversation.

    Attributes:
        USER (str): Represents the user in the conversation.
        LLM (str): Represents the AI language model in the conversation.
    """

    SYSTEM = "<|im_start|>system"
    SYSTEM_END = "<|im_end|>"
    USER = "<|im_start|>user\n{msg}<|im_end|>"
    LLM = "<|im_start|>assistant\n{msg}<|im_end|>"


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

    def format_interaction(self):
        """
        Formats the interaction by combining the formatted user input and the
        formatted AI language model output.

        Returns:
            str: The formatted interaction.
        """
        return (
            f"{self.format_user_input(self._user_input)}\n"
            f"{self.format_llm_output(self._llm_output)}"
        )

    def format_interaction_summary(self):
        """
        Formats the interaction by combining the formatted user input and the
        formatted AI language model output. Use this for summarizing the interaction
        and for entity extraction. (not using the LLM Intruct keywords)

        Returns:
            str: The formatted interaction.
        """
        return f"Player: {self._user_input}\n" f"Gamemaster: {self._llm_output}"

    @property
    def user_input(self):
        return self._user_input

    @property
    def llm_output(self):
        return self._llm_output

    @property
    def user_input_formatted(self):
        return Interaction.format_user_input(self._user_input)

    @property
    def llm_output_formatted(self):
        return Interaction.format_llm_output(self._llm_output)

    @staticmethod
    def format_user_input(user_input: str):
        """
        Formats the user input by combining it with the actor prefix.

        Args:
            user_input (str): The user input to be formatted.

        Returns:
            str: The formatted user input with the actor prefix.
        """
        return Actor.USER.value.format(msg=user_input)

    @staticmethod
    def format_llm_output(llm_output: str):
        """
        Formats the user input by combining it with the actor prefix.

        Args:
            llm_output (str): The llm output to be formatted.

        Returns:
            str: The formatted llm output with the llm prefix.
        """
        return Actor.LLM.value.format(msg=llm_output)


class Entity(BaseModel):
    """
    A class representing an entity extracted from a text.
    """

    name: str
    type: str
    summary: str
