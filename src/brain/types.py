""" Types """

from enum import Enum
from dataclasses import dataclass, field


class Actor(Enum):
    """
    An enumeration class representing the actors in a chat conversation.

    Attributes:
        USER (str): Represents the user in the conversation.
        LLM (str): Represents the AI language model in the conversation.
    """

    SYSTEM = "<s>"
    USER = "PL"
    LLM = "GM"
    SYSTEM_END = ""


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
        Formats the interaction by combining the formatted user input and the formatted AI language model output.

        Returns:
            str: The formatted interaction.
        """
        return (
            f"{self.format_user_input(self._user_input)}\n"
            f"{self.format_llm_output(self._llm_output)}"
        )

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
        return f"{Actor.USER.value}: {user_input}"

    @staticmethod
    def format_llm_output(llm_output: str):
        """
        Formats the user input by combining it with the actor prefix.

        Args:
            llm_output (str): The llm output to be formatted.

        Returns:
            str: The formatted llm output with the llm prefix.
        """
        return f"{Actor.LLM.value}: {llm_output}"
