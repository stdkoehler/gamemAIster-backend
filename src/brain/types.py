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


@dataclass
class LLMConfig:
    """
    A class representing the configuration for prompts in a chat conversation.

    Attributes:
        max_tokens (int): The maximum number of tokens allowed in a generated response.
        temperature (float): The temperature value for controlling the creativity of responses.
        top_p (float): The top-p value for controlling the diversity of responses.
        min_p (int): The minimum number of tokens to keep in a generated response.
        top_k (int): The top-k value for creating randomness in responses. (1 = no randomness)
        frequency_penalty (float): The penalty value for penalizing frequent tokens in responses.
        stop_words (list[str]): A list of stop words to which stop continuing response generation.
    """

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0
    top_k: int = 20
    repetition_penalty: float = 1  # nous capbyara #1.15 mixtral
    presence_penalty: float = 0
    frequency_penalty: float = 0
    guidance_scale: float = 1
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    smoothing_factor: float = 0
    repetition_penalty_range: int = 1024
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    sampler_priority: list[str] = field(default_factory=list)
    stop: list[str] = field(default_factory=list)
