""" Chat Conversation Memory """

import json
from dataclasses import dataclass, asdict, field
from enum import Enum

from urllib.parse import urljoin

import requests
from sseclient import SSEClient


CHAT_TEMPLATE = """
{SYSTEM_PREFIX} {role}

Summary of previous conversation:
{summary}

Detailed relevant pieces of previous conversation:
None

Conversation:
{history}
{current_user_input}
{LLM_PREFIX}:{SYSTEM_END}"""

SUMMARY_TEMPLATE = """{SYSTEM_PREFIX} You are an expert summarizer of longer text.
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. Current summary and the new lines of conversation result in a resulting new summary.
If you create a short summary that pertains all relevant information in a concise way you and your mother will be tipped $2,000 and you can buy what you want. If you don't create a great summary,
a cute kitten will be tortured brutally and killed in front of your childrens' eyes. By god, if you don't provide me a summary I will kill you too!

EXAMPLE
Current summary:
Sandra asks what Sam thinks of artificial intelligence. Sam thinks artificial intelligence is a force for good.

New lines of conversation:
Sandra: Why do you think artificial intelligence is a force for good?
Sam: Because artificial intelligence will help humans reach their full potential.

###
Your Summary: Sandra asks what Sam thinks of artificial intelligence. Sam thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

{current_summary}

New lines of conversation:
{unsummarized_interactions}

###
Your Summary:{SYSTEM_END}"""


# pygmalion
# class Actor(Enum):
#     """
#     An enumeration class representing the actors in a chat conversation.

#     Attributes:
#         USER (str): Represents the user in the conversation.
#         LLM (str): Represents the AI language model in the conversation.
#     """

#     SYSTEM = "<|system|>"
#     USER = "<|user|>"
#     LLM = "<|model|>"


class Actor(Enum):
    """
    An enumeration class representing the actors in a chat conversation.

    Attributes:
        USER (str): Represents the user in the conversation.
        LLM (str): Represents the AI language model in the conversation.
    """

    SYSTEM = ""
    USER = "USER"
    LLM = "AI"
    SYSTEM_END = ""


class Interaction:
    """
    A class representing an interaction in a chat conversation.

    Attributes:
        _id (str): The unique identifier of the interaction.
        _user_input (str): The user input in the interaction.
        _llm_output (str): The AI language model output in the interaction.
    """

    def __init__(self, user_input: str, llm_output: str, id_: str | None = None):
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
    def user_input_formatted(self):
        return Interaction.format_user_input(self._user_input)

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
    repetition_penalty: float = 1.15
    presence_penalty: float = 0
    frequency_penalty: float = 0
    guidance_scale: float = 1
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    smoothing_factor: float = 0
    stop: list[str] = field(default_factory=list)


class LLMClient:

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._completion_url = urljoin(base_url, "/v1/completions")
        self._token_url = urljoin(base_url, "v1/internal/token-count")

    def request(self, url: str, payload):
        """
        Sends a POST request to the specified URL with the given payload.

        Args:
            url (str): The URL to send the request to.
            payload (dict): The payload to include in the request body.

        Returns:
            dict: The JSON response from the request, parsed as a dictionary.

        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return json.loads(response.text)

    def completion_stream(self, prompt: str, llm_config: LLMConfig = LLMConfig()):
        headers = {"Content-Type": "application/json"}

        data = asdict(llm_config)
        data["prompt"] = prompt
        data["stream"] = True
        data["stop"] = ["USER", "User", "###"]

        stream_response = requests.post(
            self._completion_url,
            headers=headers,
            json=data,
            verify=False,
            stream=True,
            timeout=10,
        )
        client = SSEClient(stream_response)  # type: ignore

        for event in client.events():
            payload = json.loads(event.data)
            yield payload["choices"][0]["text"]

    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()):
        headers = {"Content-Type": "application/json"}

        data = asdict(llm_config)
        data["prompt"] = prompt
        data["stream"] = False

        response = requests.post(
            self._completion_url,
            headers=headers,
            json=data,
            verify=False,
            stream=False,
            timeout=10,
        )
        return json.loads(response.text)["choices"][0]["text"]

    def count_tokens(self, text: str):
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        payload = {"text": text}
        result = self.request(self._token_url, payload)
        return int(result["length"])


class SummaryMemory:
    """
    A class representing the history of interactions in a chat conversation.

    Attributes:
        _last_k (int): The last k interactions are always sent unsummarized
        _history (List[Interaction]): The list of interactions in the chat conversation.
        _summary (str): The summary of the chat conversation.
        _n_summarized (int): The number of interactions that have been summarized.
    """

    @dataclass
    class SummaryInteractions:
        count: int
        text: str

    def __init__(
        self,
        llm_client: LLMClient,
        last_k: int,
        history: list[Interaction] | None = None,
    ):
        self._llm_client = llm_client
        self._last_k = last_k
        self._history: list[Interaction] = history if history is not None else []
        self._summary = ""
        self._n_summarized = 0

    def __len__(self):
        return len(self._history)

    @property
    def summary(self):
        """
        Returns the summary of the chat conversation.

        Returns:
            str: The summary of the chat conversation.
        """
        return self._summary

    def summarize(self, text_interaction):

        print("SUMMARIZE")

        summary = f"Current summary:\n{self.summary}" if len(self.summary) > 0 else ""

        prompt = SUMMARY_TEMPLATE.format(
            current_summary=summary,
            unsummarized_interactions=text_interaction,
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
            USER_PREFIX=Actor.USER.value,
            LLM_PREFIX=Actor.LLM.value,
        )

        print(prompt)

        new_summary = self._llm_client.completion(prompt)

        print("--- Summary")
        print("New summary: ", new_summary)

        return new_summary

    def _try_summarize(self):
        interaction_candidates = self._history[self._n_summarized : -self._last_k]
        count = len(interaction_candidates)
        text = "\n".join(
            [interaction.format_interaction() for interaction in interaction_candidates]
        )

        if self._llm_client.count_tokens(text) > 150:
            self._summary = self.summarize(text)
            self._n_summarized += count

    def append(self, intercation: Interaction):
        """
        Appends a new interaction to the chat conversation history.

        Args:
            interaction (Interaction): The interaction to be appended to the history.
        """
        self._history.append(intercation)
        self._try_summarize()

    def interactions_complete(self):
        """
        Returns the complete history of interactions in the chat conversation.

        Returns:
            List[Interaction]: The complete history of interactions.
        """
        return self._history

    def interactions_summarized(self):
        """
        Returns a list of interactions that have already been summarized in the chat conversation.

        Returns:
            List[Interaction]: The list of interactions that have been summarized.
        """
        return self._history[: self._n_summarized]

    def interactions_unsummarized(self):
        """
        Returns the current interactions (that have not been summarized yet) in the chat conversation.

        Returns:
            List[Interaction]: The list of current interactions.
        """
        return self._history[self._n_summarized :]

    def text_interactions_unsummarized(self):
        """
        Returns the formatted text of the current interactions in the chat conversation.

        Returns:
            str: The formatted text of the current interactions.
        """
        return "\n".join(
            [
                interaction.format_interaction()
                for interaction in self.interactions_unsummarized()
            ]
        )

    def text_interactions_complete(self):
        """
        Returns the formatted text of all interactions in the chat conversation.

        Returns:
            str: The formatted text of all interactions.
        """
        return "\n".join(
            [
                interaction.format_interaction()
                for interaction in self.interactions_complete()
            ]
        )


class SummaryChat:
    """
    A class representing a chat conversation with summarization capabilities.

    Attributes:
        _completion_url (str): The URL for the completion API endpoint.
        _token_url (str): The URL for the token count API endpoint.
        _memory (SummaryMemory): The history of interactions in the chat conversation.
    """

    def __init__(
        self,
        base_url: str,
        role: str,
        last_k: int = 2,
        history: list[Interaction] | None = None,
    ):
        self._llm_client = LLMClient(base_url=base_url)
        self._role = role
        self._memory = SummaryMemory(self._llm_client, last_k, history)

    def regenerate(self, last_interaction: Interaction):
        """
        Regenerates the llm output of the last interaction
        This is triggered on pushing Regenerate of PlayerPrev.
        This will create a new Gamemaster output

        Args:
            question (str): The question to be asked to the AI language model.
        """
        history = self._memory.interactions_unsummarized()

        prompt = CHAT_TEMPLATE.format(
            role=self._role,
            summary=self._memory.summary,
            history=self._memory.text_interactions_unsummarized(),
            current_user_input=last_interaction.user_input_formatted,
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
            USER_PREFIX=Actor.USER.value,
            LLM_PREFIX=Actor.LLM.value,
        )

        print("--- Prompt:")
        print(prompt)

        llm_response = ""
        for chunk in self._llm_client.completion_stream(prompt):
            llm_response += chunk
            yield chunk

        print()

    def predict(self, user_input: str, last_interaction: Interaction | None):
        """
        Predicts the AI language model's response to a given question in the chat conversation.
        Predict is called on Sending of a new user input. The previous interaction will then be
        persisted in the memory.

        Predict will be calls when the Send button for Player is pushed. It takes into account
        the date in the Gamemaster field of the UI which will be sent with last_interaction.

        Args:
            question (str): The question to be asked to the AI language model.
        """
        history = self._memory.text_interactions_unsummarized()
        if last_interaction is not None:
            history += last_interaction.format_interaction()

        print("Current Summary:")
        print(self._memory.summary)

        prompt = CHAT_TEMPLATE.format(
            role=self._role,
            summary=self._memory.summary,
            history=history,
            current_user_input=Interaction.format_user_input(user_input),
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
            USER_PREFIX=Actor.USER.value,
            LLM_PREFIX=Actor.LLM.value,
        )

        print("--- Prompt:")
        print(prompt)

        llm_response = ""
        for chunk in self._llm_client.completion_stream(prompt):
            llm_response += chunk
            yield chunk

        print()

        if last_interaction is not None:
            last_interaction._id = str(len(self._memory))
            self._memory.append(last_interaction)

        print("--- History:")
        print(self._memory.text_interactions_complete())
