""" Chat Conversation Memory """

import json
import re
from dataclasses import dataclass, asdict

from urllib.parse import urljoin

import requests
from sseclient import SSEClient

from src.brain.types import Actor, Interaction, LLMConfig

from src.crud.crud import crud_instance

from src.brain.templates import CHAT_TEMPLATE_NOUS_CAPYBARA as CHAT_TEMPLATE
from src.brain.templates import SUMMARY_TEMPLATE_MIXTRAL_CHAT as SUMMARY_TEMPLATE


# strip beginning linebreaks, spaces, GM, :
strip_pattern = re.compile(r"^(?::|\n|\s)*(GM)?:?")


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
        data["stop"] = ["PL", "###"]
        data["sampler_priority"] = [
            "temperature",
            "dynamic_temperature",
            "quadratic_sampling",
            "top_k",
            "top_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "tfs",
            "top_a",
            "min_p",
            "mirostat",
        ]
        data["logits_processor"] = []

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
        session_name: str,
    ):
        self._llm_client = llm_client
        self._last_k = last_k
        self._summary = ""
        self._n_summarized = 0

        try:
            self._session_id = crud_instance.get_session_id(session_name)
        except ValueError:
            self._session_id = crud_instance.insert_session(session_name)

        self._history = crud_instance.get_interactions(self._session_id)

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

        summary = f"Current summary:\n{self.summary}" if len(self.summary) > 0 else ""

        prompt = SUMMARY_TEMPLATE.format(
            current_summary=summary,
            unsummarized_interactions=text_interaction,
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
            USER_PREFIX=Actor.USER.value,
            LLM_PREFIX=Actor.LLM.value,
        )

        new_summary = self._llm_client.completion(prompt)

        print("--- Summary")
        print("New summary: ", new_summary)

        return new_summary

    def _try_summarize(self):
        """summarize"""
        # interaction_candidates = self._history[self._n_summarized : -self._last_k]
        # count = len(interaction_candidates)
        # text = "\n".join(
        #     [interaction.format_interaction() for interaction in interaction_candidates]
        # )
        # if self._llm_client.count_tokens(text) > 150:
        #     self._summary = self.summarize(text)
        #     self._n_summarized += count

    def append(self, interaction: Interaction):
        """
        Appends a new interaction to the chat conversation history.

        Args:
            interaction (Interaction): The interaction to be appended to the history.
        """
        crud_instance.insert_interaction(self._session_id, interaction)

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
        session_name: str,
        last_k: int = 2,
    ):
        self._llm_client = LLMClient(base_url=base_url)
        self._role = role
        self._memory = SummaryMemory(self._llm_client, last_k, session_name)

    @staticmethod
    def _trim_chunk(chunk: str) -> str:
        print(chunk)
        chunk = re.sub(strip_pattern, "", chunk)
        chunk = chunk.lstrip()
        print(chunk)
        return chunk

    def predict(self, user_input: str, last_interaction: Interaction | None = None):
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

        # print("Current Summary:")
        # print(self._memory.summary)

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

        llm_response = ""
        begun = False
        for chunk in self._llm_client.completion_stream(prompt):
            if not begun:
                chunk = self._trim_chunk(chunk)
                if chunk != "":
                    begun = True
            llm_response += chunk
            yield chunk

        if last_interaction is not None:
            self._memory.append(last_interaction)

        print("--- History:")
        print(self._memory.text_interactions_complete())
