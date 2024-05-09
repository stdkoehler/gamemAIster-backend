""" Chat Conversation Memory """

import re
from dataclasses import dataclass

from src.llmclient.types import LLMConfig
from src.llmclient.llm_client import LLMClient
from src.crud.crud import crud_instance

from src.brain.types import Actor, Interaction
from src.brain.templates import CHAT_TEMPLATE_NOUS_HERMES as CHAT_TEMPLATE
from src.brain.templates import SUMMARY_TEMPLATE_NOUS_HERMES as SUMMARY_TEMPLATE
from src.brain.templates import ENTITY_TEMPLATE_NOUS_HERMES as ENTITY_TEMPLATE

# strip beginning linebreaks, spaces, GM, :
strip_pattern = re.compile(r"^(?::|\n|\s)*(GM)?:?")


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
        mission_id: int,
        min_summary_tokens: int = 1024,
    ):
        self._llm_client = llm_client
        self._last_k = last_k
        self._summary = ""
        self._n_summarized = 0
        self._mission_id = mission_id
        self._min_summary_tokens = min_summary_tokens

        self._history = crud_instance.get_interactions(self._mission_id)

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

    def extract_entities(self, text_interaction):

        prompt = ENTITY_TEMPLATE.format(
            unsummarized_interactions=text_interaction,
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
            USER_PREFIX=Actor.USER.value,
            LLM_PREFIX=Actor.LLM.value,
        )

        print("--- Extract entity prompt")
        print(prompt)

        entities = self._llm_client.completion(prompt)

        print("--- Extract entity")
        print("Entities: ", entities)

        return entities

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

        print("--- Summary prompt")
        print(prompt)

        new_summary = self._llm_client.completion(prompt)

        print("--- Summary")
        print("New summary: ", new_summary)

        return new_summary

    def _try_summarize(self):
        """summarize"""
        interaction_candidates = self._history[self._n_summarized : -self._last_k]
        count = len(interaction_candidates)
        text = "\n".join(
            [
                interaction.format_interaction_summary()
                for interaction in interaction_candidates
            ]
        )
        # if self._llm_client.count_tokens(text) > self._min_summary_tokens:
        if self._llm_client.count_tokens(text) > 256:
            self._summary = self.summarize(text)
            entities = self.extract_entities(text)
            # self._n_summarized += count

    def append(self, interaction: Interaction):
        """
        Appends a new interaction to the chat conversation history.

        Args:
            interaction (Interaction): The interaction to be appended to the history.
        """
        crud_instance.insert_interaction(self._mission_id, interaction)

        self._try_summarize()

    def update_last(self, interaction: Interaction):
        """
        Appdates the last interaction in memory.

        Args:
            interaction (Interaction): The interaction that overwrites the last interaction
        """
        crud_instance.update_last_interaction(self._mission_id, interaction)
        self._history[-1] = interaction

        # todo: debugging remove later
        self._try_summarize()

    def interactions_complete(self) -> list[Interaction]:
        """
        Returns the complete history of interactions in the chat conversation.

        Returns:
            List[Interaction]: The complete history of interactions.
        """
        return self._history

    def interactions_summarized(self) -> list[Interaction]:
        """
        Returns a list of interactions that have already been summarized in the chat conversation.

        Returns:
            List[Interaction]: The list of interactions that have been summarized.
        """
        return self._history[: self._n_summarized]

    def interactions_unsummarized(self) -> list[Interaction]:
        """
        Returns the current interactions (that have not been summarized yet) in the chat conversation.

        Returns:
            List[Interaction]: The list of current interactions.
        """
        return self._history[self._n_summarized :]

    def text_interactions_unsummarized(self) -> str:
        """
        Returns the formatted text of the current interactions in the chat conversation.

        Returns:
            str: The formatted text of the current interactions.
        """
        return "\n".join(
            interaction.format_interaction()
            for interaction in self.interactions_unsummarized()
        )

    def text_interactions_unsummarized_regenerate(self) -> tuple[str, str]:
        """
        Returns the formatted text of the current interactions in the chat conversation, excluding the last interaction.

        Returns:
            tuple[str, str]: A tuple containing two strings:
                - The formatted text of the current interactions, excluding the last interaction.
                - The user input of the last interaction.
        """
        if not self._history:
            return "", ""
        unsummarized_interactions = self.interactions_unsummarized()
        return (
            "\n".join(
                interaction.format_interaction()
                for interaction in unsummarized_interactions[:-1]
            ),
            self._history[-1].user_input,
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
        mission_id: int,
        last_k: int = 2,
    ):
        self._llm_client = LLMClient(base_url=base_url)
        self._role = role
        mission = crud_instance.get_mission_description(mission_id=mission_id)
        if mission is None:
            raise ValueError("No mission could be loaded from database.")
        self._mission = mission.description
        self._memory = SummaryMemory(self._llm_client, last_k, mission_id)

    @staticmethod
    def _trim_chunk(chunk: str) -> str:
        chunk = re.sub(strip_pattern, "", chunk)
        chunk = chunk.lstrip()
        return chunk

    def predict(
        self, user_input: str | None, last_interaction: Interaction | None = None
    ):
        """
        Predicts the AI language model's response to a given question in the chat conversation.
        Predict is called on Sending of a new user input. The previous interaction will then be
        persisted in the memory.

        Predict will be calls when the Send button for Player is pushed. It takes into account
        the data in the Gamemaster field of the UI which will be sent with last_interaction.

        Args:
            user_input (str | None): If it is None we either gegenerate the last interaction
            last_interaction (Interaction | None): The previous interaction. If it is not None, we will update the previous interaction
        """

        is_regenerate = True if user_input is None else False

        if last_interaction is not None:
            self._memory.update_last(last_interaction)

        if user_input is None:
            history, user_input = (
                self._memory.text_interactions_unsummarized_regenerate()
            )
        else:
            history = self._memory.text_interactions_unsummarized()

        # print("Current Summary:")
        # print(self._memory.summary)

        prompt = CHAT_TEMPLATE.format(
            role=self._role,
            mission=self._mission,
            summary=self._memory.summary,
            history=history,
            current_user_input=Interaction.format_user_input(user_input),
            SYSTEM_PREFIX=Actor.SYSTEM.value,
            SYSTEM_END=Actor.SYSTEM_END.value,
        )

        llm_config = LLMConfig()
        llm_config.stop = ["PL", "###", "/FIN"]

        llm_response = ""
        begun = False
        for chunk in self._llm_client.completion_stream(prompt, llm_config=llm_config):
            if not begun:
                chunk = self._trim_chunk(chunk)
                if chunk != "":
                    begun = True
            llm_response += chunk
            yield chunk

        pattern = (
            r"(?:What\ do\ you\ want\ to\ |What\ would\ you\ like\ to\ )\S[\S\s]*\?\s*"
        )
        llm_response = re.sub(pattern, "", llm_response)

        interaction = Interaction(user_input=user_input, llm_output=llm_response)

        if is_regenerate:
            self._memory.update_last(interaction)
        else:
            self._memory.append(interaction)

        print("--- History:")
        print(self._memory.text_interactions_complete())
