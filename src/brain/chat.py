"""Chat Conversation Memory"""

import re
import json
from dataclasses import dataclass
from typing import Generator
import threading

from pydantic import ValidationError


from src.llmclient.llm_parameters import LLMConfig
from src.llmclient.llm_client import LLMClientBase
from src.crud.crud import crud_instance

from src.brain.data_types import Interaction, EntityResponse, Scene
from src.brain.json_tools import extract_json_schema

from src.utils.sqllogger import SQLLogger

logger = SQLLogger()  # Defaults to SQLite in current directory

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
        llm_client: LLMClientBase,
        summary_template: str,
        entity_template: str,
        scene_template: str,
        game_name: str,
        last_k: int,
        mission_id: int,
        min_summary_tokens: int = 1024,
    ):
        self._llm_client = llm_client
        self._summary_template = summary_template
        self._entity_template = entity_template
        self._scene_template = scene_template
        self._game_name = game_name
        self._last_k = last_k
        self._n_summarized = 0
        self._mission_id = mission_id
        self._min_summary_tokens = min_summary_tokens

        self._summary, self._n_summarized = crud_instance.get_summary(self._mission_id)
        self._entities = crud_instance.get_entities(self._mission_id)
        self._scenes = crud_instance.get_scenes(self._mission_id)
        self._history = crud_instance.get_interactions(self._mission_id)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def summary(self) -> str:
        """
        Returns the summary of the chat conversation.

        Returns:
            str: The summary of the chat conversation.
        """
        return self._summary

    def scene_summary(self, text_interaction: str) -> list[Scene]:
        """
        Returns the scene summary of the chat conversation.

        Returns:
            str: The scene summary of the chat conversation.
        """
        scenes = crud_instance.get_scenes(self._mission_id)
        last_scene_id = max([scene.id for scene in scenes], default=0)
        scenes_json = json.dumps([scene.model_dump() for scene in scenes])

        scene_input = '**Input:**\n```json\n{{"previous_scenes": {scenes},"current_history": {text}}}\n```'
        messages = [
            {
                "role": "system",
                "content": self._scene_template.replace("__RPG__", self._game_name),
            },
            {
                "role": "user",
                "content": scene_input.format(
                    scenes=scenes_json, text=text_interaction
                ),
            },
        ]

        ### Scene Prompt
        log_prompt = "\n\n".join(msg["content"] for msg in messages)

        llm_config = LLMConfig()
        llm_config.temperature = 0.7
        llm_config.max_tokens = 8192

        response = self._llm_client.chat_completion(
            messages=messages, reasoning=True, llm_config=llm_config
        )

        # remove content between <think>  tags
        response_wo_think = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        json_string = extract_json_schema(response_wo_think)

        try:
            data = json.loads(json_string)
            scene_response: list[Scene] = [
                Scene.model_validate(scene) for scene in data["scenes"]
            ]
            # only update last scene and newly created scenes
            scene_response = [
                scene for scene in scene_response if scene.id >= last_scene_id
            ]
        except (json.decoder.JSONDecodeError, ValidationError) as exc:
            logger.log_scene(llm_input=log_prompt, raw_output=response)
            raise ValueError(
                f"LLM response is not valid JSON or doesn't validate as pydantic model"
            ) from exc
        except KeyError as exc:
            raise ValueError(
                f"LLM response is missing required keys: {json_string}"
            ) from exc

        logger.log_scene(
            llm_input=log_prompt,
            raw_output=response,
            processed_output=json_string,
        )

        return scene_response

    def extract_entities(self, text_interaction: str) -> EntityResponse:

        entities_json = json.dumps(
            [
                entity.model_dump()
                for entity in crud_instance.get_entities(self._mission_id)
            ]
        )

        entity_input = 'Extract entities from the following text and update the given entities:\n{{"text": {text},"entities": {entities}}}'
        messages = [
            {
                "role": "system",
                "content": self._entity_template.replace("__RPG__", self._game_name),
            },
            {
                "role": "user",
                "content": entity_input.format(
                    text=text_interaction, entities=entities_json
                ),
            },
        ]

        ### Entity Prompt
        log_prompt = "\n\n".join(msg["content"] for msg in messages)

        response = self._llm_client.chat_completion(messages=messages, reasoning=True)

        json_string = extract_json_schema(response)

        try:
            entity_response = EntityResponse.model_validate_json(json_string)
        except (json.decoder.JSONDecodeError, ValidationError) as exc:
            logger.log_entity(llm_input=log_prompt, raw_output=response)
            raise ValueError(
                f"LLM response is not valid JSON or doesn't validate as pydantic model"
            ) from exc
        except KeyError as exc:
            raise ValueError(
                f"LLM response is missing required keys: {json_string}"
            ) from exc

        logger.log_entity(
            llm_input=log_prompt,
            raw_output=response,
            processed_output=json_string,
        )

        return entity_response

    def summarize(self, text_interaction: str) -> str:
        """
        Summarize the current summary plus the new text_interactions
        """

        summary = self.summary if len(self.summary) > 0 else ""

        summary_input = 'Summarize the following text:\n{{"previous_summary": {prev}, "current_events": {current}}}'
        messages = [
            {
                "role": "system",
                "content": self._summary_template,
            },
            {
                "role": "user",
                "content": summary_input.format(prev=summary, current=text_interaction),
            },
        ]

        ### Summary Prompt
        log_prompt = "\n\n".join(msg["content"] for msg in messages)

        response = self._llm_client.chat_completion(messages=messages, reasoning=True)

        json_string = extract_json_schema(response)

        try:
            summary_obj: dict[str, str] = json.loads(json_string)
            new_summary = summary_obj["summary"]
        except (json.decoder.JSONDecodeError, ValidationError) as exc:
            logger.log_summary(llm_input=log_prompt, raw_output=response)
            raise ValueError(
                f"LLM response is not valid JSON or doesn't validate as pydantic model"
            ) from exc
        except KeyError as exc:
            raise ValueError(
                f"LLM response is missing required keys: {json_string}"
            ) from exc

        logger.log_summary(
            llm_input=log_prompt,
            raw_output=response,
            processed_output=json_string,
        )

        return new_summary

    def _try_summarize(self) -> None:
        """summarize"""
        interaction_candidates = self._history[self._n_summarized : -self._last_k]
        text = "\n".join(
            [
                interaction.format_interaction_summary()
                for interaction in interaction_candidates
            ]
        )
        if self._llm_client.count_tokens(text) > self._min_summary_tokens:
            entity_response = self.extract_entities(text)
            scene_response = self.scene_summary(text)
            self._summary = self.summarize(text)

            self._n_summarized += len(interaction_candidates)
            crud_instance.update_summary(
                self._mission_id, self._summary, self._n_summarized
            )
            crud_instance.update_entities(self._mission_id, entity_response)
            crud_instance.update_scenes(self._mission_id, scene_response)
            self._entities = crud_instance.get_entities(self._mission_id)
            self._scenes = crud_instance.get_scenes(self._mission_id)

    def append(self, interaction: Interaction) -> None:
        """
        Appends a new interaction to the chat conversation history.

        Args:
            interaction (Interaction): The interaction to be appended to the history.
        """
        crud_instance.insert_interaction(self._mission_id, interaction)
        threading.Thread(target=self._try_summarize, daemon=True).start()

    def update_last(self, interaction: Interaction) -> None:
        """
        Appdates the last interaction in memory.

        Args:
            interaction (Interaction): The interaction that overwrites the last interaction
        """
        crud_instance.update_last_interaction(self._mission_id, interaction)
        self._history[-1] = interaction

        # self._try_summarize()

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

    def text_interactions_complete(self) -> str:
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

    def chat(self) -> list[dict[str, str]]:
        messages = []
        for interaction in self._history:
            messages.append({"role": "user", "content": interaction.user_input})
            messages.append({"role": "assistant", "content": interaction.llm_output})
        return messages

    def chat_unsummarized(self) -> list[dict[str, str]]:
        messages = []
        for interaction in self.interactions_unsummarized():
            messages.append({"role": "user", "content": interaction.user_input})
            messages.append({"role": "assistant", "content": interaction.llm_output})
        return messages

    def get_summary(self) -> str:
        """
        Returns the current summary of the chat conversation.

        Returns:
            str: The current summary of the chat conversation.
        """
        return self._summary

    def get_entities_json(self) -> str:
        """
        Returns the current entities in JSON format.

        Returns:
            str: The current entities in JSON format.
        """
        return json.dumps([entity.model_dump() for entity in self._entities])

    def get_scenes_json(self) -> str:
        """
        Returns the current scenes in JSON format.

        Returns:
            str: The current scenes in JSON format.
        """
        return json.dumps(
            [
                scene.model_dump(exclude={"characters", "completed"})
                for scene in self._scenes
            ]
        )

    @property
    def n_summarized(self) -> int:
        """
        Returns the number of interactions that have been summarized.

        Returns:
            int: The number of interactions that have been summarized.
        """
        return self._n_summarized


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
        llm_client_reasoning: LLMClientBase,
        llm_client_chat: LLMClientBase,
        role: str,
        summary_template: str,
        entity_template: str,
        scene_template: str,
        summary_provider_template: str,
        game_name: str,
        mission_id: int,
        last_k: int = 2,
        min_summary_tokens: int = 1024,
    ):
        self._llm_client_chat = llm_client_chat
        self._role = role
        mission = crud_instance.get_mission_description(mission_id=mission_id)
        if mission is None:
            raise ValueError("No mission could be loaded from database.")
        self._mission = mission.description
        self._background = mission.background
        self._summary_provider_template = summary_provider_template
        self._memory = SummaryMemory(
            llm_client=llm_client_reasoning,
            summary_template=summary_template,
            entity_template=entity_template,
            scene_template=scene_template,
            game_name=game_name,
            last_k=last_k,
            min_summary_tokens=min_summary_tokens,
            mission_id=mission_id,
        )

    @staticmethod
    def _trim_chunk(chunk: str) -> str:
        chunk = re.sub(strip_pattern, "", chunk)
        chunk = chunk.lstrip()
        return chunk

    def predict(
        self, user_input: str | None, last_interaction: Interaction | None = None
    ) -> Generator[str, None, None]:
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

        # update last interaction - player changed interaction in frontend
        is_regenerate = True if user_input is None else False

        if last_interaction is not None:
            self._memory.update_last(last_interaction)

        # regenerate with previous input
        if user_input is None:
            _, user_input = self._memory.text_interactions_unsummarized_regenerate()

        # print("Current Summary:")
        # print(self._memory.summary)
        system_prompt = self._role.format(
            MISSION=self._mission, BACKGROUND=self._background
        )
        messages = [{"role": "system", "content": system_prompt}]

        if self._memory.n_summarized > 0:

            # we have summarized interactions, we add summary, entities and the
            # unsummarized interactions
            messages.append(
                {
                    "role": "user",
                    "content": self._summary_provider_template.format(
                        SUMMARY=self._memory.summary,
                        SCENES=self._memory.get_scenes_json(),
                        ENTITIES=self._memory.get_entities_json(),
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "[OOC: Thank you for the summary and entities. I will use them to continue the story.]",
                }
            )

            messages += self._memory.chat_unsummarized()
        else:
            messages += self._memory.chat()

        if is_regenerate:
            # we want to regenerate the last LLM answer, delete it from messages
            messages = messages[:-1]
        else:
            messages.append({"role": "user", "content": user_input})

        llm_config = LLMConfig()
        llm_config.stop = ["PL", "###", "/FIN"]

        llm_response = ""
        begun = False
        for chunk in self._llm_client_chat.chat_completion_stream(
            messages, llm_config=llm_config
        ):
            if not begun:
                chunk = self._trim_chunk(chunk)
                if chunk != "":
                    begun = True
            llm_response += chunk
            yield chunk

        # pattern = (
        #     r"(?:What\ do\ you\ want\ to\ |What\ would\ you\ like\ to\ )\S[\S\s]*\?\s*"
        # )
        # llm_response = re.sub(pattern, "", llm_response)

        interaction = Interaction(user_input=user_input, llm_output=llm_response)

        if is_regenerate:
            self._memory.update_last(interaction)
        else:
            self._memory.append(interaction)

        print("### Prompt")
        for mi in messages:
            print(mi)
            print("-------------")
