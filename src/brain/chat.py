"""Chat Conversation Memory"""

import re
import json
from dataclasses import dataclass
from typing import Callable, Generator, TypeVar
import threading

from pydantic import BaseModel, ValidationError


from src.llmclient.llm_parameters import LLMConfig
from src.llmclient.llm_config_registry import LLMTask
from src.llmclient.llm_client import (
    LLMClientBase,
    StreamType,
    MessageRole,
    MessageContent,
    Message,
)
from src.crud.crud import crud_instance

from src.brain.data_types import Interaction, EntityResponse, Scene
from src.brain.json_tools import extract_json_schema

from src.utils.sqllogger import SQLLogger

logger = SQLLogger()  # Defaults to SQLite in current directory

# strip beginning linebreaks, spaces, GM, :
strip_pattern = re.compile(r"^(?::|\n|\s)*(GM)?:?")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Private parse targets for structured LLM responses
# ---------------------------------------------------------------------------


class _Summary(BaseModel):
    summary: str


class _SceneList(BaseModel):
    scenes: list[Scene]


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
        min_summary_tokens: int = 2048,
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

    # ------------------------------------------------------------------
    # Shared LLM call + parse + log pattern
    # ------------------------------------------------------------------

    def _call_structured(
        self,
        messages: list[Message],
        parse: Callable[[str], T],
        log_fn: Callable[..., None],
        strip_think: bool = False,
    ) -> T:
        """
        Call the LLM, extract JSON from the response, validate it with `parse`,
        and log the outcome. Raises ValueError on any parse failure.
        `strip_think` removes <think>…</think> blocks before JSON extraction —
        needed when a model embeds visible thinking before the JSON output.
        """
        log_prompt = "\n\n".join(msg.content.text for msg in messages)
        response = self._llm_client.chat_completion(
            messages=messages,
            reasoning=True,
            config_override=LLMConfig(max_tokens=8192),
            task=LLMTask.SUMMARY,
        )
        text = (
            re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            if strip_think
            else response
        )

        try:
            json_str = extract_json_schema(text)
        except ValueError as exc:
            log_fn(
                llm_input=log_prompt,
                raw_output=response,
                extracted_json="",
                processed_output="JSON Parsing Error",
            )
            raise ValueError(
                f"LLM response does not contain valid JSON:\n{text}"
            ) from exc

        try:
            result = parse(json_str)
        except (json.decoder.JSONDecodeError, ValidationError, KeyError) as exc:
            log_fn(
                llm_input=log_prompt,
                raw_output=response,
                extracted_json=json_str,
                processed_output="Validation Error",
            )
            raise ValueError(
                "LLM response is not valid JSON or doesn't validate as pydantic model"
            ) from exc

        log_fn(
            llm_input=log_prompt,
            raw_output=response,
            extracted_json=json_str,
            processed_output=json_str,
        )
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Structured extraction methods
    # ------------------------------------------------------------------

    def scene_summary(self, text_interaction: str) -> list[Scene]:
        """
        Returns the scene summary of the chat conversation.

        Returns:
            str: The scene summary of the chat conversation.
        """
        scenes = crud_instance.get_scenes(self._mission_id)
        last_scene_id = max([scene.id for scene in scenes], default=0)
        scenes_json = json.dumps([scene.model_dump() for scene in scenes])

        scene_input = '**Input:**\n```json\n{{"previous_scenes": {scenes},"current_history": {text}}}\n\n**Output:**\n```'
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(
                    text=self._scene_template.replace("__RPG__", self._game_name)
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=scene_input.format(scenes=scenes_json, text=text_interaction)
                ),
            ),
        ]

        scene_list = self._call_structured(
            messages=messages,
            parse=lambda s: _SceneList.model_validate_json(s).scenes,
            log_fn=logger.log_scene,
            strip_think=True,
        )
        # only update last scene and newly created scenes
        return [scene for scene in scene_list if scene.id >= last_scene_id]

    def extract_entities(self, text_interaction: str) -> EntityResponse:

        entities_json = json.dumps(
            [
                entity.model_dump()
                for entity in crud_instance.get_entities(self._mission_id)
            ]
        )

        entity_input = 'Extract entities from the following text and update the given entities:\n{{"text": {text},"entities": {entities}}}'
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(
                    text=self._entity_template.replace("__RPG__", self._game_name)
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=entity_input.format(
                        text=text_interaction, entities=entities_json
                    )
                ),
            ),
        ]

        return self._call_structured(
            messages=messages,
            parse=EntityResponse.model_validate_json,
            log_fn=logger.log_entity,
        )

    def summarize(self, text_interaction: str) -> str:
        """
        Summarize the current summary plus the new text_interactions
        """

        summary = self.summary if len(self.summary) > 0 else ""

        summary_input = 'Summarize the following text:\n{{"previous_summary": {prev}, "current_events": {current}}}'
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(text=self._summary_template),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=summary_input.format(prev=summary, current=text_interaction)
                ),
            ),
        ]

        return self._call_structured(
            messages=messages,
            parse=lambda s: _Summary.model_validate_json(s).summary,
            log_fn=logger.log_summary,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _try_summarize(self) -> None:
        """
        Summarize the history using a content-aware window.
        We gather the minimum number of interactions required to satisfy
        _min_summary_tokens, ensuring we don't process too little (saving cost)
        or too much (saving input space).
        """
        eligible_history = self._history[self._n_summarized : -self._last_k]
        if not eligible_history:
            return

        def get_formatted_text(candidates: list[Interaction]) -> str:
            return "\n".join([c.format_interaction_summary() for c in candidates])

        # Find the smallest 'n' that satisfies the token threshold
        n = 0
        current_tokens = 0
        text = ""

        while n < len(eligible_history):
            n += 1
            interaction_candidates = eligible_history[:n]
            text = get_formatted_text(interaction_candidates)
            current_tokens = self._llm_client.count_tokens(text)

            # Stop as soon as we have enough content to justify the cost
            if current_tokens > self._min_summary_tokens:
                break

        # Final Gate: We only proceed if we actually met the threshold
        # (or if we reached the end of eligible history and want to force a summary)
        if current_tokens > self._min_summary_tokens:
            print(
                "Processing summary for",
                n,
                "interactions, leading to n_summarized =",
                self._n_summarized + n,
            )

            text = re.sub(r"---\s*What do you do\?\s*", "", text)
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
        else:
            print(
                "Skipping summary for",
                n,
                "interactions (only",
                current_tokens,
                "tokens / threshold of",
                self._min_summary_tokens,
                ").",
            )

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
        Updates the last interaction in memory. If llm_thinking and llm_thinking_signature
        are set, they will be updated as well, if they are None they will be kept as is.
        (Important to address the LLM api constraint that llm_thinking must not be changed)

        Args:
            interaction (Interaction): The interaction that overwrites the last interaction
        """
        crud_instance.update_last_interaction(self._mission_id, interaction)
        if interaction.llm_thinking is not None:
            self._history[-1] = interaction
        else:
            # The frontend my send a updated last interaction without thinking content,
            # in this case we want to keep the original thinking and signature to avoid
            # tampering with the thinking content
            last_interaction = crud_instance.get_interactions(self._mission_id)[-1]
            self._history[-1] = Interaction(
                user_input=interaction.user_input,
                llm_output=interaction.llm_output,
                llm_thinking=last_interaction.llm_thinking,
                llm_thinking_signature=last_interaction.llm_thinking_signature,
            )

        # self._try_summarize()

    # ------------------------------------------------------------------
    # Message building helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interactions_to_messages(interactions: list[Interaction]) -> list[Message]:
        messages = []
        for interaction in interactions:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=MessageContent(text=interaction.user_input),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=MessageContent(
                        text=interaction.llm_output,
                        thinking=interaction.llm_thinking,
                        thinking_signature=interaction.llm_thinking_signature,
                    ),
                )
            )
        return messages

    def interactions_unsummarized(self) -> list[Interaction]:
        """
        Returns the current interactions (that have not been summarized yet) in the chat conversation.

        Returns:
            List[Interaction]: The list of current interactions.
        """
        return self._history[self._n_summarized :]

    def last_user_input(self) -> str:
        """
        Returns the user input of the last interaction in the chat conversation.

        Returns:
            str: The user input of the last interaction.
        """
        return self._history[-1].user_input

    def chat(self) -> list[Message]:
        return self._interactions_to_messages(self._history)

    def chat_unsummarized(self) -> list[Message]:
        return self._interactions_to_messages(self.interactions_unsummarized())

    @property
    def n_summarized(self) -> int:
        """
        Returns the number of interactions that have been summarized.

        Returns:
            int: The number of interactions that have been summarized.
        """
        return self._n_summarized

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
        min_summary_tokens: int = 2048,
    ):
        self._llm_client_chat = llm_client_chat
        self._role = role
        mission = crud_instance.get_mission_description(mission_id=mission_id)
        if mission is None:
            raise ValueError("No mission could be loaded from database.")
        self._mission = mission.description
        self._background = mission.background
        self._detailed_background = mission.detailed_background
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

    def _build_messages(self, user_input: str, is_regenerate: bool) -> list[Message]:
        """
        Builds the messages to be sent to the LLM, including system prompt, summary,
        entities, scenes, and history.
        Args:
            user_input (str): The new user input to be added to the messages.
            is_regenerate (bool): Whether this is a regeneration (True) or a new turn (False).
        Returns:
            list[Message]: The list of messages to be sent to the LLM.
        """

        system_prompt = self._role.format(
            MISSION=self._mission, BACKGROUND=self._background
        )
        messages: list[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(text=system_prompt),
            )
        ]

        if self._detailed_background != "":
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=MessageContent(
                        text=f"[OOC: This detailed background is **AUTHORITATIVE** and I must follow and consult it throughout the game.\n\n<DETAILED_BACKGROUND>{self._detailed_background}</DETAILED_BACKGROUND>]"
                    ),
                )
            )

        if self._memory.n_summarized > 0:

            # we have summarized interactions, we add summary, entities and the
            # unsummarized interactions
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=MessageContent(
                        text=self._summary_provider_template.format(
                            SUMMARY=self._memory.summary,
                            SCENES=self._memory.get_scenes_json(),
                            ENTITIES=self._memory.get_entities_json(),
                        ),
                    ),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=MessageContent(
                        text="[OOC: Thank you for the summary and entities. I will use them to continue the story.]",
                    ),
                )
            )

            messages += self._memory.chat_unsummarized()
        else:
            messages += self._memory.chat()

        if is_regenerate:
            # we want to regenerate the last LLM answer, delete it from messages
            messages = messages[:-1]
        else:
            messages.append(
                Message(role=MessageRole.USER, content=MessageContent(text=user_input))
            )

        return messages

    def predict(
        self,
        user_input: str | None = None,
        last_interaction: Interaction | None = None,
    ) -> Generator[tuple[str, str], None, None]:
        """
        Orchestrates the LLM generation process, handling new turns, history corrections,
        and regenerations.

        Args:
            user_input (str | None): The new user input. If None, triggers regeneration
                with user input from last_interaction.
            last_interaction (Interaction | None): The previous interaction object. Used to update
                history before generation. This is always sent by the frontend because
                the frontend is the source of truth for the last interaction (e.g. if the
                user edited the previous LLM output or user input).
                Can be None if this is the first turn and there is no history yet.


        Yields:
            tuple[str, str]: Streamed chunks of the LLM's thinking process and final text response.
        """
        is_regenerate = False
        # always update last_interaction -> frontend is ground truth
        if last_interaction is not None:
            self._memory.update_last(last_interaction)
        if user_input is None:
            is_regenerate = True
            if last_interaction is None:
                raise ValueError(
                    "user_input is None but last_interaction is also None."
                )
            user_input = last_interaction.user_input

        messages = self._build_messages(
            user_input=user_input, is_regenerate=is_regenerate
        )

        llm_response = ""
        full_thinking = None
        signature = None
        for chunk in self._llm_client_chat.chat_completion_stream(
            messages,
            config_override=LLMConfig(stop=["PL", "###", "/FIN"]),
            reasoning=True,
            task=LLMTask.STORY,
        ):
            if chunk.type == StreamType.THINKING:
                yield ("thinking", chunk.delta)
            elif chunk.type == StreamType.TEXT:
                yield ("text", chunk.delta)
            elif chunk.type == StreamType.THINKING_END:
                full_thinking = chunk.full_thinking
                signature = chunk.signature
                yield ("thinking_end", chunk.delta)
            elif chunk.type == StreamType.TEXT_END:
                llm_response = chunk.full_text if chunk.full_text else ""
            else:
                raise ValueError(f"Unknown stream type: {chunk.type}")

        interaction = Interaction(
            user_input=user_input,
            llm_output=llm_response,
            llm_thinking=full_thinking,
            llm_thinking_signature=signature,
        )

        if is_regenerate:
            # we create an updated valid entry for interaction including llm_thinking
            # update_last will change llm_thinking and llm_thinking_signature when they
            # are set
            self._memory.update_last(interaction)
        else:
            self._memory.append(interaction)

        print("### Prompt")
        for mi in messages:
            print(mi)
            print("-------------")
