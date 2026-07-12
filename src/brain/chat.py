"""Chat turn orchestration: builds the story system prompt, streams the LLM
response, and owns the conversation's history via SummaryMemory.

Compression internals (entities/scenes/summary ledger+digest) live in
src/brain/conversation_memory.py. NPC roster cross-referencing for the
mission JSON lives in src/brain/npc_context.py.
"""

from typing import Generator

from src.llmclient.llm_parameters import LLMConfig, LLMLogicConfig
from src.llmclient.llm_config_registry import LLMTask
from src.llmclient.llm_client import (
    LLMClientBase,
    StreamType,
    MessageRole,
    MessageContent,
    Message,
)
from src.crud.crud import crud_instance
from src.brain.character_formatters import to_npc_summary, to_party_summary
from src.brain.conversation_memory import SummaryMemory, CompressionPrompts
from src.brain.npc_context import get_npc_roster, flag_active_key_npcs

from src.brain.data_types import Interaction

from src.utils.logger import configure_logger

_log = configure_logger("chat")


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
        story_prompt: str,
        prompts: CompressionPrompts,
        summary_provider_prompt: str,
        game_name: str,
        mission_id: int,
        logic_config: LLMLogicConfig = LLMLogicConfig.defaults(),
    ):
        self._llm_client_chat = llm_client_chat
        self._story_prompt = story_prompt
        mission = crud_instance.get_mission_description(mission_id=mission_id)
        if mission is None:
            raise ValueError("No mission could be loaded from database.")
        self._background = mission.background
        self._detailed_background = mission.detailed_background
        sheets = crud_instance.get_character_sheets(mission_id=mission_id)
        active_npcs = [
            (s.name, s.matched_key_npc)
            for s in sheets
            if s.is_npc and s.is_active and s.matched_key_npc
        ]
        self._mission = flag_active_key_npcs(mission.description, mission.game_type, active_npcs)
        self._character_summary = to_party_summary(
            mission.game_type.value,
            [
                {"content": s.content, "is_protagonist": s.is_protagonist}
                for s in sheets
                if not s.is_npc
            ],
        )
        self._npc_summary = to_npc_summary(
            mission.game_type.value,
            [{"content": s.content} for s in sheets if s.is_npc and s.is_active],
        )
        self._summary_provider_prompt = summary_provider_prompt
        self._memory = SummaryMemory(
            llm_client=llm_client_reasoning,
            prompts=prompts,
            game_name=game_name,
            mission_id=mission_id,
            npc_roster=get_npc_roster(mission.description, mission.game_type),
            logic_config=logic_config,
        )

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

        system_prompt = self._story_prompt.format(
            MISSION=self._mission,
            BACKGROUND=self._background,
            PARTY=self._character_summary,
            NPCS=self._npc_summary,
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
                        text=self._summary_provider_prompt.format(
                            SUMMARY=self._memory.injected_summary,
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

        _log.debug("Interaction | %d messages | last=%s", len(messages), messages[-1].role if messages else "-")
