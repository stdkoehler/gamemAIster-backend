"""Chat Conversation Memory"""

import re
import json
from concurrent.futures import ThreadPoolExecutor
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
from src.brain.character_formatters import to_npc_summary, to_party_summary
from src.brain.system_registry import NPC_CONFIGS

from src.brain.data_types import Interaction, EntityResponse, Scene
from src.brain.json_tools import extract_json_schema

from src.utils.sqllogger import SQLLogger, LogType
from src.utils.logger import configure_logger

_log = configure_logger("chat")
logger = SQLLogger()

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


@dataclass(frozen=True)
class _SummaryState:
    """
    Snapshot of all summarization-derived state.
    Written as a single reference swap so readers always see a consistent
    picture — never a partially-updated mix of old summary + new entities.
    """

    summary: str
    n_summarized: int
    entities: list
    scenes: list


class SummaryMemory:
    """
    A class representing the history of interactions in a chat conversation.

    Attributes:
        _last_k (int): The last k interactions are always sent unsummarized
        _history (List[Interaction]): The list of interactions in the chat conversation.
        _summary (str): The summary of the chat conversation.
        _n_summarized (int): The number of interactions that have been summarized.
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        summary_prompt: str,
        entity_prompt: str,
        scene_prompt: str,
        game_name: str,
        last_k: int,
        mission_id: int,
        min_summary_tokens: int = 2048,
    ):
        self._llm_client = llm_client
        self._summary_prompt = summary_prompt
        self._entity_prompt = entity_prompt
        self._scene_prompt = scene_prompt
        self._game_name = game_name
        self._last_k = last_k
        self._mission_id = mission_id
        self._min_summary_tokens = min_summary_tokens

        self._summarize_lock = threading.Lock()
        summary, n_summarized = crud_instance.get_summary(self._mission_id)
        self._state = _SummaryState(
            summary=summary,
            n_summarized=n_summarized,
            entities=crud_instance.get_entities(self._mission_id),
            scenes=crud_instance.get_scenes(self._mission_id),
        )
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
        return self._state.summary

    # ------------------------------------------------------------------
    # Shared LLM call + parse + log pattern
    # ------------------------------------------------------------------

    def _call_structured(
        self,
        messages: list[Message],
        parse: Callable[[str], T],
        log_type: LogType,
        strip_think: bool = False,
    ) -> T:
        """
        Call the LLM, extract JSON from the response, validate it with `parse`,
        and log the outcome. Raises ValueError on any parse failure.
        `strip_think` removes <think>…</think> blocks before JSON extraction —
        needed when a model embeds visible thinking before the JSON output.

        Use this for fire-and-forget background tasks (summarization) where
        logging parse failures matters more than retry resilience.
        For synchronous blocking calls that should retry on failure, use
        parse_with_retry from structured_output instead.
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
            logger.log_llm_call(log_type, llm_input=log_prompt, llm_output=response, processed_output="JSON Parsing Error")
            raise ValueError(
                f"LLM response does not contain valid JSON:\n{text}"
            ) from exc

        try:
            result = parse(json_str)
        except (json.decoder.JSONDecodeError, ValidationError, KeyError) as exc:
            logger.log_llm_call(log_type, llm_input=log_prompt, llm_output=response, extracted_json=json_str, processed_output="Validation Error")
            raise ValueError(
                "LLM response is not valid JSON or doesn't validate as pydantic model"
            ) from exc

        logger.log_llm_call(log_type, llm_input=log_prompt, llm_output=response, extracted_json=json_str, processed_output=json_str)
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
                    text=self._scene_prompt.replace("__RPG__", self._game_name)
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
            log_type=LogType.SCENE,
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
                    text=self._entity_prompt.replace("__RPG__", self._game_name)
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
            log_type=LogType.ENTITY,
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
                content=MessageContent(text=self._summary_prompt),
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
            log_type=LogType.SUMMARY,
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
        # Non-blocking: if a summarization is already running, skip rather than
        # queue a redundant one — the next append() will catch up.
        if not self._summarize_lock.acquire(blocking=False):
            return
        try:
            eligible_history = self._history[self._state.n_summarized : -self._last_k]
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

            if current_tokens > self._min_summary_tokens:
                _log.info("Summary | processing %d interactions | n_summarized→%d", n, self._state.n_summarized + n)

                text = re.sub(r"---\s*What do you do\?\s*", "", text)

                # All three calls take the same input and are independent — run in parallel
                with ThreadPoolExecutor(max_workers=3) as executor:
                    f_entities = executor.submit(self.extract_entities, text)
                    f_scenes = executor.submit(self.scene_summary, text)
                    f_summary = executor.submit(self.summarize, text)
                entity_response = f_entities.result()
                scene_response = f_scenes.result()
                new_summary = f_summary.result()
                new_n_summarized = self._state.n_summarized + len(interaction_candidates)

                crud_instance.update_summary(
                    self._mission_id, new_summary, new_n_summarized
                )
                crud_instance.update_entities(self._mission_id, entity_response)
                crud_instance.update_scenes(self._mission_id, scene_response)

                # Single reference swap — readers see either the old or the new
                # state in full, never a partially-updated mix.
                self._state = _SummaryState(
                    summary=new_summary,
                    n_summarized=new_n_summarized,
                    entities=crud_instance.get_entities(self._mission_id),
                    scenes=crud_instance.get_scenes(self._mission_id),
                )
            else:
                _log.info("Summary | skip | %d interactions | %d tokens < %d threshold", n, current_tokens, self._min_summary_tokens)
        finally:
            self._summarize_lock.release()

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
        return self._history[self._state.n_summarized :]

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
        return self._state.n_summarized

    def get_summary(self) -> str:
        """
        Returns the current summary of the chat conversation.

        Returns:
            str: The current summary of the chat conversation.
        """
        return self._state.summary

    def get_entities_json(self) -> str:
        """
        Returns the current entities in JSON format.

        Returns:
            str: The current entities in JSON format.
        """
        return json.dumps([entity.model_dump() for entity in self._state.entities])

    def get_scenes_json(self) -> str:
        """
        Returns the current scenes in JSON format.

        Returns:
            str: The current scenes in JSON format.
        """
        return json.dumps(
            [
                scene.model_dump(exclude={"characters", "completed"})
                for scene in self._state.scenes
            ]
        )


def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(re.sub(r"[^\w\s]", "", name.lower()).split())


def _names_could_be_same(a: frozenset[str], b: frozenset[str]) -> bool:
    """True if one name's tokens are a subset of the other's, e.g. "Vorath"
    or "Inquisitor" against "Inquisitor Vorath Bloodstone"."""
    return bool(a) and bool(b) and (a <= b or b <= a)


def _flag_active_key_npcs(
    mission_json: str, game_type, active_npcs: list[tuple[str, str | None]]
) -> str:
    """Annotates `keyNPCs` entries (or the system's equivalent roster field)
    that correspond to a currently-active NPC, so the model isn't handed two
    silently conflicting descriptions of the same character — the adventure's
    static roster entry, and the live, scene-accurate Active NPCs entry.

    `active_npcs` is a list of `(name, matched_key_npc)` pairs, one per
    active NPC. `matched_key_npc` is the roster entry name the NPC Profiler
    matched at generation time (see `Gamemaster.generate_npc` /
    `_validate_matched_key_npc`), already confirmed against this same
    roster — so it's used as a direct, trusted lookup with no fuzziness.

    Active NPCs with no recorded match (manually-created sheets, or ones
    generated before this field existed) fall back to a token-subset name
    match (a `keyNPCs` entry named "Vorath" or "Inquisitor" matches an
    active NPC named "Inquisitor Vorath Bloodstone", and vice versa). A
    `keyNPCs` entry or active NPC name that partially matches more than one
    NPC on the other side is inherently ambiguous (e.g. two active
    "Inquisitor"s, or a roster with both "Vorath" and "Bloodstone" as
    distinct characters) — those are left unflagged rather than guessed at.

    Systems without an `NPC_CONFIGS` entry (e.g. Custom) or without a roster
    field present are left untouched.
    """
    if not active_npcs:
        return mission_json
    npc_cfg = NPC_CONFIGS.get(game_type)
    if npc_cfg is None:
        return mission_json
    try:
        parsed = json.loads(mission_json)
    except json.JSONDecodeError:
        return mission_json
    if not isinstance(parsed, dict):
        return mission_json
    roster = parsed.get(npc_cfg.npc_roster_key)
    if not isinstance(roster, list):
        return mission_json

    # Group by matched_key_npc first so two active NPCs claiming the same
    # roster entry (e.g. the same character generated twice under different
    # sheet names) are dropped as ambiguous rather than letting one silently
    # overwrite the other.
    trusted_groups: dict[str, list[str]] = {}
    for name, matched in active_npcs:
        if matched:
            trusted_groups.setdefault(matched.strip().lower(), []).append(name)
    trusted_matches = {
        matched: names[0] for matched, names in trusted_groups.items() if len(names) == 1
    }
    fallback_tokens = {
        name: _name_tokens(name) for name, matched in active_npcs if not matched
    }

    entry_matches: dict[int, str] = {}
    for idx, entry in enumerate(roster):
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("name", "")).strip().lower()
        if not entry_name:
            continue
        if entry_name in trusted_matches:
            entry_matches[idx] = trusted_matches[entry_name]
            continue
        entry_tok = _name_tokens(entry_name)
        candidates = [
            name for name, tok in fallback_tokens.items() if _names_could_be_same(entry_tok, tok)
        ]
        if len(candidates) == 1:
            entry_matches[idx] = candidates[0]
        # 0 or >1 candidates: ambiguous or no match, leave unflagged.

    # An active NPC resolving to more than one keyNPCs entry is itself
    # ambiguous (e.g. the roster lists both "Vorath" and "Bloodstone" as
    # distinct characters) — drop those rather than guess which is right.
    name_use_count: dict[str, int] = {}
    for name in entry_matches.values():
        name_use_count[name] = name_use_count.get(name, 0) + 1
    entry_matches = {idx: name for idx, name in entry_matches.items() if name_use_count[name] == 1}

    if not entry_matches:
        return mission_json

    for idx in entry_matches:
        roster[idx]["_activeNpcNote"] = (
            "This NPC is currently active in the scene. Treat the "
            "matching entry in Active NPCs as authoritative for their "
            "current description, status, and equipment — this entry is "
            "background/history only."
        )

    return json.dumps(parsed, ensure_ascii=False, indent=2)


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
        summary_prompt: str,
        entity_prompt: str,
        scene_prompt: str,
        summary_provider_prompt: str,
        game_name: str,
        mission_id: int,
        last_k: int = 2,
        min_summary_tokens: int = 2048,
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
            (s.name.strip().lower(), s.matched_key_npc)
            for s in sheets
            if s.is_npc and s.is_active
        ]
        self._mission = _flag_active_key_npcs(mission.description, mission.game_type, active_npcs)
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
            summary_prompt=summary_prompt,
            entity_prompt=entity_prompt,
            scene_prompt=scene_prompt,
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

        _log.debug("Interaction | %d messages | last=%s", len(messages), messages[-1].role if messages else "-")
