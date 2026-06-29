"""Conversation memory compression: entities, scenes, and the summary ledger/digest.

See docs/conversation_memory.html for the full design write-up (data flow,
diagrams, worked example). Update that doc in the same change whenever you
touch this file, per CLAUDE.md.
"""

import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from src.llmclient.llm_parameters import LLMConfig, LLMLogicConfig
from src.llmclient.llm_config_registry import LLMTask
from src.llmclient.llm_client import LLMClientBase, MessageRole, MessageContent, Message
from src.crud.crud import crud_instance

from src.brain.data_types import Interaction, EntityResponse, Scene
from src.brain.json_tools import extract_json_schema

from src.utils.sqllogger import SQLLogger, LogType
from src.utils.logger import configure_logger

_log = configure_logger("conversation_memory")
logger = SQLLogger()

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Private parse targets for structured LLM responses
# ---------------------------------------------------------------------------


class _Recap(BaseModel):
    recap: str


class _Digest(BaseModel):
    digest: str


class _SceneList(BaseModel):
    scenes: list[Scene]


def _last_paragraph(text: str) -> str:
    """The literal tail of the ledger — used as a cheap, bounded recency
    anchor (no LLM call) alongside the digest when grounding the next
    summarize() call. See docs/conversation_memory.html."""
    text = text.strip()
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs[-1] if paragraphs else text


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
    digest: str
    digest_ledger_tokens: int


@dataclass(frozen=True)
class CompressionPrompts:
    """The four prompt templates SummaryMemory's compression calls use,
    bundled so SummaryMemory/SummaryChat don't each need four separate
    constructor params for what is really one cohesive set loaded together
    in Gamemaster's setup."""

    summary: str
    digest: str
    entity: str
    scene: str


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
        prompts: CompressionPrompts,
        game_name: str,
        mission_id: int,
        npc_roster: list[dict] | None = None,
        logic_config: LLMLogicConfig = LLMLogicConfig.defaults(),
    ):
        self._llm_client = llm_client
        self._summary_prompt = prompts.summary
        self._digest_prompt = prompts.digest
        self._entity_prompt = prompts.entity
        self._scene_prompt = prompts.scene
        self._game_name = game_name
        self._last_k = logic_config.last_k
        self._mission_id = mission_id
        self._npc_roster_json = json.dumps(npc_roster or [])
        self._min_summary_tokens = logic_config.min_summary_tokens
        self._digest_budget_tokens = logic_config.digest_budget_tokens

        self._summarize_lock = threading.Lock()
        summary, n_summarized = crud_instance.get_summary(self._mission_id)
        digest, digest_ledger_tokens = crud_instance.get_digest(self._mission_id)
        self._state = _SummaryState(
            summary=summary,
            n_summarized=n_summarized,
            entities=crud_instance.get_entities(self._mission_id),
            scenes=crud_instance.get_scenes(self._mission_id),
            digest=digest,
            digest_ledger_tokens=digest_ledger_tokens,
        )
        self._history = crud_instance.get_interactions(self._mission_id)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def summary(self) -> str:
        """
        Returns the full, append-only ledger of the chat conversation.

        Returns:
            str: The ledger.
        """
        return self._state.summary

    @property
    def injected_summary(self) -> str:
        """
        Returns what should actually be injected into the story prompt: the
        bounded digest once one exists, falling back to the raw ledger while
        it's still small enough to inject directly. See
        docs/conversation_memory.html for the full design.

        Returns:
            str: The digest, or the raw ledger if no digest has been computed yet.
        """
        return self._state.digest if self._state.digest else self._state.summary

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

        scene_input = '**Input:**\n```json\n{{"previous_scenes": {scenes},"current_history": {text},"known_npc_roster": {roster}}}\n\n**Output:**\n```'
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
                    text=scene_input.format(
                        scenes=scenes_json,
                        text=text_interaction,
                        roster=self._npc_roster_json,
                    )
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

        entity_input = 'Extract entities from the following text and update the given entities:\n{{"text": {text},"entities": {entities},"known_npc_roster": {roster}}}'
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
                        text=text_interaction,
                        entities=entities_json,
                        roster=self._npc_roster_json,
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
        Produces a recap of just `text_interaction` — this chunk only, not the
        whole campaign. Grounded by the current digest (broad, possibly
        slightly stale) and the literal last paragraph of the ledger (precise,
        always fresh), but the model never sees or rewrites the full ledger
        itself. The caller is responsible for appending the returned recap
        onto the ledger — that append is plain string concatenation, not
        something the model does, so preservation of earlier text is a
        structural guarantee rather than an instruction the model has to
        honor. See docs/conversation_memory.html.
        """

        summary_input = (
            'Summarize the following text:\n{{"story_so_far": {digest},'
            '"most_recent_events": {recent},"current_events": {current}}}'
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(
                    text=self._summary_prompt.replace("__RPG__", self._game_name)
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=summary_input.format(
                        digest=self._state.digest,
                        recent=_last_paragraph(self._state.summary),
                        current=text_interaction,
                    )
                ),
            ),
        ]

        return self._call_structured(
            messages=messages,
            parse=lambda s: _Recap.model_validate_json(s).recap,
            log_type=LogType.SUMMARY,
        )

    def _refresh_digest(self, ledger: str) -> str:
        """
        Condenses the *entire* ledger into a short, bounded "story so far"
        blurb — this is what actually gets injected into the story prompt and
        grounds future summarize() calls, once the raw ledger has grown past
        digest_budget_tokens. Infrequent (only called from _try_summarize when
        the ledger crosses the budget again), unlike summarize() which runs
        every cycle. See docs/conversation_memory.html.
        """
        digest_input = (
            'Condense the following story log into a short "story so far" digest:\n'
            '{{"ledger": {ledger}}}'
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(
                    text=self._digest_prompt.replace("__RPG__", self._game_name)
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(text=digest_input.format(ledger=ledger)),
            ),
        ]

        return self._call_structured(
            messages=messages,
            parse=lambda s: _Digest.model_validate_json(s).digest,
            log_type=LogType.DIGEST,
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
                recap = f_summary.result()
                new_n_summarized = self._state.n_summarized + len(interaction_candidates)

                # Deterministic append — code, not the model, owns preserving
                # earlier ledger text. See summarize()'s docstring.
                new_ledger = (
                    f"{self._state.summary}\n\n{recap}".strip()
                    if self._state.summary
                    else recap
                )

                new_digest = self._state.digest
                new_digest_ledger_tokens = self._state.digest_ledger_tokens
                ledger_tokens = self._llm_client.count_tokens(new_ledger)
                if ledger_tokens - self._state.digest_ledger_tokens >= self._digest_budget_tokens:
                    _log.info("Digest | refreshing | ledger_tokens=%d", ledger_tokens)
                    new_digest = self._refresh_digest(new_ledger)
                    new_digest_ledger_tokens = ledger_tokens

                crud_instance.update_summary(
                    self._mission_id, new_ledger, new_n_summarized
                )
                crud_instance.update_entities(self._mission_id, entity_response)
                crud_instance.update_scenes(self._mission_id, scene_response)
                if new_digest != self._state.digest:
                    crud_instance.update_digest(
                        self._mission_id, new_digest, new_digest_ledger_tokens
                    )

                # Single reference swap — readers see either the old or the new
                # state in full, never a partially-updated mix.
                self._state = _SummaryState(
                    summary=new_ledger,
                    n_summarized=new_n_summarized,
                    entities=crud_instance.get_entities(self._mission_id),
                    scenes=crud_instance.get_scenes(self._mission_id),
                    digest=new_digest,
                    digest_ledger_tokens=new_digest_ledger_tokens,
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
