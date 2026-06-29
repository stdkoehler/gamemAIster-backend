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
from typing import Any, TypeVar

from pydantic import BaseModel, model_validator
from pydantic_ai import Agent

from src.llmclient.llm_parameters import LLMLogicConfig
from src.llmclient.llm_config_registry import LLMTask
from src.llmclient.llm_client import LLMClientBase, MessageRole, MessageContent, Message
from src.crud.crud import crud_instance

from src.brain.data_types import Interaction, EntityResponse, Scene

from src.utils.sqllogger import SQLLogger, LogType
from src.utils.logger import configure_logger

_log = configure_logger("conversation_memory")
logger = SQLLogger()

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Private parse targets for structured LLM responses
# ---------------------------------------------------------------------------


class _Recap(BaseModel):
    recap: str

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "_Recap":
        if not self.recap.strip():
            raise ValueError(
                "recap must not be empty — summarize current_events, don't return a blank string."
            )
        return self


class _Digest(BaseModel):
    digest: str

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "_Digest":
        if not self.digest.strip():
            raise ValueError(
                "digest must not be empty — condense the ledger into a non-blank summary."
            )
        return self


class _SceneListOutput(BaseModel):
    """pydantic_ai output_type for scene_summary(). Deliberately wraps the
    list in a `scenes` field rather than using a bare `list[Scene]` — for
    models without native tool-calling support, pydantic_ai falls back to
    asking for JSON matching the output schema in the prompt, and our own
    text_scene_prompt_examples.txt already explicitly instructs the model to
    produce `{"scenes": [...]}`. A bare list[Scene] output_type makes
    pydantic_ai's own schema note say `{"response": [...]}` instead — two
    competing instructions in the same prompt — and the model follows our
    much more detailed one, so the parse fails on a missing "response" key
    before pydantic_ai's validator (or ours) ever runs. Matching the wrapper
    key to what the prompt already asks for removes the conflict instead of
    fighting it. See docs/conversation_memory.html."""

    scenes: list[Scene]

    @model_validator(mode="after")
    def _validate_single_open_scene(self) -> "_SceneListOutput":
        """Enforces the Single Open Scene Invariant (text_scene_prompt_examples.txt)
        in code. A plain ValueError here is enough — pydantic_ai already
        retries on any validation failure surfaced while parsing the
        structured output, the exact same way it retries on a schema
        mismatch, with this message preserved verbatim and fed back to the
        model. No separate @agent.output_validator needed."""
        open_scenes = [s.id for s in self.scenes if not s.completed]
        if len(open_scenes) > 1:
            raise ValueError(
                f"Invalid: {len(open_scenes)} scenes were returned with "
                f"completed=false (ids {open_scenes}). At most ONE scene "
                "may be open at a time — close every scene except the "
                "single most recent one (set completed=true), per the "
                "Single Open Scene Invariant."
            )
        return self


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
        self._npc_roster = npc_roster or []
        self._npc_roster_json = json.dumps(self._npc_roster)
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
    # Shared pydantic_ai run + log pattern
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        agent: Agent[Any, T],
        system_prompt: str,
        user_prompt: str,
        log_type: LogType,
    ) -> T:
        """
        Runs a pre-built pydantic_ai Agent and logs the outcome the same way
        for all four compression calls. Building the Agent — its output_type
        and any extra output validators — stays at each call site, since
        both differ per call; this only covers the mechanical run/log/error
        part that doesn't.
        """
        try:
            result = agent.run_sync(user_prompt)
        except Exception as exc:
            logger.log_llm_call(
                log_type,
                llm_input=f"{system_prompt}\n\n{user_prompt}",
                llm_output=str(exc),
                processed_output="pydantic_ai error",
            )
            raise ValueError(f"pydantic_ai call failed ({log_type}): {exc}") from exc

        logger.log_llm_call(
            log_type,
            llm_input=f"{system_prompt}\n\n{user_prompt}",
            llm_output=result.all_messages_json().decode("utf-8"),
            extracted_json=result.output.model_dump_json(),
            processed_output=result.output.model_dump_json(),
        )
        return result.output

    # ------------------------------------------------------------------
    # Structured extraction methods
    # ------------------------------------------------------------------

    def scene_summary(self, text_interaction: str) -> list[Scene]:
        """
        Returns the scene summary of the chat conversation.

        Uses pydantic_ai specifically to get its validator-driven retry loop
        for the Single Open Scene Invariant (see text_scene_prompt_examples.txt
        and _SceneListOutput._validate_single_open_scene) — if the model
        returns more than one incomplete scene, pydantic_ai automatically
        re-prompts with that exact reason attached, instead of us either
        trusting the prompt alone or silently patching the result in code.
        See docs/conversation_memory.html.
        """
        scenes = crud_instance.get_scenes(self._mission_id)
        last_scene_id = max([scene.id for scene in scenes], default=0)
        scenes_json = json.dumps([scene.model_dump() for scene in scenes])

        scene_input = '**Input:**\n```json\n{{"previous_scenes": {scenes},"current_history": {text},"known_npc_roster": {roster}}}\n\n**Output:**\n```'
        system_prompt = self._scene_prompt.replace("__RPG__", self._game_name)
        user_prompt = scene_input.format(
            scenes=scenes_json,
            text=text_interaction,
            roster=self._npc_roster_json,
        )

        model = self._llm_client.to_pydantic_ai_model(LLMTask.SUMMARY)
        agent = Agent(model, output_type=_SceneListOutput, system_prompt=system_prompt)

        output = self._run_agent(agent, system_prompt, user_prompt, LogType.SCENE)

        # only update last scene and newly created scenes
        return [scene for scene in output.scenes if scene.id >= last_scene_id]

    def extract_entities(self, text_interaction: str) -> EntityResponse:
        """
        Extracts/updates entities from `text_interaction`.

        EntityResponse._validate_references() enforces the reference-integrity
        invariants (matched_key_npc must be a real roster entry,
        updated_entities[].name must be a previously known entity) — it's a
        model_validator on EntityResponse itself, not an
        @agent.output_validator here, since the rule belongs with the model
        it's validating. It needs roster/known-name context the model's own
        fields don't carry, so that context is passed in via
        validation_context below, which pydantic_ai threads through to
        pydantic's ValidationInfo.context. See docs/conversation_memory.html.
        """
        known_entities = crud_instance.get_entities(self._mission_id)
        entities_json = json.dumps([entity.model_dump() for entity in known_entities])

        entity_input = 'Extract entities from the following text and update the given entities:\n{{"text": {text},"entities": {entities},"known_npc_roster": {roster}}}'
        system_prompt = self._entity_prompt.replace("__RPG__", self._game_name)
        user_prompt = entity_input.format(
            text=text_interaction,
            entities=entities_json,
            roster=self._npc_roster_json,
        )

        model = self._llm_client.to_pydantic_ai_model(LLMTask.SUMMARY)
        agent = Agent(
            model,
            output_type=EntityResponse,
            system_prompt=system_prompt,
            validation_context={
                "known_entity_names": {e.name.strip().lower() for e in known_entities},
                "roster_names": {r["name"].strip().lower() for r in self._npc_roster},
            },
        )

        return self._run_agent(agent, system_prompt, user_prompt, LogType.ENTITY)

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
        system_prompt = self._summary_prompt.replace("__RPG__", self._game_name)
        user_prompt = summary_input.format(
            digest=self._state.digest,
            recent=_last_paragraph(self._state.summary),
            current=text_interaction,
        )

        model = self._llm_client.to_pydantic_ai_model(LLMTask.SUMMARY)
        agent = Agent(model, output_type=_Recap, system_prompt=system_prompt)

        return self._run_agent(agent, system_prompt, user_prompt, LogType.SUMMARY).recap

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
        system_prompt = self._digest_prompt.replace("__RPG__", self._game_name)
        user_prompt = digest_input.format(ledger=ledger)

        model = self._llm_client.to_pydantic_ai_model(LLMTask.SUMMARY)
        agent = Agent(model, output_type=_Digest, system_prompt=system_prompt)

        return self._run_agent(agent, system_prompt, user_prompt, LogType.DIGEST).digest

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
