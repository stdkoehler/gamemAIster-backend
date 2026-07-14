"""LLM Client"""

from __future__ import annotations

import json
import re

from typing import Generator, Any, Generic, Protocol, TypeVar
from dataclasses import asdict, dataclass
from enum import StrEnum
from abc import ABC, abstractmethod

from src.utils.logger import configure_logger
from src.utils.sqllogger import LogType, SQLLogger

_log = configure_logger("llm_client")
_sql_logger = SQLLogger()

from urllib.parse import urljoin

import requests
from sseclient import SSEClient
import openai
import tiktoken

# from google import genai
# from google.genai.types import (
#     GenerateContentConfig,
#     Content,
#     UserContent,
#     ModelContent,
#     ThinkingConfig,
# )
import anthropic
from anthropic.types import (
    MessageParam,
    ThinkingConfigEnabledParam,
    ThinkingConfigDisabledParam,
    ThinkingBlock,
    TextBlock,
)

from src.llmclient.llm_config_registry import ConfigRegistry, LLMTask
from src.llmclient.llm_parameters import (
    LLMConfig,
    LLMLogicConfig,
    ThinkingFeebackPolicy,
    UNSET,
)

from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.output import OutputSpec
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel, ValidationError

from src.brain.json_tools import extract_json_schema

_OutputT = TypeVar("_OutputT", bound=BaseModel)


class _RunResultLike(Protocol[_OutputT]):
    """Structural shape run_agent() needs from a run_sync() result — matches
    pydantic_ai's own AgentRunResult, but doesn't require importing it, so
    LLMClientLocal's non-pydantic_ai CompletionAgent (see below) satisfies
    this without inheriting from anything pydantic_ai-specific."""

    output: _OutputT

    def all_messages_json(self) -> bytes: ...


class _RunnableAgent(Protocol[_OutputT]):
    """Structural shape run_agent() needs from whatever build_agent() returns.
    A real pydantic_ai.Agent satisfies this already; so does CompletionAgent."""

    def run_sync(self, user_prompt: str) -> _RunResultLike[_OutputT]: ...


class MessageRole(StrEnum):
    """Message role for conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class MessageContent:
    """Message content with type and content string."""

    text: str
    """The actual content string."""

    thinking: str | None = None
    """The actual thinking string, if applicable."""

    thinking_signature: str | None = None
    """Unique signature for the thinking content, if applicable."""

    reasoning_details: list[dict[str, Any]] | None = None
    """Complete reasoning_details array (for OpenRouter multi-turn reasoning preservation)."""


@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: MessageRole
    """Role of the message sender (e.g., 'system', 'user', 'assistant')."""

    content: MessageContent
    """Content of the message."""


class StreamType(StrEnum):
    """Enumeration for different types of streaming responses."""

    TEXT = "text"
    TEXT_END = "text_end"
    THINKING = "thinking"
    THINKING_END = "thinking_end"


@dataclass
class StreamResponse:
    """
    Streaming response can be delta for frontend rendering and complete thinking/text
    with signature for backend and data storage.
    """

    type: StreamType
    """Type of the stream response."""

    delta: str
    """Delta content for incremental updates."""

    signature: str | None = None
    """Signature for identifying the thinking response. Relevant for multi-turn thinking."""

    full_thinking: str | None = None
    """Complete thinking content to be handled by the backend."""

    full_text: str | None = None
    """Complete text content to be handled by the backend."""

    reasoning_details: list[dict[str, Any]] | None = None
    """Complete reasoning_details array from the provider (for OpenRouter interleaved reasoning)."""


class LLMClientBase(ABC):
    """
    Abstract Base Class defining all methods required for an LLM Interface.
    Handles the multi-layer configuration resolution and UNSET cleanup.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        model_name: str | None = None,
        reasoning_warmstart: str | None = None,
    ) -> None:
        """
        Initializes client with an identity and a 'member config'.
        """
        self.model_name = model_name
        # member_config holds instance-level preferences
        # It is NOT resolved here so that it can still be a 'partial' config
        self.member_config = config if config else LLMConfig()
        self.reasoning_warmstart = reasoning_warmstart

    @property
    def model_identifier(self) -> str:
        """Returns the class name of the client."""
        return self.model_name or self.__class__.__name__

    def get_task_config(
        self, task: LLMTask, call_override: LLMConfig | None = None
    ) -> LLMConfig:
        """
        The 4-Layer Resolve:
        1. Member Config (Instance-level defaults)
        2. Registry (Model Name or Class Name + Task defaults) override Member Config
            with task specific settings.
        3. Call Override (The final word for this specific call) override the above
            with caller defined settings.
        4. Resolution (Fill remaining holes with system defaults) remove all UNSETs.
        """
        # Layer 1: Identity/Hardware baseline (e.g. max_tokens for this model)
        registry_resolved = ConfigRegistry.get_llm_config(self.model_identifier, task)

        # Layer 2: Apply the Registry personality ON TOP OF the client instance
        # This means ARCHITECT's temperature (0.95) will OVERWRITE
        # the client's default temperature.
        active_config = registry_resolved.apply_to(self.member_config)

        # Layer 3: The Per-Call Override (The ultimate authority)
        if call_override:
            active_config = call_override.apply_to(active_config)

        # Layer 4: Final Bake
        return active_config.resolve()

    def get_logic_config(self) -> LLMLogicConfig:
        """
        Retrieves the logic configuration for the current model.
        This includes how many interaction steps to keep unsummarized, how many
        thinking turns to feed back, etc.
        """
        return ConfigRegistry.get_llm_logic_config(self.model_identifier)

    def chat_completion(
        self,
        messages: list[Message],
        reasoning: bool = False,
        config_override: LLMConfig | None = None,
        task: LLMTask = LLMTask.STORY,
    ) -> str:
        """
        Public method to execute a chat completion.
        Ensures active_config is fully resolved (no UNSETs) before execution.
        """
        # We use get_task_config to handle the heavy lifting and UNSET removal
        active_config = self.get_task_config(task, call_override=config_override)

        return self._execute_chat_completion(messages, reasoning, active_config)

    def chat_completion_stream(
        self,
        messages: list[Message],
        reasoning: bool = False,
        config_override: LLMConfig | None = None,
        task: LLMTask = LLMTask.STORY,
    ) -> Generator[StreamResponse, None, None]:
        """
        Public method for streaming. Ensures UNSET removal via get_task_config.
        """
        active_config = self.get_task_config(task, call_override=config_override)

        return self._execute_chat_completion_stream(messages, reasoning, active_config)

    @abstractmethod
    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        """
        Internal abstract method: Child classes MUST implement the
        actual API call logic here.
        """

    @abstractmethod
    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:
        """
        Internal abstract method: Child classes MUST implement the
        actual API call logic here.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """

    def stop_generation(self) -> None:
        """Stop the current generation. Override in clients that support mid-stream cancellation."""

    # ------------------------------------------------------------------
    # pydantic_ai bridge (pilot — see docs/conversation_memory.html)
    # ------------------------------------------------------------------

    def _to_pydantic_ai_model(self, task: LLMTask, reasoning: bool = False) -> Model:
        """
        Builds the pydantic_ai Model equivalent to this client, for callers
        that want pydantic_ai's validator-driven retry loop instead of our
        own chat_completion(). Reuses this client's own connection details
        and the same ConfigRegistry-resolved settings every other call for
        `task` already goes through.

        `reasoning` mirrors the `reasoning` flag our own chat_completion()
        takes — every structured-output call site (compaction, NPC
        generation, mission generation, oracle alignment) always passed
        reasoning=True before this bridge existed, so callers should keep
        passing reasoning=True for parity rather than silently dropping it.
        Each subclass decides how (or whether) to honor it — see the
        per-client overrides.

        Overridden by every concrete client (LLMClientLocal, LLMClientDeepSeek,
        the Anthropic-compatible clients, LLMClientOpenRouter). Defined here
        rather than each override reaching into another instance's private
        attributes from outside the class. The base implementation only
        matters for any future client that hasn't added an override yet.
        """
        raise NotImplementedError(
            f"No pydantic_ai bridge yet for {type(self).__name__}."
        )

    @staticmethod
    def _to_pydantic_ai_settings(config: LLMConfig) -> ModelSettings:
        """Maps the subset of our LLMConfig that OpenAI/Anthropic-style APIs
        actually accept. Local-model-only sampler knobs (min_p, repetition
        penalty, mirostat, smoothing_factor, ...) have no equivalent here and
        are dropped — they wouldn't have been sent to these providers by our
        own clients either."""
        settings: ModelSettings = {}
        if config.max_tokens is not UNSET:
            settings["max_tokens"] = config.max_tokens
        if config.temperature is not UNSET:
            settings["temperature"] = config.temperature
        if config.top_p is not UNSET:
            settings["top_p"] = config.top_p
        if config.presence_penalty is not UNSET:
            settings["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not UNSET:
            settings["frequency_penalty"] = config.frequency_penalty
        if config.stop is not UNSET:
            settings["stop_sequences"] = config.stop
        return settings

    # Whether this client needs output_type swapped to PromptedOutput when
    # reasoning is requested, to avoid forcing a tool call. This is a static
    # fact about the client/pydantic_ai Model class it wraps, not something
    # that varies per call — forcing a tool call (pydantic_ai's default
    # output_type=SomeModel behavior) is rejected by some reasoning-capable
    # models' real APIs when thinking is also enabled (Anthropic,
    # deepseek-reasoner). See _to_pydantic_ai_model()'s docstring for which
    # of our clients already get a free pass on this via pydantic_ai's own
    # per-provider handling (Claude/MiniMax via AnthropicModel's internal
    # auto-switch, DeepSeek via DeepSeekProvider's profile fix) — those stay
    # False. Only LLMClientOpenRouter, which has no such handling, sets this
    # True.
    _needs_prompted_output_for_reasoning: bool = False

    def build_agent(
        self,
        task: LLMTask,
        output_type: type[_OutputT],
        system_prompt: str,
        reasoning: bool = False,
        **agent_kwargs: Any,
    ) -> _RunnableAgent[_OutputT]:
        """
        One-stop agent builder — the entry point business logic
        (SummaryMemory, Gamemaster) should use instead of calling
        _to_pydantic_ai_model() separately or constructing Agent() directly.
        Composes the reasoning model-setting with the output-mode adjustment
        some clients need to keep structured output working (see
        _needs_prompted_output_for_reasoning above). Callers never need to
        know which clients need which — that's the whole point of
        encapsulating it here rather than branching on client type in
        business logic.

        `**agent_kwargs` passes through to Agent() for the few things that
        vary per call site (e.g. extract_entities()'s validation_context).

        This base implementation always returns a real pydantic_ai Agent.
        LLMClientLocal overrides this method entirely to return a
        CompletionAgent instead — see its docstring for why pydantic_ai's
        Agent/Model interface can't express what local structured-output
        calls actually need.
        """
        model = self._to_pydantic_ai_model(task, reasoning=reasoning)
        wrapped_output_type: OutputSpec[_OutputT] = (
            PromptedOutput(output_type)
            if reasoning and self._needs_prompted_output_for_reasoning
            else output_type
        )
        return Agent(model, output_type=wrapped_output_type, system_prompt=system_prompt, **agent_kwargs)

    @staticmethod
    def run_agent(
        agent: _RunnableAgent[_OutputT],
        system_prompt: str,
        user_prompt: str,
        log_type: LogType,
    ) -> _OutputT:
        """
        Runs a pre-built agent (from build_agent()) and logs the outcome.
        Shared by every structured-output call site (SummaryMemory's
        compaction calls, Gamemaster's mission/NPC generation, Oracle
        alignment) — building the agent (its output_type and any extra
        output validators/validation_context) stays at each call site, since
        those differ per call; this only covers the mechanical run/log/error
        part that doesn't.

        Typed against the `_RunnableAgent` structural protocol rather than
        `pydantic_ai.Agent` directly: every client except LLMClientLocal
        returns a real pydantic_ai Agent (retries on validation failure
        handled by pydantic_ai itself — a Pydantic validator's ValueError is
        fed back to the model verbatim and retried automatically).
        LLMClientLocal.build_agent() returns CompletionAgent instead, which
        satisfies the same protocol but drives retries itself through
        chat_completion(); see its docstring for why.
        """
        try:
            result = agent.run_sync(user_prompt)
        except Exception as exc:
            _sql_logger.log_llm_call(
                log_type,
                llm_input=f"{system_prompt}\n\n{user_prompt}",
                llm_output=str(exc),
                processed_output="pydantic_ai error",
            )
            raise ValueError(f"pydantic_ai call failed ({log_type}): {exc}") from exc

        output_json = result.output.model_dump_json()
        _sql_logger.log_llm_call(
            log_type,
            llm_input=f"{system_prompt}\n\n{user_prompt}",
            llm_output=result.all_messages_json().decode("utf-8"),
            extracted_json=output_json,
            processed_output=output_json,
        )
        return result.output

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _maybe_extract_hidden_state(self, full_thinking: str) -> str:
        """
        Extracts the content of the <hidden_state> tag from the given text.
        Only active when reasoning_warmstart is set — local models that use
        hidden_state-based multi-turn reasoning store only the extracted state,
        not the full thinking trace.
        """
        if self.reasoning_warmstart is not None and full_thinking is not None:
            pattern = re.compile(
                r"<hidden_state>(.*?)</hidden_state>",
                re.DOTALL | re.IGNORECASE,
            )
            all_states = pattern.findall(full_thinking)
            if not all_states:
                return ""
            return "\n".join([s.strip() for s in all_states if s.strip()])
        return full_thinking

    def _message_to_dict(self, message: Message) -> dict[str, str]:
        """Convert Message to the basic {role, content} dict understood by OpenAI-compatible APIs."""
        return {"role": message.role.value, "content": message.content.text}

    @staticmethod
    def _build_turn_map(messages: list[Message]) -> dict[int, int]:
        """
        Maps each assistant message's list index to how many turns ago it appeared.
        Used to apply thinking-feedback policies that limit how far back we inject thinking.
        """
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg.role == MessageRole.ASSISTANT
        ]
        num = len(assistant_indices)
        return {idx: (num - rank) for rank, idx in enumerate(assistant_indices)}


@dataclass
class _CompletionRunResult(Generic[_OutputT]):
    """Duck-typed stand-in for pydantic_ai's AgentRunResult — see
    CompletionAgent's docstring for why LLMClientLocal needs one."""

    output: _OutputT
    _log_payload: dict[str, Any]

    def all_messages_json(self) -> bytes:
        return json.dumps(self._log_payload, ensure_ascii=False, indent=2).encode("utf-8")


class CompletionAgent(Generic[_OutputT]):
    """
    Duck-typed pydantic_ai.Agent replacement, used only by
    LLMClientLocal.build_agent(). Drives structured output through
    chat_completion() — the raw completions path — instead of pydantic_ai's
    OpenAIChatModel/Agent machinery.

    Why: two separate, hard-won findings this session showed pydantic_ai's
    Agent/Model interface structurally cannot serve local models well:

    1. Schema delivery is unreliable. pydantic_ai's default output mode
       sends the schema via an OpenAI-style `tools` field, a separate
       request parameter from the prompt text. Verified live (HTTP
       interception) that at least one real local backend/template
       combination silently drops that field — the model never sees the
       schema at all, in any form — while returning a normal 200 response,
       so nothing about the failure is visible without inspecting the raw
       request. PromptedOutput (an in-prompt schema instruction) sidesteps
       this, but the fact that it was ever needed at all — for structured
       output style local models genuinely are supposed to support well —
       is itself evidence the request-shape approach doesn't transfer
       cleanly to arbitrary local model/server/template combinations the
       way it does for well-known hosted providers.
    2. Real reasoning support is impossible to express. The local
       reasoning-tuned models this client targets (see _adjust_reasoning())
       need an assistant-message prefill + a `continue_` request flag to
       actually engage their thinking mode — chat_completion() already
       implements this correctly and has for the whole time this client has
       existed. pydantic_ai's Agent.run_sync() always appends the prompt as
       a *new* user turn; there's no supported way to make the final message
       in the request a partial assistant continuation instead. `reasoning`
       is consequently a silent no-op for LLMClientLocal through
       pydantic_ai — confirmed live (zero ThinkingPart content across
       repeated runs of a multi-step word problem).

    Routing back through chat_completion() closes both gaps at once, using
    code this project has relied on since before pydantic_ai was
    introduced. The cost: retries are driven by hand here (one retry,
    feeding the validation error back as a follow-up user turn) rather than
    by pydantic_ai's own loop, and the schema has to be spelled out in the
    prompt text ourselves (mirroring PromptedOutput's own instruction
    template) since nothing else will communicate it. Exposes exactly the
    surface run_agent() needs (see _RunnableAgent/_RunResultLike above), so
    this is fully transparent to callers — SummaryMemory/Gamemaster never
    know LLMClientLocal uses a different mechanism than every other client.
    """

    _MAX_RETRIES = 1

    def __init__(
        self,
        client: LLMClientLocal,
        task: LLMTask,
        output_type: type[_OutputT],
        system_prompt: str,
        reasoning: bool,
        validation_context: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._task = task
        self._output_type = output_type
        self._reasoning = reasoning
        self._validation_context = validation_context
        # Same instruction template pydantic_ai's own PromptedOutput uses —
        # kept identical since that phrasing is already proven to work.
        schema = json.dumps(output_type.model_json_schema())
        self._system_prompt = (
            f"{system_prompt}\n\n"
            f"Always respond with a JSON object that's compatible with this schema:\n\n{schema}\n\n"
            "Don't include any text or Markdown fencing before or after."
        )

    def run_sync(self, user_prompt: str) -> _CompletionRunResult[_OutputT]:
        messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=self._system_prompt)),
            Message(role=MessageRole.USER, content=MessageContent(text=user_prompt)),
        ]
        last_error: Exception | None = None
        response = ""
        for attempt in range(self._MAX_RETRIES + 1):
            response = self._client.chat_completion(
                messages=messages, reasoning=self._reasoning, task=self._task
            )
            try:
                json_str = extract_json_schema(response)
                output = self._output_type.model_validate_json(
                    json_str, context=self._validation_context
                )
                return _CompletionRunResult(
                    output=output,
                    _log_payload={
                        "system_prompt": self._system_prompt,
                        "conversation": [
                            {"role": m.role.value, "content": m.content.text} for m in messages
                        ],
                        "final_response": response,
                    },
                )
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < self._MAX_RETRIES:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=MessageContent(text=response)))
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=MessageContent(
                                text=(
                                    f"Your last response could not be parsed/validated: {exc}\n\n"
                                    "Respond again with ONLY the corrected JSON object — no other text."
                                )
                            ),
                        )
                    )

        raise ValueError(
            f"LLMClientLocal structured-output call failed after {self._MAX_RETRIES + 1} "
            f"attempt(s): {last_error}"
        )


class LLMClientLocal(LLMClientBase):
    """
    LLMClient is a class that provides methods for interacting with the LLM API.

    Attributes:
        _base_url: The base URL of the LLM API.
        _completion_url: The URL for making completion requests.
        _token_url: The URL for counting tokens in a text.

    """

    def build_agent(
        self,
        task: LLMTask,
        output_type: type[_OutputT],
        system_prompt: str,
        reasoning: bool = False,
        **agent_kwargs: Any,
    ) -> CompletionAgent[_OutputT]:
        """
        Overrides the base pydantic_ai-based implementation entirely — see
        CompletionAgent's docstring for why. `**agent_kwargs` only carries
        `validation_context` in practice (the one thing call sites pass
        beyond the fixed params); threaded through to
        model_validate_json(context=...) exactly like pydantic_ai's own
        validation_context threads to ValidationInfo.context.
        """
        return CompletionAgent(
            client=self,
            task=task,
            output_type=output_type,
            system_prompt=system_prompt,
            reasoning=reasoning,
            validation_context=agent_kwargs.get("validation_context"),
        )

    def __init__(
        self,
        base_url: str,
        config: LLMConfig | None = None,
        model_name: str | None = None,
        reasoning_warmstart: str | None = None,
    ):
        super().__init__(
            config=config,
            model_name=model_name,
            reasoning_warmstart=reasoning_warmstart,
        )
        self._base_url = base_url
        self._completion_url = urljoin(base_url, "/v1/completions")
        self._chat_completion_url = urljoin(base_url, "/v1/chat/completions")
        self._stop_generation_url = urljoin(base_url, "/v1/internal/stop-generation")
        self._token_url = urljoin(base_url, "/v1/internal/token-count")
        self._headers = {"Content-Type": "application/json"}

    def _post_request(
        self, url: str, payload: dict[str, Any], timeout: int = 60
    ) -> Any:
        """
        Sends a POST request to the specified URL with the given payload.

        Args:
            url (str): The URL to send the request to.
            payload (dict): The payload to include in the request body.

        Returns:
            dict: The JSON response from the request, parsed as a dictionary.

        """
        response = requests.post(
            url, headers=self._headers, json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    def _adjust_reasoning_mistral24b(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        """
        Adjusts the messages to include reasoning prompts if requested.
        Required for DeepHermes Mistral 24b
        """
        system_msg = next(
            (msg for msg in messages if msg.role == MessageRole.SYSTEM),
            Message(role=MessageRole.SYSTEM, content=MessageContent(text="")),
        )
        user_msgs = [
            msg
            for msg in messages
            if msg.role == MessageRole.USER or msg.role == MessageRole.ASSISTANT
        ]

        # We add the system message to the first user message because our actual
        # system message needs to be the thinking command. Builds a new Message
        # rather than mutating user_msgs[0] in place: user_msgs[0] is the same
        # object living in the caller's own `messages` list, and CompletionAgent
        # (llm_client.py) reuses that same list/objects across its own retry
        # loop — an in-place mutation here would re-prepend the system message
        # onto an already-adjusted first user message on every retry.
        if user_msgs:
            user_msgs[0] = Message(
                role=user_msgs[0].role,
                content=MessageContent(
                    text=f"{system_msg.content.text}\n\n{user_msgs[0].content.text}",
                    thinking=user_msgs[0].content.thinking,
                    thinking_signature=user_msgs[0].content.thinking_signature,
                    reasoning_details=user_msgs[0].content.reasoning_details,
                ),
            )

        # this is the message the assistant should complete, we add the warmstart
        if self.reasoning_warmstart is not None:
            user_msgs.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=MessageContent(text=self.reasoning_warmstart),
                )
            )
            payload["continue_"] = True

        return [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(
                    text="You are a deep thinking AI. Enclose thoughts in <think> </think> tags."
                ),
            )
        ] + user_msgs, payload

    def _adjust_reasoning_gemma3_r1(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        """
        Gemma3 R1 requires refilled <think> to trigger thinking
        """
        if self.reasoning_warmstart is not None:
            payload["continue_"] = True
            return (
                messages
                + [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=MessageContent(text=self.reasoning_warmstart),
                    )
                ],
                payload,
            )
        return messages, payload

    def _adjust_reasoning(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        """different models need different reasoning adjustments"""
        if self.model_name == "mistral-24b-hermes":
            messages, payload = self._adjust_reasoning_mistral24b(messages, payload)
        elif self.model_name == "gemma-3-r1-27b":
            messages, payload = self._adjust_reasoning_gemma3_r1(messages, payload)

        # Only previously extracted hidden state is stored as thinking in db for multi-turn preservation,
        # Prefix the Assistant's messages with the thinking in hidden state
        if self.reasoning_warmstart is not None:
            for msg in messages:
                if msg.role == MessageRole.ASSISTANT and msg.content.thinking:
                    msg.content.text = f"<hidden_state>{msg.content.thinking}</hidden_state>\n\n{msg.content.text}"

        return messages, payload

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        payload = asdict(config)

        if reasoning:
            messages, payload = self._adjust_reasoning(messages, payload)

        payload.update(
            {
                "messages": [self._message_to_dict(msg) for msg in messages],
                "stream": True,
            }
        )

        stream_response = requests.post(
            self._chat_completion_url,
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=360,
        )
        client = SSEClient(stream_response)  # type: ignore

        # Local model doesn't have specific thinking events
        # We need to built them ourselves with the think tags
        # 1. Define constants to avoid "Magic Strings" and calculation errors
        TAG_START = "<think>"
        TAG_END = "</think>"
        # We only need to buffer enough to catch the longest tag being split across chunks
        MAX_TAG_LEN = max(len(TAG_START), len(TAG_END))

        class ParseState(StrEnum):
            PRE_THINK = "pre_think"  # Looking for <think>
            THINKING = "thinking"  # Looking for </think>
            TEXT = "text"  # Thinking is done, just stream text

        buffer = ""
        state = ParseState.PRE_THINK

        accumulated_text = ""
        accumulated_thinking = ""

        for event in client.events():
            if event.data == "[DONE]":
                break

            try:
                result = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            chunk = result["choices"][0]["delta"].get("content", "")

            if not chunk:
                continue

            buffer += chunk

            # The 'while True' allows us to process state transitions
            # immediately within the same chunk (e.g., "Hello <think>")
            while True:

                # --- PHASE 1: Normal Text (Looking for Start Tag) ---
                if state == ParseState.PRE_THINK:
                    if (idx := buffer.find(TAG_START)) != -1:
                        # Tag found: Emit text before tag, emit tag, switch state
                        before, after = buffer[:idx], buffer[idx + len(TAG_START) :]

                        if before:
                            accumulated_text += before
                            yield StreamResponse(type=StreamType.TEXT, delta=before)

                        # Don't emit the tags
                        # yield StreamResponse(type=StreamType.THINKING, delta=TAG_START)

                        buffer = after
                        state = ParseState.THINKING
                        continue  # Re-process 'after' in the new state immediately

                    # Tag NOT found: Emit safe part of buffer, keep tail
                    # at least tag length to catch split tags
                    if len(buffer) > MAX_TAG_LEN:
                        safe_content = buffer[:-MAX_TAG_LEN]
                        buffer = buffer[-MAX_TAG_LEN:]
                        accumulated_text += safe_content
                        yield StreamResponse(type=StreamType.TEXT, delta=safe_content)
                    break

                # --- PHASE 2: Thinking (Looking for End Tag) ---
                elif state == ParseState.THINKING:
                    if (idx := buffer.find(TAG_END)) != -1:
                        # Tag found: Emit thought before tag, emit tag, finalize thought
                        before, after = buffer[:idx], buffer[idx + len(TAG_END) :]

                        if before:
                            accumulated_thinking += before
                            yield StreamResponse(type=StreamType.THINKING, delta=before)

                        # Don't emit the tags
                        # yield StreamResponse(type=StreamType.THINKING, delta=TAG_END)

                        # Send the full thinking object or, if we're using prefill with
                        # hidden_state, just send hidden state
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            full_thinking=self._maybe_extract_hidden_state(
                                accumulated_thinking.strip()
                            ),
                        )

                        buffer = after
                        state = ParseState.TEXT
                        continue

                    # Tag NOT found: Emit safe part of buffer, keep tail
                    if len(buffer) > MAX_TAG_LEN:
                        safe_content = buffer[:-MAX_TAG_LEN]
                        buffer = buffer[-MAX_TAG_LEN:]
                        accumulated_thinking += safe_content
                        yield StreamResponse(
                            type=StreamType.THINKING, delta=safe_content
                        )
                    break

                # --- PHASE 3: Done (Passthrough) ---
                elif state == ParseState.TEXT:
                    if buffer:
                        accumulated_text += buffer
                        yield StreamResponse(type=StreamType.TEXT, delta=buffer)
                        buffer = ""
                    break

        # Handle any remaining buffer if stream ends unexpectedly
        if buffer and state != ParseState.THINKING:
            accumulated_text += buffer
            yield StreamResponse(type=StreamType.TEXT, delta=buffer)

        yield StreamResponse(
            type=StreamType.TEXT_END,
            delta="",
            full_text=accumulated_text.strip(),
        )

    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:
        payload = asdict(config)

        if reasoning:
            messages, payload = self._adjust_reasoning(messages, payload)

        payload.update(
            {
                "messages": [self._message_to_dict(msg) for msg in messages],
                "stream": False,
            }
        )
        result = self._post_request(self._chat_completion_url, payload, timeout=3600)
        message = result["choices"][0]["message"]
        # With reasoning_warmstart's continue_ trick, this server has been observed
        # to classify the ENTIRE completion as reasoning_content (leaving content
        # empty) when the model never emits a literal closing </think> in its own
        # continuation — confirmed live against Gemma-3-R1, independent of the
        # --jinja/--enable-thinking server flags. Falling back to reasoning_content
        # recovers the answer either way; extract_json_schema() already strips any
        # ```json fencing regardless of which field it came from.
        text = message.get("content") or message.get("reasoning_content")
        return text if text is not None else ""

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        result = self._post_request(self._token_url, {"text": text})
        return int(result.get("length", 0))

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """
        requests.post(self._stop_generation_url, timeout=60)

    # No _to_pydantic_ai_model() override: build_agent() is overridden
    # entirely above (see CompletionAgent), so the base class's
    # NotImplementedError-raising default is correct here — nothing should
    # ever call this for LLMClientLocal.


# --- Local Client for natively reasoning/tool-calling models (pydantic_ai) ---


class LLMClientLocalOpenAI(LLMClientLocal):
    """
    Local client for models that natively support reasoning AND tool-calling
    over textgen-webui's OpenAI-compatible endpoint — e.g. Gemma 4 26B.

    The whole reason the parent LLMClientLocal bypasses pydantic_ai (returning
    CompletionAgent instead of a real Agent) is that its target models —
    DeepHermes, Gemma-3-R1 — are reasoning-tuned GGUFs that (a) only engage
    thinking via a <think>-prefill + `continue_` request trick pydantic_ai's
    Agent interface structurally cannot express, and (b) run on Gemma chat
    templates with no tool-call tokens at all, so the schema never reaches the
    model. See CompletionAgent's docstring and docs/conversation_memory.html.

    A model with *native* reasoning removes blocker (a) entirely: no prefill
    trick is needed, so pydantic_ai's ordinary "append a user turn" flow is
    fine — exactly the situation deepseek-reasoner is already in. This subclass
    therefore keeps all of LLMClientLocal's streaming / token-counting /
    raw-completion machinery (reused as-is for the narrative chat path) but
    routes structured output back through pydantic_ai's real Agent, via an
    OpenAIChatModel pointed at the local server.

    Two decisions, both live-verified against a real Gemma 4 26B GGUF served
    by textgen-webui/llama.cpp (tests/test_llm_clients_live.py):

      * Native tool-calling (`_needs_prompted_output_for_reasoning = False`,
        inherited from base). Unlike the Gemma 3 "R1" GGUFs that motivated
        blocker (b) — whose chat template has no tool-call tokens at all —
        Gemma 4's template DOES support tool-calling: verified that a forced
        `tool_choice: "required"` returns a proper `tool_calls` response,
        AND that it coexists with thinking (the model reasons first, then
        emits the tool call, given enough token budget — no 400, no conflict).
        Forced tool-calling is pydantic_ai's most reliable structured-output
        mode (decode-level schema enforcement), so it's preferred here.
        PromptedOutput was tried first and *failed live* on this model — with
        the schema in the prompt, Gemma 4 echoed the schema back verbatim as
        its answer instead of an instance — which is exactly why this is
        False, not True.
      * Thinking always on. This model reasons unconditionally: verified that
        `chat_template_kwargs.enable_thinking` (the HF/vLLM convention) is a
        no-op on this build — identical reasoning output with it true vs false
        — and that thinking comes back in a `reasoning_content` response field
        that pydantic_ai's OpenAIChatModel already surfaces as a ThinkingPart.
        `_thinking_body()` is still sent defensively (see there).
      * Sampling lives in the ConfigRegistry, not here. In this codebase the
        narrative-vs-structured distinction maps 1:1 to the task: STORY is the
        only narrative (chat_completion) task, and ARCHITECT/SUMMARY are always
        structured (build_agent) output — so the registry's per-(model, task)
        resolution already expresses the whole split. The `LLMClientLocalOpenAI`
        _MATRIX entry sets Gemma 4's narrative sampling on STORY (temp 1.0,
        top_p 0.95, top_k 64) and the structured sampler set on ARCHITECT +
        SUMMARY (low temp + scoped repetition penalty — temp=1.0 can't be used
        for tool calls: llama.cpp doesn't hard-enforce tool_choice, so the model
        answers in prose instead of calling the tool, and loops without a
        repetition penalty; verified live). This client only *forwards* the
        local-only sampler knobs the registry resolves into extra_body (see
        `_to_pydantic_ai_model`), since pydantic_ai's ModelSettings can't carry
        them — it holds no sampling values of its own.
    """

    # Native tool-calling verified working (incl. with thinking) — see the
    # class docstring. Left at the base default of False explicitly for the
    # reader, since the sibling LLMClientOpenRouter flips it to True.
    _needs_prompted_output_for_reasoning = False

    @property
    def model_identifier(self) -> str:
        """Key the ConfigRegistry by class name rather than the dynamic
        LOCAL_MODEL string, so the `LLMClientLocalOpenAI` _MATRIX entry (which
        carries this model's per-task sampling) is hit regardless of what the
        loaded model is named. `self.model_name` still drives the actual API
        model field in _to_pydantic_ai_model()."""
        return type(self).__name__

    # Local sampler knobs to forward via extra_body on the structured path.
    # pydantic_ai's ModelSettings only carries temperature/top_p/max_tokens/
    # penalties/stop; these have no OpenAI equivalent and would otherwise be
    # dropped, but textgen-webui accepts them. Values come from the registry
    # (get_task_config), not from here.
    _STRUCTURED_EXTRA_SAMPLER_FIELDS = (
        "top_k",
        "min_p",
        "repetition_penalty",
        "repetition_penalty_range",
        "smoothing_factor",
        "sampler_priority",
    )

    @staticmethod
    def _thinking_body(reasoning: bool) -> dict[str, Any]:
        """The `chat_template_kwargs.enable_thinking` toggle (HF/vLLM convention).
        Sent by both the raw streaming/completion path (_adjust_reasoning) and
        the pydantic_ai bridge (_to_pydantic_ai_model's extra_body) so the two
        can't drift — same containment pattern as LLMClientDeepSeek._thinking_extra_body.

        VERIFIED to be a no-op on the tested Gemma 4 26B / llama.cpp build
        (the model reasons unconditionally regardless of this flag), but kept
        because it's harmless, self-documenting, and the correct lever for any
        other model routed through LLM=LOCAL_OPENAI whose server DOES honor it.
        Deliberately NOT pydantic_ai's unified `thinking=True`, which the
        generic OpenAIChatModel maps to OpenAI's `reasoning_effort` — a dead
        field against textgen-webui, same as the DeepSeek/OpenRouter cases in
        conversation_memory.html."""
        return {"chat_template_kwargs": {"enable_thinking": reasoning}}

    def build_agent(
        self,
        task: LLMTask,
        output_type: type[_OutputT],
        system_prompt: str,
        reasoning: bool = False,
        **agent_kwargs: Any,
    ) -> _RunnableAgent[_OutputT]:
        """Bypass LLMClientLocal's CompletionAgent override and use the base
        pydantic_ai Agent path — the entire point of this subclass. Calls the
        grandparent (LLMClientBase) implementation explicitly, since plain
        super() would resolve to LLMClientLocal's CompletionAgent version.

        (No retry-count bump here: the structured-output failures found on this
        model were sampling-driven — a low-temp forced-tool-call repetition
        loop, and prose-instead-of-tool-call at high temp — fixed by the
        registry's structured sampler set (temp/repetition_penalty) and the
        larger structured max_tokens, not by extra retries, which just re-run
        the whole slow always-on reasoning and fail the same way.)"""
        return LLMClientBase.build_agent(
            self, task, output_type, system_prompt, reasoning=reasoning, **agent_kwargs
        )

    def _adjust_reasoning(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        """Native-reasoning path for the raw chat/streaming completions: request
        the model's own thinking mode via the chat template instead of the
        parent's <think>-prefill + continue_ hack (which is only for the
        reasoning-tuned models LLMClientLocal targets)."""
        payload.update(self._thinking_body(True))
        return messages, payload

    def _to_pydantic_ai_model(self, task: LLMTask, reasoning: bool = False) -> Model:
        # Sampling (incl. the structured low-temp set) and max_tokens come from
        # the ConfigRegistry entry for this client — see llm_config_registry.py.
        config = self.get_task_config(task)
        settings = self._to_pydantic_ai_settings(config)
        # extra_body carries the thinking toggle plus the local-only sampler
        # knobs pydantic_ai's ModelSettings can't express (temperature/top_p/
        # max_tokens already went through settings).
        extra_body = dict(self._thinking_body(reasoning))
        for field in self._STRUCTURED_EXTRA_SAMPLER_FIELDS:
            value = getattr(config, field)
            if value is not UNSET:
                extra_body[field] = value
        settings["extra_body"] = extra_body
        return OpenAIChatModel(
            self.model_name or "local-model",
            provider=OpenAIProvider(
                base_url=f"{self._base_url.rstrip('/')}/v1",
                api_key="not-needed",  # textgen-webui ignores it; SDK requires non-empty
            ),
            settings=settings,
        )

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        """Overrides the parent's inline-`<think>`-tag stream parser. A native
        reasoning model on this server streams thinking in a separate
        `reasoning_content` delta field (verified live — same shape DeepSeek
        uses), never as `<think>` tokens inside `content`, so the parent's
        tag-based state machine would emit zero THINKING events and the whole
        reasoning trace would be silently dropped from the narrative stream and
        the stored `full_thinking`. This handles the reasoning_content/content
        split directly instead. Sampling (Gemma's narrative config on STORY)
        comes from the registry via `config` — no overlay needed here."""
        payload = asdict(config)
        if reasoning:
            messages, payload = self._adjust_reasoning(messages, payload)
        payload.update(
            {
                "messages": [self._message_to_dict(msg) for msg in messages],
                "stream": True,
            }
        )

        stream_response = requests.post(
            self._chat_completion_url,
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=360,
        )
        client = SSEClient(stream_response)  # type: ignore

        accumulated_thinking = ""
        accumulated_text = ""
        in_reasoning = False
        reasoning_ended = False

        for event in client.events():
            if event.data == "[DONE]":
                break
            try:
                result = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            choices = result.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            reasoning_chunk = delta.get("reasoning_content")
            if reasoning_chunk:
                if not in_reasoning:
                    in_reasoning = True
                    yield StreamResponse(type=StreamType.THINKING, delta="")
                accumulated_thinking += reasoning_chunk
                yield StreamResponse(type=StreamType.THINKING, delta=reasoning_chunk)

            content_chunk = delta.get("content")
            if content_chunk:
                # First content token marks the end of the thinking phase.
                if in_reasoning and not reasoning_ended:
                    in_reasoning = False
                    reasoning_ended = True
                    yield StreamResponse(
                        type=StreamType.THINKING_END,
                        delta="",
                        full_thinking=self._maybe_extract_hidden_state(
                            accumulated_thinking.strip()
                        ),
                    )
                accumulated_text += content_chunk
                yield StreamResponse(type=StreamType.TEXT, delta=content_chunk)

        # Stream ended — close out any still-open thinking (e.g. a pure-thinking
        # response that never produced content), then always the text.
        if in_reasoning and not reasoning_ended:
            yield StreamResponse(
                type=StreamType.THINKING_END,
                delta="",
                full_thinking=self._maybe_extract_hidden_state(
                    accumulated_thinking.strip()
                ),
            )
        yield StreamResponse(
            type=StreamType.TEXT_END,
            delta="",
            full_text=accumulated_text.strip(),
        )


# --- DeepSeek Client (OpenAI-Compatible SDK) ---


class LLMClientDeepSeek(LLMClientBase):
    """
    LLMClient implementation for DeepSeek using the OpenAI SDK.
    https://api-docs.deepseek.com/api/create-chat-completion
    with beta we can also use prefill (=prefix)
    Assistant message parameters:
    name
    string
    An optional name for the participant. Provides the model information to differentiate
    between participants of the same role.

    prefix
    bool
    (Beta) Set this to true to force the model to start its answer by the content of the
    supplied prefix in this assistant message.
    You must set base_url="https://api.deepseek.com/beta" to use this feature.

    reasoning_content
    string
    nullable
    (Beta) Used for the deepseek-reasoner model in the Chat Prefix Completion feature
    as the input for the CoT in the last assistant message. When using this feature,
    the prefix parameter must be set to true.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        config: LLMConfig | None = None,
        reasoning_warmstart: str | None = None,
    ):
        super().__init__(config=config, reasoning_warmstart=reasoning_warmstart)
        self._client = openai.OpenAI(
            base_url="https://api.deepseek.com", api_key=api_key
        )
        self._model = model

    @staticmethod
    def _thinking_extra_body(reasoning: bool) -> dict[str, Any]:
        """DeepSeek's `thinking` toggle, sent unconditionally based on `reasoning`
        regardless of which model (deepseek-chat vs. deepseek-reasoner) is
        selected — shared by the raw chat_completion path (streaming and
        non-streaming) and the pydantic_ai bridge so all three stay in sync."""
        return {"thinking": {"type": "enabled" if reasoning else "disabled"}}

    def _adjust_reasoning(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        # Only previously extracted hidden state is stored as thinking in db for multi-turn preservation,
        # Prefix the Assistant's messages with the thinking in hidden state
        if self.reasoning_warmstart is not None:
            for msg in messages:
                if msg.role == MessageRole.ASSISTANT and msg.content.thinking:
                    msg.content.text = f"<hidden_state>{msg.content.thinking}</hidden_state>\n\n{msg.content.text}"

        return messages, payload

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        try:
            payload = asdict(config)
            if reasoning:
                messages, payload = self._adjust_reasoning(messages, payload)

            messages_dicts: list[dict[str, Any]] = [
                self._message_to_dict(msg) for msg in messages
            ]

            # In theory this works. However DeepSeek is often confused and either
            # does not provide reasoning content or puts <think> tags in the normal
            # text output: For now we should not use warmstart.
            # if reasoning and self.reasoning_warmstart is not None:
            #     messages_dicts.append(
            #         {
            #             "role": "assistant",
            #             "reasoning_content": self.reasoning_warmstart,
            #             "content": "",
            #             "prefix": True,
            #         }
            #     )

            accumulated_thinking = (
                self.reasoning_warmstart
                if reasoning and self.reasoning_warmstart is not None
                else ""
            )
            accumulated_text = ""
            in_reasoning = False
            reasoning_ended = False

            stream_response = self._client.chat.completions.create(
                model=self._model,
                messages=messages_dicts,  # type: ignore
                max_tokens=config.max_tokens,  # type: ignore
                temperature=config.temperature,  # type: ignore
                top_p=config.top_p,  # type: ignore
                stream=True,
                extra_body=self._thinking_extra_body(reasoning),
            )
            for event in stream_response:
                delta = event.choices[0].delta  # type: ignore
                if delta.reasoning_content is not None:  # type: ignore
                    content_chunk = delta.reasoning_content  # type: ignore

                    # If we already finished reasoning but the model is still
                    # sending data in the wrong slot, pivot it to TEXT.
                    if reasoning_ended:
                        accumulated_text += content_chunk
                        yield StreamResponse(type=StreamType.TEXT, delta=content_chunk)
                        continue

                    # Check for the transition tag inside the reasoning slot
                    if "</think>" in content_chunk:
                        parts = content_chunk.split("</think>", 1)

                        # Send the prefix to thinking
                        if parts[0]:
                            accumulated_thinking += parts[0]
                            yield StreamResponse(
                                type=StreamType.THINKING, delta=parts[0]
                            )

                        # Latch the state to 'Ended'
                        reasoning_ended = True
                        in_reasoning = False
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            full_thinking=accumulated_thinking.strip(),
                        )

                        # Send the suffix to text
                        if parts[1]:
                            accumulated_text += parts[1]
                            yield StreamResponse(type=StreamType.TEXT, delta=parts[1])
                    else:
                        # Normal thinking flow
                        if not in_reasoning:
                            in_reasoning = True
                        accumulated_thinking += content_chunk
                        yield StreamResponse(
                            type=StreamType.THINKING, delta=content_chunk
                        )
                if delta.content is not None:
                    if in_reasoning and not reasoning_ended:
                        reasoning_ended = True
                        in_reasoning = False
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            full_thinking=self._maybe_extract_hidden_state(
                                accumulated_thinking.strip()
                            ),
                        )
                    accumulated_text += delta.content
                    yield StreamResponse(type=StreamType.TEXT, delta=delta.content)

                if event.choices[0] and event.choices[0].finish_reason is not None:  # type: ignore
                    if in_reasoning:
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            full_thinking=self._maybe_extract_hidden_state(
                                accumulated_thinking.strip()
                            ),
                        )
                    yield StreamResponse(
                        type=StreamType.TEXT_END,
                        delta="",
                        full_text=accumulated_text.strip(),
                    )

        except openai.APIError:
            _log.error("API error | provider=openai")

    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:

        messages_dicts: list[dict[str, Any]] = [
            self._message_to_dict(msg) for msg in messages
        ]

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages_dicts,  # type: ignore
            max_tokens=config.max_tokens,  # type: ignore
            temperature=config.temperature,  # type: ignore
            top_p=config.top_p,  # type: ignore
            extra_body=self._thinking_extra_body(reasoning),
        )

        if reasoning:
            reasoning_content = completion.choices[0].message.reasoning_content  # type: ignore
            if reasoning_content is not None:
                _log.debug("Reasoning | provider=openai-compat | chars=%d", len(reasoning_content))
                _sql_logger.log_reasoning(provider="openai-compat", content=reasoning_content)

        return completion.choices[0].message.content or ""  # type: ignore

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        encoding = tiktoken.encoding_for_model(
            "gpt-4o"
        )  # DeepSeek uses similar tokenization
        return len(encoding.encode(text))

    def _to_pydantic_ai_model(self, task: LLMTask, reasoning: bool = False) -> Model:
        # Uses the same _thinking_extra_body() as our own _execute_chat_completion(_stream)
        # so the "thinking" toggle can't drift between the raw and pydantic_ai paths.
        #
        # Uses pydantic_ai's dedicated DeepSeekProvider rather than a generic
        # OpenAIProvider pointed at DeepSeek's base_url — verified live that
        # this matters: with the generic provider, deepseek-reasoner returns
        # a 400 ("Thinking mode does not support this tool_choice") on any
        # structured-output call, because our output_type forces a specific
        # tool_choice and the generic OpenAI profile assumes that's safe.
        # DeepSeekProvider's model_profile() sets
        # openai_supports_tool_choice_required=False for deepseek-reasoner
        # specifically so pydantic_ai falls back to tool_choice='auto'
        # instead of forcing it — which is exactly what's needed here.
        #
        # NOTE: tool_choice='auto' means the model is no longer *forced* to
        # call the structured-output tool — deepseek-reasoner has reliably
        # chosen to call it anyway in live testing, but this is "best
        # effort" compliance, not a hard guarantee, the same trade-off
        # LLMClientOpenRouter's _needs_prompted_output_for_reasoning makes
        # explicit via PromptedOutput. If SQL call logs (LogType.* tables) start
        # showing a real increase in pydantic_ai validation retries for
        # this client, that's the signal to stop forcing reasoning=True
        # here rather than silently eating the extra round trips.
        settings = self._to_pydantic_ai_settings(self.get_task_config(task))
        settings["extra_body"] = self._thinking_extra_body(reasoning)
        return OpenAIChatModel(
            self._model,
            provider=DeepSeekProvider(api_key=self._client.api_key),
            settings=settings,
        )


# --- Claude & MiniMax Client (Anthropic SDK) ---


class LLMClientAnthropicBase(LLMClientBase):
    """
    Shared base for Anthropic-compatible providers (Claude, MiniMax).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        config: LLMConfig | None = None,
    ):
        super().__init__(config)
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._base_url = base_url

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[MessageParam]]:
        """
        Converts a list of messages to the format required by the Anthropic API.

        This is were we would feed back thinking for interleaved thinking:
        thinking = ThinkingBlock(
            signature=final_message.content[0].signature,
            thinking=final_message.content[0].thinking,
            type="thinking",
            citations=None,
            text=None
        )

        text = TextBlock(text=final_message.content[1].text, type="text")
        content = [thinking, text]

        Args:
            messages (list[Message]): List of messages to be converted.

        Returns:
            list[MessageParam]: List of MessageParam objects representing the messages.
        """

        messages_converted = []
        # Mypy being stupid
        raw_policy = self.get_logic_config().keep_thinking_turns
        policy = (
            raw_policy
            if isinstance(raw_policy, ThinkingFeebackPolicy)
            else ThinkingFeebackPolicy.never()
        )

        turn_map = self._build_turn_map(messages)

        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                # System messages usually aren't sent in the 'messages' list
                # for most APIs (Anthropic/OpenAI), they go in a top-level param.
                continue

            if msg.role in MessageRole.USER:
                messages_converted.append(
                    MessageParam(
                        content=[TextBlock(text=msg.content.text, type="text")],
                        role="user",
                    )
                )

            elif msg.role in MessageRole.ASSISTANT:
                turns_ago = turn_map[i]

                content: list[ThinkingBlock | TextBlock] = []
                if policy.should_keep(turns_ago):
                    content.append(
                        ThinkingBlock(
                            signature=msg.content.thinking_signature or "",
                            thinking=msg.content.thinking or "",
                            type="thinking",
                            citations=None,
                            text=None,
                        ),
                    )

                content.append(TextBlock(text=msg.content.text, type="text"))

                messages_converted.append(
                    MessageParam(
                        content=content,
                        role="assistant",
                    )
                )

        system_instruction = next(
            (msg.content.text for msg in messages if msg.role == MessageRole.SYSTEM),
            "You're an helpful assistant",
        )
        return system_instruction, messages_converted

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        sys_inst, messages = self._convert_messages(messages)

        thinking_cfg = (
            ThinkingConfigEnabledParam(
                type="enabled", budget_tokens=int(config.max_tokens / 2)  # type: ignore
            )
            if reasoning
            else ThinkingConfigDisabledParam(type="disabled")
        )

        with self._client.messages.stream(
            model=self._model,
            max_tokens=config.max_tokens,  # type: ignore
            temperature=(1.0 if reasoning else config.temperature),  # type: ignore
            thinking=thinking_cfg,
            system=sys_inst,
            messages=messages,
        ) as stream:
            for event in stream:
                # Capture Thinking Start/End
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        yield StreamResponse(type=StreamType.THINKING, delta="")
                elif event.type == "content_block_stop":
                    if event.content_block.type == "thinking":
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            signature=event.content_block.signature,
                            full_thinking=event.content_block.thinking,
                        )
                    if event.content_block.type == "text":
                        yield StreamResponse(
                            type=StreamType.TEXT_END,
                            delta="",
                            full_text=event.content_block.text,
                        )
                # Emit deltas for streaming
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        yield StreamResponse(
                            type=StreamType.THINKING, delta=event.delta.thinking
                        )
                    elif event.delta.type == "text_delta":
                        yield StreamResponse(
                            type=StreamType.TEXT, delta=event.delta.text
                        )
                # 3. Handle the end of the message (optional)
                elif event.type == "message_stop":
                    break

            # To feed thinking back in subsequent turns (required for MiniMax M2.1 and
            # Claude extended thinking), reconstruct the content list from the final message:
            #   final_message = stream.get_final_message()
            #   content = [
            #     ThinkingBlock(signature=..., thinking=..., type="thinking", citations=None, text=None),
            #     TextBlock(text=..., type="text"),
            #   ]
            # citations=None and text=None are required for ThinkingBlock equality with the API response.
            # The reconstructed list can then be passed as the assistant turn's content in the next call.

    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:
        sys_inst, msgs = self._convert_messages(messages)

        thinking_cfg = (
            ThinkingConfigEnabledParam(
                type="enabled", budget_tokens=int(config.max_tokens / 2)  # type: ignore
            )
            if reasoning
            else ThinkingConfigDisabledParam(type="disabled")
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=config.max_tokens,  # type: ignore
            temperature=config.temperature if not reasoning else 1.0,  # type: ignore
            thinking=thinking_cfg,
            system=sys_inst,
            messages=msgs,
        )

        if reasoning:
            thinking_content = next(
                (c.thinking for c in response.content if c.type == "thinking"), None
            )
            if thinking_content:
                _log.debug("Reasoning | provider=claude | chars=%d", len(thinking_content))
                _sql_logger.log_reasoning(provider="claude", content=thinking_content)

        return next(c.text for c in response.content if c.type == "text")

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def _to_pydantic_ai_model(self, task: LLMTask, reasoning: bool = False) -> Model:
        # Unified pydantic_ai setting — translates to extended thinking with
        # a sensible default budget, same intent as the
        # ThinkingConfigEnabledParam our own _execute_chat_completion builds
        # when reasoning=True.
        #
        # NOTE: Anthropic's real API rejects combining extended thinking
        # with a *forced* tool call, which is what our output_type=SomeModel
        # calls normally do. AnthropicModel already handles this internally
        # (auto-switches to native/prompted output instead of forcing a
        # tool call when it detects the conflict) — verified
        # _needs_prompted_output_for_reasoning can stay False here, unlike
        # LLMClientOpenRouter.
        # That auto-switch is itself a "best effort" structured-output mode
        # rather than the (slightly more reliable) forced-tool-call default.
        # If SQL call logs (LogType.* tables) start showing a real increase
        # in pydantic_ai validation retries for this client, that's the
        # signal to stop forcing reasoning=True here rather than silently
        # eating the extra round trips.
        settings = self._to_pydantic_ai_settings(self.get_task_config(task))
        if reasoning:
            settings["thinking"] = True
        return AnthropicModel(
            self._model,
            provider=AnthropicProvider(
                api_key=self._client.api_key, base_url=self._base_url
            ),
            settings=settings,
        )


class LLMClientClaude(LLMClientAnthropicBase):
    """
    LLMClient implementation for Claude using the Anthropic SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        config: LLMConfig | None = None,
    ):
        super().__init__(api_key, model, config=config)


class LLMClientMiniMax(LLMClientAnthropicBase):
    """
    LLMClient implementation for MiniMax using the Anthropic SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.1",
        config: LLMConfig | None = None,
    ):
        super().__init__(
            api_key, model, base_url="https://api.minimax.io/anthropic", config=config
        )


class LLMClientOpenRouter(LLMClientBase):
    """
    LLMClient implementation for OpenRouter API.

    OpenRouter provides a unified API for multiple LLM providers with normalized
    reasoning token support. This client handles:

    1. Reasoning configuration (effort, max_tokens, exclude)
    2. Reasoning details parsing (text, summary, encrypted types)
    3. Preserving reasoning blocks for multi-turn conversations
    4. Streaming with SSE parsing

    Model Examples:
        - "deepseek/deepseek-r1" - DeepSeek reasoning model
        - "anthropic/claude-sonnet-4" - Claude with reasoning
        - "openai/gpt-5-mini" - OpenAI reasoning model
        - "google/gemini-flash-thinking" - Gemini thinking model
    """

    # Unlike LLMClientAnthropicBase/LLMClientDeepSeek, OpenRouterModel has no
    # internal handling for the thinking-vs-forced-tool-call conflict (see
    # _to_pydantic_ai_model() below) — the upstream provider's profile
    # functions it reuses (e.g. anthropic_model_profile) don't carry that
    # protection, only the dedicated AnthropicModel/DeepSeekProvider classes
    # do, and OpenRouterModel is neither. So build_agent() swaps to
    # PromptedOutput when reasoning is on, to sidestep tool-calling entirely
    # instead of forcing a tool call the downstream model may have to
    # silently drop thinking to honor. Verified live: still produces
    # correct, fully-validated output (same pydantic validation/retry loop
    # as the default tool-calling mode), just via a prompted JSON
    # instruction instead of a tool call — a "best effort" compliance mode
    # rather than a hard guarantee, so if retry rates climb, that's the
    # trade-off being made here.
    _needs_prompted_output_for_reasoning = True

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        config: LLMConfig | None = None,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "anthropic/claude-sonnet-4")
            config: Optional LLM configuration
        """
        super().__init__(config=config, model_name=model)
        self._model = model
        self._base_url = "https://openrouter.ai/api/v1"
        self._chat_url = f"{self._base_url}/chat/completions"
        self._api_key = api_key

        # Build headers
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _to_pydantic_ai_model(self, task: LLMTask, reasoning: bool = False) -> Model:
        # Uses pydantic_ai's dedicated OpenRouterModel/OpenRouterProvider
        # rather than a generic OpenAIChatModel + OpenAIProvider pointed at
        # OpenRouter's base_url — OpenRouter's actual reasoning toggle is
        # `extra_body['reasoning'] = {"effort": ..., "enabled": ...}`, not
        # OpenAI's `reasoning_effort` parameter, and OpenRouterModel is the
        # one pydantic_ai class that overrides _translate_thinking() to
        # build that shape from the unified `thinking` setting. Plain
        # OpenAIChatModel would silently send `reasoning_effort`, which
        # OpenRouter's API doesn't use for this and would just ignore.
        settings = self._to_pydantic_ai_settings(self.get_task_config(task))
        if reasoning:
            settings["thinking"] = True
        return OpenRouterModel(
            self._model,
            provider=OpenRouterProvider(api_key=self._api_key),
            settings=settings,
        )

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal Message format to OpenRouter API format.

        This handles:
        - System message extraction
        - Preserving reasoning_details for multi-turn reasoning
        - Applying thinking feedback policy

        Args:
            messages: List of messages in internal format

        Returns:
            Tuple of (system_message, converted_messages)
        """
        converted = []
        system_message = None

        # Get thinking feedback policy
        raw_policy = self.get_logic_config().keep_thinking_turns
        policy = (
            raw_policy
            if isinstance(raw_policy, ThinkingFeebackPolicy)
            else ThinkingFeebackPolicy.never()
        )

        turn_map = self._build_turn_map(messages)

        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content.text
                continue

            if msg.role == MessageRole.USER:
                converted.append(
                    {
                        "role": "user",
                        "content": msg.content.text,
                    }
                )

            elif msg.role == MessageRole.ASSISTANT:
                turns_ago = turn_map[i]

                # Build assistant message
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content.text,
                }

                # Add reasoning_details if policy allows
                # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#example-preserving-reasoning-blocks-with-openrouter-and-claude
                if policy.should_keep(turns_ago) and msg.content.thinking:
                    # Reconstruct reasoning_details in OpenRouter format
                    if self.model_name == "anthropic/claude-sonnet-4":
                        if msg.content.reasoning_details:
                            # Use the exact reasoning_details from the original response
                            assistant_msg["reasoning_details"] = (
                                msg.content.reasoning_details
                            )
                        elif msg.content.thinking:
                            # Fallback: reconstruct for backward compatibility
                            assistant_msg["reasoning_details"] = [
                                {
                                    "type": "reasoning.text",
                                    "text": msg.content.thinking,
                                    "signature": msg.content.thinking_signature,
                                    "id": f"reasoning-{i}",
                                    "format": "anthropic-claude-v1",
                                }
                            ]
                    else:
                        raise ValueError(
                            f"No reasoning feedback implemented for model {self.model_name}"
                        )

                converted.append(assistant_msg)

        return system_message, converted

    def _build_reasoning_config(
        self, reasoning: bool, config: LLMConfig
    ) -> dict[str, Any] | None:
        """
        Build OpenRouter reasoning configuration.

        Args:
            reasoning: Whether reasoning is enabled
            config: LLM configuration with max_tokens

        Returns:
            Reasoning config dict or None if reasoning disabled
        """
        if not reasoning:
            return None

        # Use max_tokens to determine reasoning budget
        # Allocate up to 50% for reasoning (medium effort)
        max_tokens: int = (
            config.max_tokens if isinstance(config.max_tokens, int) else 4096
        )
        reasoning_max_tokens = min(max_tokens // 2, 32000)

        return {
            "max_tokens": max(reasoning_max_tokens, 1024),  # Min 1024 tokens
            "exclude": False,  # Include reasoning in response by default
        }

    def _parse_reasoning_details(
        self, reasoning_details: list[dict[str, Any]] | None
    ) -> tuple[str, str | None]:
        """
        Parse OpenRouter reasoning_details array into thinking text and signature.

        OpenRouter returns reasoning in different formats:
        - reasoning.text: Raw thinking text
        - reasoning.summary: High-level summary
        - reasoning.encrypted: Protected/redacted reasoning

        Args:
            reasoning_details: Array of reasoning detail objects

        Returns:
            Tuple of (thinking_text, signature)
        """
        if not reasoning_details:
            return "", None

        thinking_parts = []
        signature = None

        for detail in reasoning_details:
            detail_type = detail.get("type", "")

            if detail_type == "reasoning.text":
                text = detail.get("text", "")
                if text:
                    thinking_parts.append(text)
                # Use signature from first reasoning.text block
                if signature is None:
                    signature = detail.get("signature")

            elif detail_type == "reasoning.summary":
                summary = detail.get("summary", "")
                if summary:
                    thinking_parts.append(f"[Summary: {summary}]")

            elif detail_type == "reasoning.encrypted":
                # Encrypted/redacted reasoning
                thinking_parts.append("[REDACTED]")

        thinking_text = "\n".join(thinking_parts)

        return thinking_text, signature

    def _build_request_payload(
        self,
        converted_msgs: list[dict[str, Any]],
        system_msg: str | None,
        config: LLMConfig,
        reasoning: bool,
        stream: bool,
    ) -> dict[str, Any]:
        # OpenRouter puts system in the messages array (same as OpenAI), not a top-level param
        messages: list[dict[str, Any]] = list(converted_msgs)
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": stream,
        }
        reasoning_config = self._build_reasoning_config(reasoning, config)
        if reasoning_config:
            payload["reasoning"] = reasoning_config
        return payload

    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:
        system_msg, converted_msgs = self._convert_messages(messages)
        payload = self._build_request_payload(
            converted_msgs, system_msg, config, reasoning, stream=False
        )

        response = requests.post(
            self._chat_url, headers=self._headers, json=payload, timeout=600
        )
        response.raise_for_status()
        result = response.json()

        choice = result["choices"][0]
        message = choice["message"]

        if reasoning and "reasoning_details" in message:
            thinking, _ = self._parse_reasoning_details(message.get("reasoning_details"))
            if thinking:
                _log.debug("Reasoning | provider=openrouter | chars=%d", len(thinking))
                _sql_logger.log_reasoning(provider="openrouter", content=thinking)

        return message.get("content", "")  # type: ignore

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        """
        Handles OpenRouter's streaming format where reasoning_details come in delta
        chunks that need to be accumulated and may interleave with content deltas.
        """
        system_msg, converted_msgs = self._convert_messages(messages)
        payload = self._build_request_payload(
            converted_msgs, system_msg, config, reasoning, stream=True
        )

        # Make streaming request
        response = requests.post(
            self._chat_url,
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        # Parse SSE stream
        client = SSEClient(response)  # type: ignore

        # Accumulate reasoning and text
        accumulated_thinking = ""
        accumulated_text = ""
        thinking_signature = None
        in_reasoning = False
        reasoning_ended = False
        # Accumulate reasoning_details by consolidating deltas into complete blocks
        # Key: (type, index) -> reasoning detail object
        reasoning_blocks: dict[tuple[str, int], dict[str, Any]] = {}

        for event in client.events():
            if event.data == "[DONE]":
                break

            try:
                chunk = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if not chunk.get("choices"):
                continue

            delta = chunk["choices"][0].get("delta", {})

            # Check if we have reasoning_details in this delta
            has_reasoning = "reasoning_details" in delta and delta["reasoning_details"]
            has_content = "content" in delta and delta["content"]

            # Handle reasoning_details in delta
            if has_reasoning:
                reasoning_details = delta["reasoning_details"]

                # Start of reasoning (first time we see reasoning_details)
                if not in_reasoning:
                    in_reasoning = True
                    reasoning_ended = False
                    yield StreamResponse(type=StreamType.THINKING, delta="")

                    # Consolidate reasoning_details deltas into complete blocks
                for detail in reasoning_details:
                    detail_type = detail.get("type", "reasoning.text")
                    detail_index = detail.get("index", 0)
                    key = (detail_type, detail_index)

                    if key not in reasoning_blocks:
                        # Initialize new block with all available fields
                        reasoning_blocks[key] = {
                            "type": detail_type,
                            "index": detail_index,
                            "format": detail.get("format", "unknown"),
                            "id": detail.get("id"),
                        }

                        # Initialize content fields based on type
                        if detail_type == "reasoning.text":
                            reasoning_blocks[key]["text"] = ""
                            reasoning_blocks[key]["signature"] = None
                        elif detail_type == "reasoning.summary":
                            reasoning_blocks[key]["summary"] = ""
                        elif detail_type == "reasoning.encrypted":
                            reasoning_blocks[key]["data"] = ""

                    # Update the block with delta content
                    if detail_type == "reasoning.text":
                        if "text" in detail:
                            reasoning_blocks[key]["text"] += detail["text"]
                        if "signature" in detail and detail["signature"]:
                            reasoning_blocks[key]["signature"] = detail["signature"]
                    elif detail_type == "reasoning.summary":
                        if "summary" in detail:
                            reasoning_blocks[key]["summary"] += detail["summary"]
                    elif detail_type == "reasoning.encrypted":
                        if "data" in detail:
                            reasoning_blocks[key]["data"] += detail["data"]

                    # Update id if provided
                    if "id" in detail and detail["id"]:
                        reasoning_blocks[key]["id"] = detail["id"]

                # Parse and accumulate reasoning
                thinking_delta, sig = self._parse_reasoning_details(reasoning_details)
                if thinking_delta:
                    accumulated_thinking += thinking_delta
                    yield StreamResponse(
                        type=StreamType.THINKING,
                        delta=thinking_delta,
                    )
                if sig and thinking_signature is None:
                    thinking_signature = sig

            # If we were in reasoning but this delta has NO reasoning_details,
            # it means reasoning has ended
            if in_reasoning and not has_reasoning and not reasoning_ended:
                in_reasoning = False
                reasoning_ended = True

                # Convert reasoning_blocks dict to list for reasoning_details
                final_reasoning_details = list(reasoning_blocks.values())

                yield StreamResponse(
                    type=StreamType.THINKING_END,
                    delta="",
                    signature=thinking_signature,
                    full_thinking=accumulated_thinking,
                    reasoning_details=final_reasoning_details,
                )

            # Handle content delta (can coexist with reasoning_details!)
            if has_content:
                content_delta = delta["content"]
                accumulated_text += content_delta
                yield StreamResponse(
                    type=StreamType.TEXT,
                    delta=content_delta,
                )

            # Handle finish_reason
            finish_reason = chunk["choices"][0].get("finish_reason")
            if finish_reason:
                # Emit final thinking if we never emitted THINKING_END
                if in_reasoning and not reasoning_ended:
                    # Convert reasoning_blocks dict to list for reasoning_details
                    final_reasoning_details = list(reasoning_blocks.values())
                    yield StreamResponse(
                        type=StreamType.THINKING_END,
                        delta="",
                        signature=thinking_signature,
                        full_thinking=accumulated_thinking,
                        reasoning_details=final_reasoning_details,
                    )

                # Emit final text
                yield StreamResponse(
                    type=StreamType.TEXT_END,
                    delta="",
                    full_text=accumulated_text,
                )
                break

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        OpenRouter doesn't provide a token counting endpoint, so we use
        a simple approximation based on character count.

        For more accurate counting, consider using tiktoken for OpenAI models
        or the respective tokenizer for other providers.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4


# class LLMClientGemini(LLMClientBase):
#     """
#     LLMClient implementation using Google Gemini via the google-genai SDK.
#     """

#     def __init__(
#         self,
#         api_key: str,
#         model: str = "gemini-3-flash-preview",
#         config: LLMConfig | None = None,
#     ):
#         super().__init__(config)
#         self._client = genai.Client(api_key=api_key)
#         self._model = model

#     def _generate_contents(
#         self, messages: list[dict[str, str]]
#     ) -> tuple[str, list[Content]]:
#         """
#         Generates content using the Google Gemini model.
#         Args:
#             messages (list[dict[str, str]]): List of messages to be sent to the model.

#         Returns:
#             str: The system instruction extracted from the messages.
#             list[Content]: List of content objects representing user and model messages.
#         """
#         try:
#             system_instruction = next(m for m in messages if m["role"] == "system")[
#                 "content"
#             ]
#         except StopIteration:
#             system_instruction = "You're an helpful assistant"

#         contents: list[Content] = []
#         for m in messages:
#             if m["role"] == "user":
#                 contents.append(UserContent(m["content"]))
#             elif m["role"] == "assistant":
#                 contents.append(ModelContent(m["content"]))
#         return system_instruction, contents

#     def _get_gen_config(
#         self, system_instruction: str, reasoning: bool, config: LLMConfig
#     ) -> GenerateContentConfig:
#         thinking_cfg = (
#             None
#             if reasoning
#             else ThinkingConfig(thinking_budget=0, include_thoughts=False)
#         )
#         return GenerateContentConfig(
#             system_instruction=system_instruction,
#             max_output_tokens=config.max_tokens,
#             temperature=config.temperature,
#             top_p=config.top_p,
#             top_k=config.top_k,
#             thinking_config=thinking_cfg,
#         )

#     def _execute_chat_completion_stream(
#         self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
#     ) -> Generator[str, None, None]:
#         sys_inst, contents = self._generate_contents(messages)
#         gen_config = self._get_gen_config(sys_inst, reasoning, config)

#         response = self._client.models.generate_content_stream(
#             model=self._model, contents=contents, config=gen_config
#         )
#         for chunk in response:
#             yield chunk.text or ""

#     def _execute_chat_completion(
#         self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
#     ) -> str:
#         sys_inst, contents = self._generate_contents(messages)
#         gen_config = self._get_gen_config(sys_inst, reasoning, config)

#         response = self._client.models.generate_content(
#             model=self._model, contents=contents, config=gen_config
#         )
#         return response.text or ""

#     def count_tokens(self, text: str) -> int:
#         """
#         Counts the number of tokens in a given text by API call.

#         Args:
#             text (str): The text to count the tokens in.

#         Returns:
#             int: The number of tokens in the text.

#         """
#         try:
#             enc = tiktoken.encoding_for_model(self._model)
#         except KeyError:
#             enc = tiktoken.get_encoding("cl100k_base")
#         return len(enc.encode(text))

#     def stop_generation(self) -> None:
#         """
#         Stops the generation process.
#         """
