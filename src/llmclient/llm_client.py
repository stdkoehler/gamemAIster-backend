"""LLM Client"""

import json
import re

from typing import Generator, Any
from dataclasses import asdict, dataclass
from enum import StrEnum
from abc import ABC, abstractmethod

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
)


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

    @abstractmethod
    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """


class LLMClientLocal(LLMClientBase):
    """
    LLMClient is a class that provides methods for interacting with the LLM API.

    Attributes:
        _base_url: The base URL of the LLM API.
        _completion_url: The URL for making completion requests.
        _token_url: The URL for counting tokens in a text.

    """

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

    def _maybe_extract_hidden_state(self, full_thinking: str) -> str:
        """
        Extracts the content of the <hidden_state> tag from the given text.

        Args:
            full_thinking (str): The text to extract the hidden state from.

        Returns:
            str: The content of the <hidden_state> tag if warmstart is used
        """
        if self.reasoning_warmstart is not None and full_thinking is not None:
            # Use findall to get every instance of content between <state> tags
            # re.DOTALL is crucial for multi-line content
            pattern = re.compile(
                r"<hidden_state>(.*?)</hidden_state>",
                re.DOTALL | re.IGNORECASE,
            )
            all_states = pattern.findall(full_thinking)

            if not all_states:
                full_thinking = ""

            # Join them with a newline or a specific separator
            # .strip() cleans up the whitespace from the model's output
            full_thinking = "\n".join([s.strip() for s in all_states if s.strip()])

        return full_thinking

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

        # In-place safety for the user message content, we add the system message to the
        # first user message because our actual system message needs to be the thinking
        # command
        if user_msgs:
            user_msgs[0].content.text = (
                f"{system_msg.content.text}\n\n{user_msgs[0].content.text}"
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

    def _message_to_dict(self, message: Message) -> dict[str, str]:
        """Convert Message object to dictionary format for API compatibility."""
        return {"role": message.role.value, "content": message.content.text}

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
            result = json.loads(event.data)
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
        text = result["choices"][0]["message"]["content"]
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


# --- 1. DeepSeek Client (OpenAI-Compatible SDK) ---


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
    ):
        super().__init__(config)
        self._client = openai.OpenAI(
            base_url="https://api.deepseek.com/beta", api_key=api_key
        )
        self._model = model

    def _maybe_extract_hidden_state(self, full_thinking: str) -> str:
        """
        Extracts the content of the <hidden_state> tag from the given text.

        Args:
            full_thinking (str): The text to extract the hidden state from.

        Returns:
            str: The content of the <hidden_state> tag if warmstart is used
        """
        if self.reasoning_warmstart is not None and full_thinking is not None:
            # Use findall to get every instance of content between <state> tags
            # re.DOTALL is crucial for multi-line content
            pattern = re.compile(
                r"<hidden_state>(.*?)</hidden_state>",
                re.DOTALL | re.IGNORECASE,
            )
            all_states = pattern.findall(full_thinking)

            if not all_states:
                full_thinking = ""

            # Join them with a newline or a specific separator
            # .strip() cleans up the whitespace from the model's output
            full_thinking = "\n".join([s.strip() for s in all_states if s.strip()])

        return full_thinking

    def _message_to_dict(self, message: Message) -> dict[str, str]:
        """Convert Message object to dictionary format for API compatibility."""
        return {"role": message.role.value, "content": message.content.text}

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
            if reasoning and self.reasoning_warmstart is not None:
                messages_dicts.append(
                    {
                        "role": "assistant",
                        "reasoning_content": self.reasoning_warmstart,
                        "prefix": True,
                    }
                )

            accumulated_thinking = ""
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
                extra_body=(
                    {"thinking": {"type": "enabled"}}
                    if reasoning
                    else {"thinking": {"type": "disabled"}}
                ),
            )
            for event in stream_response:
                delta = event.choices[0].delta  # type: ignore
                if delta.reasoning_content is not None:  # type: ignore
                    if not in_reasoning:
                        in_reasoning = True
                    accumulated_thinking += delta.reasoning_content  # type: ignore
                    yield StreamResponse(
                        type=StreamType.THINKING,
                        delta=delta.reasoning_content,  # type: ignore
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
            print("Api Error")

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
            extra_body=(
                {"thinking": {"type": "enabled"}}
                if reasoning
                else {"thinking": {"type": "disabled"}}
            ),
        )

        if reasoning:
            if completion.choices[0].message.reasoning_content is not None:  # type: ignore
                print(
                    f"### Reasoning\n{completion.choices[0].message.reasoning_content}"  # type: ignore
                )

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

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """


# --- 3. Claude & MiniMax Client (Anthropic SDK) ---


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

        # assistant messages to turn map for reasoning feedback policy
        # maps message index to "turns ago" for the assistant messages only, so we can
        # apply the feedback policy
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg.role == MessageRole.ASSISTANT
        ]
        num_assistant_msgs = len(assistant_indices)
        turn_map = {
            idx: (num_assistant_msgs - rank)
            for rank, idx in enumerate(assistant_indices)
        }

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

            # final_message.content is what we have to send back to MiniMaxM2.1 for multi-turn thinking
            # [
            #     ThinkingBlock(signature, thinking, type="thinking")
            #     TextBlock(text, type="text")
            # ]
            # test_thinking = ThinkingBlock(signature=final_message.content[0].signature, thinking=final_message.content[0].thinking, type="thinking", citations=None, text=None)
            # test_thinking == final_message.content[0] # is true only if we add citations=None and text=None
            # test_text = TextBlock(text=final_message.content[1].text, type="text")
            # test_text == final_message.content[1] # is true
            # test_content = [test_thinking, test_text]
            # test_content == final_message.content # is true
            # So we can reconstruct the final message from the content blocks
            final_message = stream.get_final_message()
            print(final_message)

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
                print(f"### Reasoning\n{thinking_content}")

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

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """


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
        self._api_key = api_key
        self._model = model
        self._base_url = "https://openrouter.ai/api/v1"
        self._chat_url = f"{self._base_url}/chat/completions"

        # Build headers
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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

        # Map assistant message indices to "turns ago"
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg.role == MessageRole.ASSISTANT
        ]
        num_assistant_msgs = len(assistant_indices)
        turn_map = {
            idx: (num_assistant_msgs - rank)
            for rank, idx in enumerate(assistant_indices)
        }

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

    def _execute_chat_completion(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> str:
        """
        Execute non-streaming chat completion.

        Args:
            messages: Conversation messages
            reasoning: Whether to enable reasoning
            config: Resolved LLM configuration

        Returns:
            Response text
        """
        system_msg, converted_msgs = self._convert_messages(messages)

        # Build payload
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": converted_msgs,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }

        # Add system message if present
        if system_msg:
            # OpenRouter uses the same format as OpenAI - system in messages array
            payload["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": system_msg,
                },
            )

        # Add reasoning config
        reasoning_config = self._build_reasoning_config(reasoning, config)
        if reasoning_config:
            payload["reasoning"] = reasoning_config

        # Make API request
        response = requests.post(
            self._chat_url,
            headers=self._headers,
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()

        # Extract response
        choice = result["choices"][0]
        message = choice["message"]

        # Parse reasoning if present
        if reasoning and "reasoning_details" in message:
            thinking, _ = self._parse_reasoning_details(
                message.get("reasoning_details")
            )
            if thinking:
                print(f"### Reasoning\n{thinking}")

        return message.get("content", "")  # type: ignore

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        """
        Execute streaming chat completion with reasoning support.

        This handles OpenRouter's streaming format where reasoning_details
        come in delta chunks that need to be accumulated.

        Args:
            messages: Conversation messages
            reasoning: Whether to enable reasoning
            config: Resolved LLM configuration

        Yields:
            StreamResponse objects for incremental updates
        """
        system_msg, converted_msgs = self._convert_messages(messages)

        # Build payload
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": converted_msgs,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
        }

        # Add system message if present
        if system_msg:
            payload["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": system_msg,
                },
            )

        # Add reasoning config
        reasoning_config = self._build_reasoning_config(reasoning, config)
        if reasoning_config:
            payload["reasoning"] = reasoning_config

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

    def stop_generation(self) -> None:
        """
        Stop generation is not supported by OpenRouter API.
        This is a no-op for API compatibility.
        """


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
