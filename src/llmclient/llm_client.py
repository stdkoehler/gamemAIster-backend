"""LLM Client"""

import json
import time

from typing import Generator, Any
from dataclasses import asdict, dataclass
from enum import StrEnum
from abc import ABC, abstractmethod

from urllib.parse import urljoin

import requests
from sseclient import SSEClient
import openai
import tiktoken
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    Content,
    UserContent,
    ModelContent,
    ThinkingConfig,
)
import anthropic
from anthropic.types import (
    MessageParam,
    ThinkingConfigEnabledParam,
    ThinkingConfigDisabledParam,
    ThinkingBlock,
    TextBlock,
)

from src.llmclient.llm_config_registry import ConfigRegistry, LLMTask
from src.llmclient.llm_parameters import LLMConfig, LLMLogicConfig


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

    def adjust_reasoning_mistral24b(
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

    def adjust_reasoning_gemma3_r1(
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

    def adjust_reasoning(
        self, messages: list[Message], payload: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any]]:
        """different models need different reasoning adjustments"""
        if self.model_name == "mistral-24b-hermes":
            messages, payload = self.adjust_reasoning_mistral24b(messages, payload)
        elif self.model_name == "gemma-3-r1-27b":
            messages, payload = self.adjust_reasoning_gemma3_r1(messages, payload)

        return messages, payload

    def message_to_dict(self, message: Message) -> dict[str, str]:
        """Convert Message object to dictionary format for API compatibility."""
        return {"role": message.role.value, "content": message.content.text}

    def _execute_chat_completion_stream(
        self, messages: list[Message], reasoning: bool, config: LLMConfig
    ) -> Generator[StreamResponse, None, None]:
        payload = asdict(config)

        if reasoning:
            messages, payload = self.adjust_reasoning(messages, payload)

        payload.update(
            {
                "messages": [self.message_to_dict(msg) for msg in messages],
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

                        # Send the full thinking object
                        yield StreamResponse(
                            type=StreamType.THINKING_END,
                            delta="",
                            full_thinking=accumulated_thinking.strip(),
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
            messages, payload = self.adjust_reasoning(messages, payload)

        payload.update(
            {
                "messages": [self.message_to_dict(msg) for msg in messages],
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
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        config: LLMConfig | None = None,
    ):
        super().__init__(config)
        self._client = openai.OpenAI(
            base_url="https://api.deepseek.com", api_key=api_key
        )
        self._model = model

    def _execute_chat_completion_stream(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> Generator[str, None, None]:
        try:

            stream_response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                max_tokens=config.max_tokens,  # type: ignore
                temperature=config.temperature,  # type: ignore
                top_p=config.top_p,  # type: ignore
                stream=True,
            )
            for event in stream_response:
                content = event.choices[0].delta.content  # type: ignore
                yield content if content is not None else ""
        except openai.APIError:
            print("Api Error")

    def _execute_chat_completion(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> str:
        response = None
        while response is None:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                max_tokens=config.max_tokens,  # type: ignore
                temperature=config.temperature,  # type: ignore
                top_p=config.top_p,  # type: ignore
            )
            try:
                response = completion.choices[0].message.content
                # DeepSeek specific reasoning extraction
                if (
                    hasattr(completion.choices[0].message, "model_extra")
                    and completion.choices[0].message.model_extra
                ):
                    reasoning_text = completion.choices[0].message.model_extra.get(
                        "reasoning_content"
                    )
                    if reasoning_text:
                        print(f"### Reasoning\n{reasoning_text}")
            except (TypeError, IndexError):
                print("Empty LLM response, retrying...")
                time.sleep(2)
        return response or ""

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


# --- 2. Gemini Client (Google GenAI SDK) ---


class LLMClientGemini(LLMClientBase):
    """
    LLMClient implementation using Google Gemini via the google-genai SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        config: LLMConfig | None = None,
    ):
        super().__init__(config)
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _generate_contents(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[Content]]:
        """
        Generates content using the Google Gemini model.
        Args:
            messages (list[dict[str, str]]): List of messages to be sent to the model.

        Returns:
            str: The system instruction extracted from the messages.
            list[Content]: List of content objects representing user and model messages.
        """
        try:
            system_instruction = next(m for m in messages if m["role"] == "system")[
                "content"
            ]
        except StopIteration:
            system_instruction = "You're an helpful assistant"

        contents: list[Content] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(UserContent(m["content"]))
            elif m["role"] == "assistant":
                contents.append(ModelContent(m["content"]))
        return system_instruction, contents

    def _get_gen_config(
        self, system_instruction: str, reasoning: bool, config: LLMConfig
    ) -> GenerateContentConfig:
        thinking_cfg = (
            None
            if reasoning
            else ThinkingConfig(thinking_budget=0, include_thoughts=False)
        )
        return GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            thinking_config=thinking_cfg,
        )

    def _execute_chat_completion_stream(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> Generator[str, None, None]:
        sys_inst, contents = self._generate_contents(messages)
        gen_config = self._get_gen_config(sys_inst, reasoning, config)

        response = self._client.models.generate_content_stream(
            model=self._model, contents=contents, config=gen_config
        )
        for chunk in response:
            yield chunk.text or ""

    def _execute_chat_completion(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> str:
        sys_inst, contents = self._generate_contents(messages)
        gen_config = self._get_gen_config(sys_inst, reasoning, config)

        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=gen_config
        )
        return response.text or ""

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

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
        Args:
            messages (list[Message]): List of messages to be converted.

        Returns:
            list[MessageParam]: List of MessageParam objects representing the messages.
        """
        messages_converted = []
        # this is were we would feed back thinking for interleaved thinking:
        # test_thinking = ThinkingBlock(signature=final_message.content[0].signature, thinking=final_message.content[0].thinking, type="thinking", citations=None, text=None)
        # test_thinking == final_message.content[0] # is true only if we add citations=None and text=None
        # test_text = TextBlock(text=final_message.content[1].text, type="text")
        # test_text == final_message.content[1] # is true
        # test_content = [test_thinking, test_text]

        for msg in messages:
            if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                messages_converted.append(
                    MessageParam(
                        content=[TextBlock(text=msg.content.text, type="text")],
                        role="user" if msg.role == MessageRole.USER else "assistant",
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
