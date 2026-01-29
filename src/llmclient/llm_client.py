"""LLM Client"""

import json
import time

from typing import Generator, Any
from dataclasses import asdict
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
)

from src.llmclient.llm_parameters import LLMConfig


class LLMClientBase(ABC):
    """
    Abstract Base Class defining all methods required for an LLM Interface
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initializes client with a 'member config'.
        Inherits from system defaults + optional custom overrides.
        """
        base = LLMConfig.defaults()
        self.member_config = config.apply_to(base) if config else base

    @abstractmethod
    def _execute_chat_completion_stream(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> Generator[str, None, None]:
        """
        Internal abstract method: Child classes MUST implement the
        actual API call logic here.
        """

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        config_override: LLMConfig | None = None,
    ) -> Generator[str, None, None]:
        """
        Public method to execute a chat completion with optional config overrides.
        Merges member config with any provided overrides before execution.
        """
        active_config = (
            config_override.apply_to(self.member_config)
            if config_override
            else self.member_config
        )

        return self._execute_chat_completion_stream(messages, reasoning, active_config)

    @abstractmethod
    def _execute_chat_completion(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> str:
        """
        Internal abstract method: Child classes MUST implement the
        actual API call logic here.
        """

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        config_override: LLMConfig | None = None,
    ) -> str:
        """
        Public method to execute a chat completion with optional config overrides.
        Merges member config with any provided overrides before execution.
        """
        active_config = (
            config_override.apply_to(self.member_config)
            if config_override
            else self.member_config
        )

        return self._execute_chat_completion(messages, reasoning, active_config)

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

    def __init__(self, base_url: str, config: LLMConfig | None = None):
        super().__init__(config)
        self._base_url = base_url
        self._completion_url = urljoin(base_url, "/v1/completions")
        self._chat_completion_url = urljoin(base_url, "/v1/chat/completions")
        self._stop_generation_url = urljoin(base_url, "/v1/internal/stop-generation")
        self._token_url = urljoin(base_url, "/v1/internal/token-count")
        self._headers = {"Content-Type": "application/json"}

    def _prepare_payload(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> dict[str, Any]:
        if reasoning:
            messages = self.adjust_reasoning_messages(messages)

        payload = asdict(config)
        payload["messages"] = messages
        return payload

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

    def adjust_reasoning_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Adjusts the messages to include reasoning prompts if requested.
        Required for DeepHermes Mistral 24b
        """
        system_msg = next(
            (msg for msg in messages if msg["role"] == "system"), {"content": ""}
        )
        user_msgs = [msg for msg in messages if msg["role"] == "user"]

        # In-place safety for the user message content
        if user_msgs:
            user_msgs[0][
                "content"
            ] = f"{system_msg['content']}\n\n{user_msgs[0]['content']}"

        return [
            {
                "role": "system",
                "content": "You are a deep thinking AI. Enclose thoughts in <think> </think> tags.",
            }
        ] + user_msgs

    def _execute_chat_completion_stream(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> Generator[str, None, None]:
        if reasoning:
            messages = self.adjust_reasoning_messages(messages)

        payload = asdict(config)
        payload.update({"messages": messages, "stream": True})

        response = requests.post(
            self._chat_completion_url,
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=360,
        )
        response.raise_for_status()
        lines_gen = (line for line in response.iter_lines())
        client = SSEClient(lines_gen)

        for event in client.events():
            if event.data:
                payload = json.loads(event.data)
                chunk = payload["choices"][0]["delta"].get("content")
                if chunk:
                    yield chunk

    def _execute_chat_completion(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> str:
        if reasoning:
            messages = self.adjust_reasoning_messages(messages)

        payload = asdict(config)
        payload.update({"messages": messages, "stream": False})

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
        encoding = tiktoken.encoding_for_model(
            "gpt-4o"
        )  # DeepSeek uses similar tokenization
        return len(encoding.encode(text))

    def stop_generation(self) -> None:
        pass


# --- 2. Gemini Client (Google GenAI SDK) ---


class LLMClientGemini(LLMClientBase):
    """
    LLMClient implementation using Google Gemini via the google-genai SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        config: LLMConfig | None = None,
    ):
        super().__init__(config)
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _generate_contents(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[Content]]:
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
        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def stop_generation(self) -> None:
        pass


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
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[MessageParam]]:
        messages_converted = []
        for m in messages:
            if m["role"] in ["user", "assistant"]:
                messages_converted.append(MessageParam(content=m["content"], role=m["role"]))  # type: ignore

        system_instruction = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "You're an helpful assistant",
        )
        return system_instruction, messages_converted

    def _execute_chat_completion_stream(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
    ) -> Generator[str, None, None]:
        sys_inst, msgs = self._convert_messages(messages)

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
            temperature=(
                config.temperature if not reasoning else 1.0  # type: ignore
            ),  # Anthropic requires temp 1.0 for thinking
            thinking=thinking_cfg,
            system=sys_inst,
            messages=msgs,
        ) as stream:
            yield from stream.text_stream

    def _execute_chat_completion(
        self, messages: list[dict[str, str]], reasoning: bool, config: LLMConfig
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
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def stop_generation(self) -> None:
        pass


class LLMClientClaude(LLMClientAnthropicBase):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        config: LLMConfig | None = None,
    ):
        super().__init__(api_key, model, config=config)


class LLMClientMinMax(LLMClientAnthropicBase):
    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.1",
        config: LLMConfig | None = None,
    ):
        super().__init__(
            api_key, model, base_url="https://api.minimax.io/anthropic", config=config
        )
