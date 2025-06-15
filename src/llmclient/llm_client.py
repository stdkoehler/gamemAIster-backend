"""LLM Client"""

import json
import time

from typing import Generator, Any, Literal
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

    @abstractmethod
    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> str:
        pass

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

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._completion_url = urljoin(base_url, "/v1/completions")
        self._chat_completion_url = urljoin(base_url, "v1/chat/completions")
        self._stop_generation_url = urljoin(base_url, "v1/internal/stop-generation")
        self._token_url = urljoin(base_url, "v1/internal/token-count")
        self._headers = {"Content-Type": "application/json"}

    def request(self, url: str, payload: dict[str, Any]) -> Any:
        """
        Sends a POST request to the specified URL with the given payload.

        Args:
            url (str): The URL to send the request to.
            payload (dict): The payload to include in the request body.

        Returns:
            dict: The JSON response from the request, parsed as a dictionary.

        """
        response = requests.post(url, headers=self._headers, json=payload, timeout=60)
        if response.status_code == 200:
            return json.loads(response.text)

    def adjust_reasoning_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Adjusts the messages to include reasoning prompts if requested.
        Required for DeepHermes Mistral 24b
        """
        system_msg = next(msg for msg in messages if msg["role"] == "system")
        user_msgs = [msg for msg in messages if msg["role"] == "user"]
        user_msgs[0]["content"] = (
            system_msg["content"] + "\n\n" + user_msgs[0]["content"]
        )
        return [
            {
                "role": "system",
                # "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.", # Hernes
                "content": "A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.\n\nYour thinking process must follow the template below:\n<think>\nYour thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.\n</think>\n\nHere, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.",  # Magistral
            }
        ] + user_msgs

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> Generator[str, None, None]:
        data = asdict(llm_config)
        data["messages"] = messages
        data["stream"] = True

        if reasoning:
            messages = self.adjust_reasoning_messages(messages=messages)

        stream_response = requests.post(
            self._chat_completion_url,
            headers=self._headers,
            json=data,
            stream=True,
            timeout=360,
        )
        client = SSEClient(stream_response)  # type: ignore

        for event in client.events():
            payload = json.loads(event.data)
            yield payload["choices"][0]["delta"]["content"]

    def completion_stream(
        self, prompt: str, llm_config: LLMConfig = LLMConfig()
    ) -> Generator[str, None, None]:
        data = asdict(llm_config)
        data["prompt"] = prompt
        data["stream"] = True

        stream_response = requests.post(
            self._completion_url,
            headers=self._headers,
            json=data,
            stream=True,
            timeout=360,
        )
        client = SSEClient(stream_response)  # type: ignore

        for event in client.events():
            payload = json.loads(event.data)
            yield payload["choices"][0]["text"]

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> str:

        if reasoning:
            messages = self.adjust_reasoning_messages(messages=messages)

        data = asdict(llm_config)
        data["messages"] = messages
        data["stream"] = False

        response = requests.post(
            self._chat_completion_url,
            headers=self._headers,
            json=data,
            stream=False,
            timeout=360,
        )
        text = json.loads(response.text)["choices"][0]["message"]["content"]
        return text if text is not None else ""

    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()) -> str:
        data = asdict(llm_config)
        data["prompt"] = prompt
        data["stream"] = False

        response = requests.post(
            self._completion_url,
            headers=self._headers,
            json=data,
            stream=False,
            timeout=60,
        )
        text = json.loads(response.text)["choices"][0]["text"]
        return text if text is not None else ""

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        payload = {"text": text}
        result = self.request(self._token_url, payload)
        return int(result["length"])

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """
        requests.post(
            self._stop_generation_url,
            timeout=60,
        )
        return None


class LLMClientDeepSeek(LLMClientBase):
    """
    LLMClient is a class that provides methods for interacting with the LLM API.

    Attributes:
        _base_url: The base URL of the LLM API.
        _completion_url: The URL for making completion requests.
        _token_url: The URL for counting tokens in a text.

    """

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self._client = openai.OpenAI(
            base_url="https://api.deepseek.com", api_key=api_key
        )
        self._model = model

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> Generator[str, None, None]:
        try:
            stream_response = self._client.chat.completions.create(
                extra_body={},
                model=self._model,
                messages=messages,  # type: ignore
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                stream=True,
            )
            for event in stream_response:
                content = event.choices[0].delta.content  # type: ignore
                yield content if content is not None else ""
        except openai.APIError:
            print("Api Error")

    def completion_stream(
        self, prompt: str, llm_config: LLMConfig = LLMConfig()
    ) -> Generator[str, None, None]:
        stream_response = self._client.completions.create(
            extra_body={},
            model=self._model,
            prompt=prompt,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            stream=True,
        )
        for event in stream_response:
            content = event.choices[0].delta.content  # type: ignore
            yield content if content is not None else ""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> str:

        response = None
        while response is None:
            completion = self._client.chat.completions.create(
                extra_body={},
                model=self._model,
                messages=messages,  # type: ignore
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
            )
            try:
                response = completion.choices[0].message.content
                if completion.choices[0].message.model_extra is not None:
                    try:
                        reasoning = completion.choices[0].message.model_extra[
                            "reasoning_content"
                        ]
                        print("### Reasoning")
                        print(reasoning)
                    except KeyError:
                        pass
            except TypeError:
                print("Empty LLM response")
                time.sleep(2)

        return response

    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()) -> str:

        response = None
        while response is None:
            completion = self._client.completions.create(
                extra_body={},
                model=self._model,
                prompt=prompt,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
            )
            try:
                response = completion.choices[0].text
            except TypeError:
                print("Empty LLM response")
        return response

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text by API call.

        Args:
            text (str): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.

        """
        encoding = tiktoken.encoding_for_model("gpt-4o")
        num_tokens = len(encoding.encode(text))
        print(len(text), num_tokens)
        return num_tokens

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """


class LLMClientGemini(LLMClientBase):
    """
    LLMClient implementation using Google Gemini via the google-genai SDK.
    """

    def __init__(
        self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17"
    ):  # adjust default as needed
        # Initialize the GenAI client with provided API key
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
            system_instruction = next(
                message for message in messages if message["role"] == "system"
            )["content"]
        except StopIteration:
            system_instruction = "You're an helpful assistant"

        contents: list[Content] = []
        for message in messages:
            if message["role"] == "user":
                contents.append(UserContent(message["content"]))
            elif message["role"] == "assistant":
                contents.append(ModelContent(message["content"]))

        return system_instruction, contents

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> Generator[str, None, None]:

        system_instruction, contents = self._generate_contents(messages=messages)

        # Reasoning is on by default
        if reasoning:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=GenerateContentConfig(system_instruction=system_instruction),
            )
        else:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    thinking_config=ThinkingConfig(
                        thinking_budget=0,
                        include_thoughts=False,
                    ),
                ),
            )

        for chunk in response:
            yield chunk.text or ""

    def completion_stream(
        self, prompt: str, llm_config: LLMConfig = LLMConfig()
    ) -> Generator[str, None, None]:

        response = self._client.models.generate_content_stream(
            model=self._model,
            contents=[UserContent(prompt)],
            config=GenerateContentConfig(
                system_instruction="You're an helpful assistant",
            ),
        )

        for chunk in response:
            yield chunk.text or ""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> str:

        system_instruction, contents = self._generate_contents(messages=messages)

        # Reasoning is on by default
        if reasoning:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=GenerateContentConfig(system_instruction=system_instruction),
            )
        else:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    thinking_config=ThinkingConfig(
                        thinking_budget=0,
                        include_thoughts=False,
                    ),
                ),
            )

        return response.text if response.text is not None else ""

    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()) -> str:

        response = self._client.models.generate_content(
            model=self._model,
            contents=[UserContent(prompt)],
            config=GenerateContentConfig(
                system_instruction="You're an helpful assistant",
            ),
        )

        return response.text if response.text is not None else ""

    def count_tokens(self, text: str) -> int:
        # Count tokens using tiktoken for the specified model
        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def stop_generation(self) -> None:
        # No direct stop support in google-genai; implement no-op or track cancellation
        pass


class LLMClientClaude(LLMClientBase):
    """
    LLMClient implementation using Google Gemini via the google-genai SDK.
    """

    def __init__(
        self, api_key: str, model: str = "claude-3-5-haiku-latest"
    ):  # adjust default as needed
        # Initialize the GenAI client with provided API key
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[MessageParam]]:
        """
        Converts a list of messages to the format required by the Gemini API.
        Args:
            messages (list[dict[str, str]]): List of messages to be converted.

        Returns:
            list[MessageParam]: List of MessageParam objects representing the messages.
        """
        messages_converted = []
        role: Literal["user", "assistant"]
        for message in messages:
            content = message["content"]
            if message["role"] == "user":
                role = "user"
            elif message["role"] == "assistant":
                role = "assistant"
            else:
                continue
            messages_converted.append(MessageParam(content=content, role=role))

        try:
            system_instruction = next(
                msg["content"] for msg in messages if msg["role"] == "system"
            )
        except StopIteration:
            system_instruction = "You're an helpful assistant"

        return system_instruction, messages_converted

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> Generator[str, None, None]:

        system_instruction, messages = self._convert_messages(messages=messages)

        if reasoning and not (
            self._model == "claude-3-7-sonnet-latest"
            or self._model == "claude-3-7-sonnet-20250219"
        ):
            reasoning = False
            print(f"Reasoning disabled for {self._model}")

        with self._client.messages.stream(
            model=self._model,
            max_tokens=llm_config.max_tokens,
            thinking=(
                ThinkingConfigEnabledParam(
                    type="enabled", budget_tokens=int(llm_config.max_tokens / 2)
                )
                if reasoning
                else ThinkingConfigDisabledParam(type="disabled")
            ),
            system=system_instruction,
            messages=messages,
        ) as stream:
            yield from stream.text_stream

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        reasoning: bool = False,
        llm_config: LLMConfig = LLMConfig(),
    ) -> str:

        system_instruction, messages = self._convert_messages(messages=messages)

        if reasoning and not (
            self._model == "claude-3-7-sonnet-latest"
            or self._model == "claude-3-7-sonnet-20250219"
        ):
            reasoning = False
            print(f"Reasoning disabled for {self._model}")

        response = self._client.messages.create(
            model=self._model,
            max_tokens=llm_config.max_tokens,
            thinking=(
                ThinkingConfigEnabledParam(
                    type="enabled", budget_tokens=int(llm_config.max_tokens / 2)
                )
                if reasoning
                else ThinkingConfigDisabledParam(type="disabled")
            ),
            system=system_instruction,
            messages=messages,
        )

        if reasoning:
            # Extract reasoning content if available
            reasoning_content = next(
                c for c in response.content if c.type == "thinking"
            )
            if reasoning_content:
                print("### Reasoning")
                print(reasoning_content.thinking)

        return next(c for c in response.content if c.type == "text").text

    def count_tokens(self, text: str) -> int:
        # Count tokens using tiktoken for the specified model
        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def stop_generation(self) -> None:
        # No direct stop support in google-genai; implement no-op or track cancellation
        pass
