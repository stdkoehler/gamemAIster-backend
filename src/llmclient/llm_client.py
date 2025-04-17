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
    def completion_stream(
        self, prompt: str, llm_config: LLMConfig = LLMConfig()
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
    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()) -> str:
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


class LLMClient(LLMClientBase):
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
                "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
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
            verify=False,
            stream=True,
            timeout=10,
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
            verify=False,
            stream=True,
            timeout=10,
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
            verify=False,
            stream=False,
            timeout=60,
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
            verify=False,
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


class LLMClientOpenRouter(LLMClientBase):
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
        # not sure how to count tokens for openrouter models
        return len(text)

    def stop_generation(self) -> None:
        """
        Stops the generation process.
        """
