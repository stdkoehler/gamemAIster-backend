"""LLM Client"""

import json
from dataclasses import asdict

from urllib.parse import urljoin

import requests
from sseclient import SSEClient

from src.llmclient.types import LLMConfig


class LLMClient:
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

    def request(self, url: str, payload):
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

    def chat_completion_stream(
        self, messages: list[dict[str, str]], llm_config: LLMConfig = LLMConfig()
    ):
        data = asdict(llm_config)
        data["messages"] = messages
        data["stream"] = True

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

    def completion_stream(self, prompt: str, llm_config: LLMConfig = LLMConfig()):
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
        self, messages: list[dict[str, str]], llm_config: LLMConfig = LLMConfig()
    ):
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
        return json.loads(response.text)["choices"][0]["message"]["content"]

    def completion(self, prompt: str, llm_config: LLMConfig = LLMConfig()):
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
        return json.loads(response.text)["choices"][0]["text"]

    def count_tokens(self, text: str):
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

    def stop_generation(self):
        """
        Stops the generation process.
        """
        requests.post(
            self._stop_generation_url,
            timeout=60,
        )
        return None
