"""WIP Gamemaster"""

import json
from typing import cast

from src.llmclient.llm_client import LLMClient

from src.brain.templates import GENERATE_SESSION


class Gamemaster:

    def __init__(
        self,
        base_url: str,
    ):
        self._llm_client = LLMClient(base_url=base_url)

    def generate_mission(self) -> dict[str, str]:

        llm_response = self._llm_client.completion(
            prompt=GENERATE_SESSION,
        )

        try:
            return cast(dict[str, str], json.loads(llm_response))
        except json.decoder.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc
