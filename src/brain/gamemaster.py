"""WIP Gamemaster"""

import json

from src.llmclient.llm_client import LLMClient

from src.brain.templates import GENERATE_SESSION

import src.routers.schema.mission as api_schema_mission


class Gamemaster:

    def __init__(
        self,
        base_url: str,
    ):
        self._llm_client = LLMClient(base_url=base_url)

    def generate_mission(self) -> api_schema_mission.Mission:

        llm_response = self._llm_client.completion(
            prompt=GENERATE_SESSION,
        )

        try:
            data = json.loads(llm_response)
            mission = {"name": data["title"], "description": data["task"]}
            return api_schema_mission.Mission.model_validate(mission)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc
        except KeyError as exc:
            raise ValueError("LLM response is missing required keys.") from exc
