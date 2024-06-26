"""WIP Gamemaster"""

import json

from src.llmclient.llm_client import LLMClient

from src.brain.oracle import Oracle
from src.brain.utils import extract_json_schema
from src.brain.templates import GENERATE_SESSION_EXPANDED as GENERATE_SESSION

import src.routers.schema.mission as api_schema_mission


class Gamemaster:

    def __init__(
        self,
        base_url: str,
    ):
        self._llm_client = LLMClient(base_url=base_url)

    def generate_mission(self) -> api_schema_mission.Mission:

        llm_response = self._llm_client.completion(
            prompt=GENERATE_SESSION.format(question=Oracle.mission()),
        )

        print(llm_response)

        json_string = extract_json_schema(llm_response)

        try:
            data = json.loads(json_string)
            print(data)
            mission = {
                "name": data["title"],
                "description": data["task"]
                + data["important_details"]
                + data["possible_outcomes"],
            }
            return api_schema_mission.Mission.model_validate(mission)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {json_string}") from exc
        except KeyError as exc:
            raise ValueError(
                f"LLM response is missing required keys: {json_string}"
            ) from exc
