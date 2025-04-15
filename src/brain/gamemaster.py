"""WIP Gamemaster"""

import json

from pathlib import Path

from src.brain.chat import SummaryChat
from src.llmclient.llm_client import LLMClient

from src.brain.oracle import Oracle
from src.brain.utils import extract_json_schema
from src.brain.templates import GENERATE_SESSION_EXPANDED_CHAT as GENERATE_SESSION

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction


class Gamemaster:

    def __init__(
        self,
        llm_client: LLMClient,
    ):
        self._llm_client = llm_client

        with open(
            Path(__file__).parent / "prompt_templates" / "gamemaster_system_prompt.txt",
            "r",
            encoding="utf-8",
        ) as f:
            self._role = f.read()

        with open(
            Path(__file__).parent
            / "prompt_templates"
            / "gamemaster_mission_prompt.txt",
            "r",
            encoding="utf-8",
        ) as f:
            self._mission_template = f.read()

    def summary_chat(self, mission_id: int):
        """
        Provide summary chat
        """
        return SummaryChat(
            llm_client=self._llm_client,
            role=self._role,
            mission_id=mission_id,
        )

    def generate_mission(self) -> api_schema_mission.Mission:
        """
        Generate a mission using the LLM client.
        """

        oracle_topic = Oracle.mission()
        print("### Generate Mission")
        print(oracle_topic)

        # llm_response = self._llm_client.completion(
        #     prompt=GENERATE_SESSION.format(question=oracle_topic),
        # )

        llm_response = self._llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.",
                },
                {
                    "role": "user",
                    "content": self._mission_template.format(question=oracle_topic),
                },
            ]
        )

        print(llm_response)

        json_string = extract_json_schema(llm_response)

        try:
            data = json.loads(json_string)
            print("### Result")
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
