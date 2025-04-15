"""WIP Gamemaster"""

import json

from pathlib import Path

from src.brain.data_types import Interaction
from src.brain.chat import SummaryChat
from src.llmclient.llm_client import LLMClient

from src.brain.oracle import Oracle
from src.brain.utils import extract_json_schema

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

        with open(
            Path(__file__).parent / "prompt_templates" / "text_summary_prompt.txt",
            "r",
            encoding="utf-8",
        ) as f:
            self._summary_template = f.read()

        with open(
            Path(__file__).parent / "prompt_templates" / "text_entity_prompt.txt",
            "r",
            encoding="utf-8",
        ) as f:
            self._entity_template = f.read()

    async def stream_interaction_response(
        self, prompt: api_schema_interaction.InteractionPrompt
    ):
        """
        Provide summary chat
        """
        chat = SummaryChat(
            llm_client=self._llm_client,
            role=self._role,
            summary_template=self._summary_template,
            entity_template=self._entity_template,
            mission_id=prompt.mission_id,
        )

        interaction = (
            Interaction(
                prompt.prev_interaction.user_input, prompt.prev_interaction.llm_output
            )
            if prompt.prev_interaction is not None
            else None
        )
        for chunk in chat.predict(prompt.prompt, interaction):
            yield json.dumps({"text": chunk}) + "\n"

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
