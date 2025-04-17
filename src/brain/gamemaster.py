"""WIP Gamemaster"""

import json

from typing import AsyncGenerator
from pathlib import Path

from src.brain.data_types import Interaction
from src.brain.chat import SummaryChat
from src.llmclient.llm_client import LLMClientBase
from src.llmclient.llm_parameters import LLMConfig

from src.brain.oracle import Oracle
from src.brain.json_tools import extract_json_schema

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction


class Gamemaster:

    def __init__(
        self,
        llm_client_reasoning: LLMClientBase,
        llm_client_chat: LLMClientBase,
    ):
        self._llm_client_reasoning = llm_client_reasoning
        self._llm_client_chat = llm_client_chat

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
    ) -> AsyncGenerator[str, None]:
        """
        Provide summary chat
        """
        chat = SummaryChat(
            llm_client_chat=self._llm_client_chat,
            llm_client_reasoning=self._llm_client_reasoning,
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

        llm_response = self._llm_client_reasoning.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self._mission_template,
                },
                {"role": "user", "content": "Create scenario around:\n" + oracle_topic},
            ],
            reasoning=True,
            llm_config=LLMConfig(max_tokens=4096),
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
