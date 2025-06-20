"""WIP Gamemaster"""

import json

from typing import AsyncGenerator
from pathlib import Path

from src.brain.data_types import Interaction
from src.brain.chat import SummaryChat
from src.llmclient.llm_client import LLMClientBase
from src.llmclient.llm_parameters import LLMConfig

from src.brain.oracle import (
    BaseOracle,
    ExpanseOracle,
    SeventhSeaOracle,
    ShadowrunOracle,
    VampireOracle,
    CthulhuOracle,
)
from src.brain.json_tools import extract_json_schema

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction


class Gamemaster:

    def __init__(
        self,
        llm_client_reasoning: LLMClientBase,
        llm_client_chat: LLMClientBase,
        game_type: api_schema_mission.GameType,
        non_hero_mode: bool = False,
    ):
        self._llm_client_reasoning = llm_client_reasoning
        self._llm_client_chat = llm_client_chat
        self._game_type = game_type
        self._non_hero_mode = non_hero_mode

        prompt_dir = Path(__file__).parent / "prompt_templates"

        if game_type == api_schema_mission.GameType.SHADOWRUN:
            if non_hero_mode:
                raise ValueError(
                    "Non-hero mode is not supported for Shadowrun 6th Edition"
                )
            else:
                mission_prompt = (
                    prompt_dir / "shadowrun" / "shadowrun_mission_prompt.txt"
                )
                system_prompt = prompt_dir / "shadowrun" / "shadowrun_system_prompt.txt"
            self._game_name = "Shadowrun 6th Edition"

        elif game_type == api_schema_mission.GameType.VAMPIRE_THE_MASQUERADE:
            if non_hero_mode:
                raise ValueError(
                    "Non-hero mode is not supported for Vampire the Masquerade 5th Edition"
                )
            else:
                mission_prompt = prompt_dir / "vampire" / "vampire_mission_prompt.txt"
                system_prompt = prompt_dir / "vampire" / "vampire_system_prompt.txt"
            self._game_name = "Vampire the Masquerade 5th Edition"
        elif game_type == api_schema_mission.GameType.CALL_OF_CTHULHU:
            if non_hero_mode:
                raise ValueError(
                    "Non-hero mode is not supported for Call of Cthulhu 7th Edition"
                )
            else:
                mission_prompt = prompt_dir / "cthulhu" / "cthulhu_mission_prompt.txt"
                system_prompt = prompt_dir / "cthulhu" / "cthulhu_system_prompt.txt"
            self._game_name = "Call of Cthulhu 7th Edition"
        elif game_type == api_schema_mission.GameType.SEVENTH_SEA:
            mission_prompt = (
                prompt_dir / "seventh_sea" / "seventh_sea_mission_prompt.txt"
            )
            system_prompt = prompt_dir / "seventh_sea" / "seventh_sea_system_prompt.txt"
            self._game_name = "Seventh Sea 2nd Edition"
        elif game_type == api_schema_mission.GameType.EXPANSE:
            if non_hero_mode:
                mission_prompt = (
                    prompt_dir / "expanse" / "expanse_mission_prompt_non_hero.txt"
                )
                system_prompt = (
                    prompt_dir / "expanse" / "expanse_system_prompt_non_hero.txt"
                )
            else:
                mission_prompt = prompt_dir / "expanse" / "expanse_mission_prompt.txt"
                system_prompt = prompt_dir / "expanse" / "expanse_system_prompt.txt"
            self._game_name = "The Expanse RPG"
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        with open(system_prompt, "r", encoding="utf-8") as f:
            self._role = f.read()

        with open(mission_prompt, "r", encoding="utf-8") as f:
            self._mission_template = f.read()

        with open(prompt_dir / "text_summary_prompt.txt", "r", encoding="utf-8") as f:
            self._summary_template = f.read()

        with open(prompt_dir / "text_entity_prompt.txt", "r", encoding="utf-8") as f:
            self._entity_template = f.read()

        with open(
            prompt_dir / "text_scene_prompt_examples.txt", "r", encoding="utf-8"
        ) as f:
            self._scene_template = f.read()

        with open(prompt_dir / "summary_provider.txt", "r", encoding="utf-8") as f:
            self._summary_provider_template = f.read()

    async def stream_interaction_response(
        self, prompt: api_schema_interaction.InteractionPrompt
    ) -> AsyncGenerator[str, None]:
        """
        Provide summary chat
        """

        chat = SummaryChat(
            llm_client_chat=self._llm_client_chat,
            llm_client_reasoning=self._llm_client_reasoning,
            last_k=5,
            min_summary_tokens=1024,
            role=self._role,
            summary_template=self._summary_template,
            entity_template=self._entity_template,
            scene_template=self._scene_template,
            summary_provider_template=self._summary_provider_template,
            game_name=self._game_name,
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

    def generate_mission(self, background: str) -> api_schema_mission.Mission:
        """
        Generate a mission using the LLM client.

        Args:
            background (str): User supplied background information to seed the mission.
        """
        oracle: BaseOracle
        if self._game_type == api_schema_mission.GameType.SHADOWRUN:
            oracle = ShadowrunOracle(llm_client=self._llm_client_reasoning)
            oracle_topic = oracle.mission(background)
        elif self._game_type == api_schema_mission.GameType.VAMPIRE_THE_MASQUERADE:
            oracle = VampireOracle(llm_client=self._llm_client_reasoning)
            oracle_topic = oracle.mission(background)
        elif self._game_type == api_schema_mission.GameType.CALL_OF_CTHULHU:
            oracle = CthulhuOracle(llm_client=self._llm_client_reasoning)
            oracle_topic = oracle.mission(background)
        elif self._game_type == api_schema_mission.GameType.SEVENTH_SEA:
            oracle = SeventhSeaOracle(llm_client=self._llm_client_reasoning)
            oracle_topic = oracle.mission(background)
        elif self._game_type == api_schema_mission.GameType.EXPANSE:
            oracle = ExpanseOracle(
                llm_client=self._llm_client_reasoning, non_hero_mode=self._non_hero_mode
            )
            oracle_topic = oracle.mission(background)
        else:
            oracle_topic = ""

        print("### GenerateMission")
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
                {"role": "user", "content": oracle_topic},
            ],
            reasoning=True,
            llm_config=LLMConfig(max_tokens=4096),
        )

        print("### LLM Response")
        print(llm_response)

        json_string = extract_json_schema(llm_response)

        try:
            data = json.loads(json_string)
            print("### Result")
            print(data)
            if self._game_type == api_schema_mission.GameType.SHADOWRUN:
                name = data["mission"]["meta"]["title"]
            else:
                name = data["meta"]["title"]
            mission = {
                "name": name,
                "description": json_string,
                "game_type": self._game_type,
                "background": background,
                "non_hero_mode": self._non_hero_mode,
            }
            return api_schema_mission.Mission.model_validate(mission)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {json_string}") from exc
        except KeyError as exc:
            raise ValueError(
                f"LLM response is missing required keys: {json_string}"
            ) from exc
