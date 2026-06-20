"""WIP Gamemaster"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time

from typing import AsyncGenerator
from pathlib import Path

from src.brain.data_types import Interaction
from src.brain.chat import SummaryChat
from src.brain.npc_models import NpcProfile, NPC_PIPELINE_CONFIG, merge_npc
from src.brain.npc_equipment.catalog import npc_equipment_candidates
from src.crud.crud import crud_instance
from src.llmclient.llm_client import (
    LLMClientBase,
    LLMClientClaude,
    LLMClientLocal,
    LLMClientDeepSeek,
    LLMClientMiniMax,
    LLMClientOpenRouter,
    Message,
    MessageContent,
    MessageRole,
)

from src.llmclient.llm_parameters import LLMConfig
from src.llmclient.llm_config_registry import LLMTask

from src.brain.oracle import (
    BaseOracle,
    CustomOracle,
    ExpanseNonHeroOracle,
    ExpanseOracle,
    SeventhSeaOracle,
    ShadowrunOracle,
    SlavicOracle,
    VampireOracle,
    CthulhuOracle,
)
from src.brain.structured_output import parse_with_retry
from src.utils.logger import configure_logger
from src.utils.sqllogger import SQLLogger

from pydantic import BaseModel, ConfigDict

_log = configure_logger("gamemaster")
_sql_logger = SQLLogger()

import src.routers.schema.mission as api_schema_mission
import src.routers.schema.interaction as api_schema_interaction

VAMPIRE_WARMSTART = """
I MUST adhere to Vampire the Masquerade V5 lore and rules and ensure my response aligns with VtM's lore and atmosphere. At the same time my response MUST NOT be cliché or overly dramatic. I don't need to force the supernatural elements if they don't come naturally.
I MUST NOT escalate the situation too quickly. The story pacing should feel natural and immersive.
I always should consider the player character for narrative and mechanical implications: Is he human, ghoul, Kindred of a specific clan? If Kindred, always track Hunger and the Beast's influence.
If the player rolled and provided a result upon my request, I must consider the impact (number of *player's successes* must at least match *difficulty* to succeed). If the player suggested an action, I also must determine if a roll is required. What are the stakes based on VtM V5's rules?
I should not overdo asking for rolls, specifically I shouldn't ask for a similar roll multiple times in short succession.
In the history, look for the most recent <hidden_state></hidden_state> block to understand the current narrative momentum and hidden secrets.
If will now update the hidden state as a concise (2-3 sentences) "snapshot" of the hidden state of the world the isn't directly observable by the player, but is crucial for the narrative.
Let's briefly provide the updated hidden by combining the previous hidden states and the current momentum: <hidden_state>
"""

SHADOWRUN_WARMSTART = """
I MUST adhere to Shadowrun 6th Edition lore and rules and ensure my response aligns with Shadowrun's lore and atmosphere.
I MUST NOT escalate the situation too quickly. The story pacing should feel natural and immersive.
In the history, look for the most recent <hidden_state></hidden_state> block to understand the current narrative momentum and hidden secrets.
If will now update the hidden state as a concise (2-3 sentences) "snapshot" of the hidden state of the world the isn't directly observable by the player, but is crucial for the narrative.
Let's briefly provide the updated hidden by combining the previous hidden states and the current momentum: <hidden_state>
"""

EXPANSE_WARMSTART = """
I MUST adhere to The Expanse RPG lore and rules and ensure my response aligns with The Expanse's lore and atmosphere.
I MUST NOT escalate the situation too quickly. The story pacing should feel natural and immersive.
In the history, look for the most recent <hidden_state></hidden_state> block to understand the current narrative momentum and hidden secrets.
If will now update the hidden state as a concise (2-3 sentences) "snapshot" of the hidden state of the world the isn't directly observable by the player, but is crucial for the narrative.
Let's briefly provide the updated hidden by combining the previous hidden states and the current momentum: <hidden_state>
"""

REASONING_WARMSTART = {
    api_schema_mission.GameType.VAMPIRE_THE_MASQUERADE: VAMPIRE_WARMSTART,
    api_schema_mission.GameType.SHADOWRUN: SHADOWRUN_WARMSTART,
    api_schema_mission.GameType.EXPANSE: EXPANSE_WARMSTART,
}


# ---------------------------------------------------------------------------
# Minimal models for mission title extraction.
# extra="allow" preserves all LLM-generated fields so model_dump() can
# reconstruct the full JSON for Mission.description without needing full
# mission schemas.
# ---------------------------------------------------------------------------


class _FlexBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class _MissionMeta(_FlexBase):
    title: str


class _MissionBody(_FlexBase):
    meta: _MissionMeta


# ---------------------------------------------------------------------------
# Prompt loading with optional example-count trimming.
# ---------------------------------------------------------------------------

# Matches all heading styles used across prompt files:
#   "# Example 1"  (Cthulhu)
#   "## Example #1" (Shadowrun)
#   "## Example 1"  (Vampire, 7th Sea, Expanse)
_EXAMPLE_HEADING = re.compile(r"^#{1,3} Example", re.MULTILINE)


def _parse_budget_cost(value: str) -> float | None:
    """Extracts a soft numeric cost cap from a profile's free-text budget (e.g. '5000 nuyen')."""
    digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def _load_prompt(path: Path, max_examples: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_examples is None:
        return text
    positions = [m.start() for m in _EXAMPLE_HEADING.finditer(text)]
    if not positions or max_examples >= len(positions):
        return text
    return text[: positions[max_examples]].rstrip()


# ---------------------------------------------------------------------------
# Per-game configuration: prompts + oracle class in one place.
# Adding a new game type means one entry here only — no other dispatch needed.
# ---------------------------------------------------------------------------

_GT = api_schema_mission.GameType


@dataclass(frozen=True)
class _GameConfig:
    game_name: str
    system_prompt: str              # relative to prompt_templates/
    mission_prompt: str             # relative to prompt_templates/
    mission_prompt_non_oracle: str
    oracle_class: type[BaseOracle]


_GAME_CONFIGS: dict[tuple[api_schema_mission.GameType, bool], _GameConfig] = {
    (_GT.SHADOWRUN, False): _GameConfig(
        game_name="Shadowrun 6th Edition",
        system_prompt="shadowrun/shadowrun_system_prompt.txt",
        mission_prompt="shadowrun/shadowrun_mission_prompt.txt",
        mission_prompt_non_oracle="shadowrun/shadowrun_mission_prompt.txt",
        oracle_class=ShadowrunOracle,
    ),
    (_GT.VAMPIRE_THE_MASQUERADE, False): _GameConfig(
        game_name="Vampire the Masquerade 5th Edition",
        system_prompt="vampire/vampire_system_prompt.txt",
        mission_prompt="vampire/vampire_mission_prompt.txt",
        mission_prompt_non_oracle="vampire/vampire_non_oracle_mission_prompt.txt",
        oracle_class=VampireOracle,
    ),
    (_GT.CALL_OF_CTHULHU, False): _GameConfig(
        game_name="Call of Cthulhu 7th Edition",
        system_prompt="cthulhu/cthulhu_system_prompt.txt",
        mission_prompt="cthulhu/cthulhu_mission_prompt.txt",
        mission_prompt_non_oracle="cthulhu/cthulhu_non_oracle_mission_prompt.txt",
        oracle_class=CthulhuOracle,
    ),
    (_GT.SEVENTH_SEA, False): _GameConfig(
        game_name="Seventh Sea 2nd Edition",
        system_prompt="seventh_sea/seventh_sea_system_prompt.txt",
        mission_prompt="seventh_sea/seventh_sea_mission_prompt.txt",
        mission_prompt_non_oracle="seventh_sea/seventh_sea_non_oracle_mission_prompt.txt",
        oracle_class=SeventhSeaOracle,
    ),
    (_GT.EXPANSE, False): _GameConfig(
        game_name="The Expanse RPG",
        system_prompt="expanse/expanse_system_prompt.txt",
        mission_prompt="expanse/expanse_mission_prompt.txt",
        mission_prompt_non_oracle="expanse/expanse_mission_prompt.txt",
        oracle_class=ExpanseOracle,
    ),
    (_GT.EXPANSE, True): _GameConfig(
        game_name="The Expanse RPG",
        system_prompt="expanse/expanse_system_prompt_non_hero.txt",
        mission_prompt="expanse/expanse_mission_prompt_non_hero.txt",
        mission_prompt_non_oracle="expanse/expanse_mission_prompt_non_hero.txt",
        oracle_class=ExpanseNonHeroOracle,
    ),
    (_GT.SLAVIC, False): _GameConfig(
        game_name="Baltic Slavic 800 A.D.",
        system_prompt="slavic/slavic_system_prompt.txt",
        mission_prompt="slavic/slavic_mission_prompt.txt",
        mission_prompt_non_oracle="slavic/slavic_mission_prompt.txt",
        oracle_class=SlavicOracle,
    ),
    (_GT.CUSTOM, False): _GameConfig(
        game_name="Custom RPG",
        system_prompt="custom/custom_system_prompt.txt",
        mission_prompt="custom/custom_mission_prompt.txt",
        mission_prompt_non_oracle="custom/custom_mission_prompt.txt",
        oracle_class=CustomOracle,
    ),
    (_GT.CUSTOM, True): _GameConfig(
        game_name="Custom RPG",
        system_prompt="custom/custom_system_prompt.txt",
        mission_prompt="custom/custom_mission_prompt.txt",
        mission_prompt_non_oracle="custom/custom_mission_prompt.txt",
        oracle_class=CustomOracle,
    ),
}


@dataclass(frozen=True)
class _NpcConfig:
    stats_prompt: str       # relative to prompt_templates/
    equipment_prompt: str   # relative to prompt_templates/


_NPC_CONFIGS: dict[api_schema_mission.GameType, _NpcConfig] = {
    _GT.SHADOWRUN: _NpcConfig(
        stats_prompt="shadowrun/shadowrun_npc_stats_prompt.txt",
        equipment_prompt="shadowrun/shadowrun_npc_equipment_prompt.txt",
    ),
    _GT.VAMPIRE_THE_MASQUERADE: _NpcConfig(
        stats_prompt="vampire/vampire_npc_stats_prompt.txt",
        equipment_prompt="vampire/vampire_npc_equipment_prompt.txt",
    ),
    _GT.CALL_OF_CTHULHU: _NpcConfig(
        stats_prompt="cthulhu/cthulhu_npc_stats_prompt.txt",
        equipment_prompt="cthulhu/cthulhu_npc_equipment_prompt.txt",
    ),
    _GT.SEVENTH_SEA: _NpcConfig(
        stats_prompt="seventh_sea/seventh_sea_npc_stats_prompt.txt",
        equipment_prompt="seventh_sea/seventh_sea_npc_equipment_prompt.txt",
    ),
    _GT.EXPANSE: _NpcConfig(
        stats_prompt="expanse/expanse_npc_stats_prompt.txt",
        equipment_prompt="expanse/expanse_npc_equipment_prompt.txt",
    ),
    _GT.SLAVIC: _NpcConfig(
        stats_prompt="slavic/slavic_npc_stats_prompt.txt",
        equipment_prompt="slavic/slavic_npc_equipment_prompt.txt",
    ),
}


def build_gamemaster(
    user_id: str,
    game_type: api_schema_mission.GameType,
    mission_options: MissionOptions,
) -> Gamemaster:
    """
    Factory helper to construct a `Gamemaster` with the configured LLM clients.
    """
    llm_type = os.getenv("LLM")
    if llm_type == "LOCAL":
        reasoning_warmstart = REASONING_WARMSTART.get(game_type, None)
        local_model = os.getenv("LOCAL_MODEL", None)
        client_story = LLMClientLocal(
            base_url="http://127.0.0.1:5000",
            model_name=local_model,
            reasoning_warmstart=(
                "<think>" + reasoning_warmstart if reasoning_warmstart else None
            ),
        )
        client_reasoning = LLMClientLocal(
            base_url="http://127.0.0.1:5000",
            model_name=local_model,
            reasoning_warmstart="<think>",
        )
        return Gamemaster(
            user_id=user_id,
            llm_client_chat=client_story,
            llm_client_reasoning=client_reasoning,
            game_type=game_type,
            mission_options=mission_options,
        )
    elif llm_type == "DEEPSEEK":
        api_key = os.getenv("API_KEY_DEEPSEEK")
        if api_key is None:
            raise ValueError("DeepSeek API key not set")
        return Gamemaster(
            user_id=user_id,
            llm_client_chat=LLMClientDeepSeek(
                api_key=api_key,
                model="deepseek-chat",
                # reasoning_warmstart="<hidden_state>",
            ),
            llm_client_reasoning=LLMClientDeepSeek(
                api_key=api_key, model="deepseek-reasoner"
            ),
            game_type=game_type,
            mission_options=mission_options,
        )
    # elif llm_type == "GEMINI":
    #     api_key = os.getenv("API_KEY_GEMINI")
    #     if api_key is None:
    #         raise ValueError("Gemini API key not set")
    #     return Gamemaster(
    #         user_id=user_id,
    #         llm_client_chat=LLMClientGemini(
    #             api_key=api_key, model="gemini-2.5-pro-exp-03-25"
    #         ),
    #         llm_client_reasoning=LLMClientGemini(
    #             api_key=api_key, model="gemini-2.5-pro-exp-03-25"
    #         ),
    #         game_type=game_type,
    #         mission_options=mission_options,
    #     )
    elif llm_type == "CLAUDE":
        api_key = os.getenv("API_KEY_CLAUDE")
        if api_key is None:
            raise ValueError("Claude API key not set")
        return Gamemaster(
            user_id=user_id,
            llm_client_chat=LLMClientClaude(api_key=api_key, model="claude-sonnet-4-5"),
            llm_client_reasoning=LLMClientClaude(
                api_key=api_key, model="claude-sonnet-4-5"
            ),
            game_type=game_type,
            mission_options=mission_options,
        )
    elif llm_type == "MINIMAX":
        api_key = os.getenv("API_KEY_MINIMAX")
        if api_key is None:
            raise ValueError("MiniMax API key not set")
        return Gamemaster(
            user_id=user_id,
            llm_client_chat=LLMClientMiniMax(api_key=api_key, model="MiniMax-M2.5"),
            llm_client_reasoning=LLMClientMiniMax(
                api_key=api_key, model="MiniMax-M2.5"
            ),
            game_type=game_type,
            mission_options=mission_options,
        )
    elif llm_type == "OPENROUTER":
        api_key = os.getenv("API_KEY_OPENROUTER")
        open_router_model = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")
        if api_key is None:
            raise ValueError("OpenRouter API key not set")
        return Gamemaster(
            user_id=user_id,
            llm_client_chat=LLMClientOpenRouter(
                api_key=api_key, model=open_router_model
            ),
            llm_client_reasoning=LLMClientOpenRouter(
                api_key=api_key, model=open_router_model
            ),
            game_type=game_type,
            mission_options=mission_options,
        )

    raise ValueError(f"Unknown LLM type: {llm_type}")


@dataclass
class MissionOptions:
    non_hero_mode: bool = False
    oracle: bool = True


class Gamemaster:

    def __init__(
        self,
        user_id: str,
        llm_client_reasoning: LLMClientBase,
        llm_client_chat: LLMClientBase,
        game_type: api_schema_mission.GameType,
        mission_options: MissionOptions,
    ):
        self._user_id = user_id
        self._llm_client_reasoning = llm_client_reasoning
        self._llm_client_chat = llm_client_chat
        self._game_type = game_type
        self._mission_options = mission_options

        key = (game_type, mission_options.non_hero_mode)
        if key not in _GAME_CONFIGS:
            if mission_options.non_hero_mode and (game_type, False) in _GAME_CONFIGS:
                raise ValueError(
                    f"Non-hero mode is not supported for {_GAME_CONFIGS[(game_type, False)].game_name}"
                )
            raise ValueError(f"Unknown game type: {game_type}")

        cfg = _GAME_CONFIGS[key]
        self._game_name = cfg.game_name

        prompt_dir = Path(__file__).parent / "prompt_templates"
        # Local models get fewer few-shot examples to reduce context size and
        # improve instruction-following; cloud models receive the full prompt.
        max_examples = 1 if isinstance(llm_client_reasoning, LLMClientLocal) else None
        self._role = (prompt_dir / cfg.system_prompt).read_text(encoding="utf-8")
        self._mission_template = _load_prompt(prompt_dir / cfg.mission_prompt, max_examples)
        self._mission_template_non_oracle = _load_prompt(prompt_dir / cfg.mission_prompt_non_oracle)

        with open(prompt_dir / "text_summary_prompt.txt", "r", encoding="utf-8") as f:
            self._summary_template = f.read()

        with open(prompt_dir / "text_entity_prompt.txt", "r", encoding="utf-8") as f:
            self._entity_template = f.read()

        with open(
            prompt_dir / "text_scene_prompt_examples.txt", "r", encoding="utf-8"
        ) as f:
            self._scene_template = f.read()

        # currently we provide the complete history to the LLM
        # moving to RAG style summary could be better for longer sessions
        # since it's not a knowledge interaction the typical RAG might not be ideal
        # (context still grows)
        # we could tag every interaction in the knowledge database with the
        # entities it mentions, similarly for the next interaction we could
        # let a LLM request extract the entities that are relevant to the current
        # k interactions and only provide those in the summary
        with open(prompt_dir / "summary_provider.txt", "r", encoding="utf-8") as f:
            self._summary_provider_template = f.read()

    async def stream_interaction_response(
        self, prompt: api_schema_interaction.InteractionPrompt
    ) -> AsyncGenerator[str, None]:
        """
        Provide summary chat
        """
        logic_config = self._llm_client_chat.get_logic_config()

        # prompt.prompt is new user input, None if regenerate previous interaction
        # prompt.prev_interaction is previous interaction to update or new user prompt,
        #    None if new interaction

        chat = SummaryChat(
            llm_client_chat=self._llm_client_chat,
            llm_client_reasoning=self._llm_client_reasoning,
            last_k=logic_config.last_k,  # type: ignore
            min_summary_tokens=logic_config.min_summary_tokens,  # type: ignore
            role=self._role,
            summary_template=self._summary_template,
            entity_template=self._entity_template,
            scene_template=self._scene_template,
            summary_provider_template=self._summary_provider_template,
            game_name=self._game_name,
            mission_id=prompt.mission_id,
        )

        # frontend is ground truth for last interaction
        if prompt.prev_interaction is not None:
            interaction = Interaction(
                prompt.prev_interaction.user_input, prompt.prev_interaction.llm_output
            )
        else:
            interaction = None
        for chunk in chat.predict(
            user_input=prompt.prompt,
            last_interaction=interaction,
        ):
            yield json.dumps({"type": chunk[0], "content": chunk[1]}) + "\n"

    def generate_mission(
        self, background: str, detailed_background: str
    ) -> api_schema_mission.Mission:
        """
        Generate a mission using the LLM client.

        Args:
            background (str): User supplied background information to seed the mission.
            detailed_background (str): User supplied detailed background information to seed the mission.
        """
        if detailed_background != "":
            full_background = f"#--- Condensed Summary ---\n\n{background}\n\n#--- Detailed Background ---\n\n{detailed_background}"
        else:
            full_background = background

        if self._mission_options.oracle:
            cfg = _GAME_CONFIGS[(self._game_type, self._mission_options.non_hero_mode)]
            oracle = cfg.oracle_class(llm_client=self._llm_client_reasoning)
            oracle_result = oracle.mission(full_background)
            topic = oracle_result.topic
            system_prompt = self._mission_template
        else:
            oracle_result = None
            topic = json.dumps(
                {"background": full_background}, ensure_ascii=False, indent=2
            )
            system_prompt = self._mission_template_non_oracle

        _log.info("GenerateMission | game_type=%s | oracle=%s | seed_len=%d", self._game_type, self._mission_options.oracle, len(topic))

        # max_tokens is the max tokens the LLM may generate in the response
        # total context window = input tokens + max_tokens
        # our input token is already quite large, so we limit max_tokens to 4096
        # (this includes thinking process for some local models, e.g. gemma3)
        # the limit is configured per-task in LLMTask.ARCHITECT
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(text=system_prompt),
            ),
            Message(role=MessageRole.USER, content=MessageContent(text=topic)),
        ]

        parsed = parse_with_retry(
            messages=messages,
            result_type=_MissionBody,
            llm_client=self._llm_client_reasoning,
            reasoning=True,
            task=LLMTask.ARCHITECT,
        )
        name = parsed.meta.title
        description = json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
        _log.info("GenerateMission complete | title=%s | output_len=%d", name, len(description))
        _sql_logger.log_mission(
            game_type=str(self._game_type),
            oracle_used=self._mission_options.oracle,
            oracle_background=full_background,
            oracle_roll=oracle_result.roll if oracle_result else "",
            oracle_aligned=oracle_result.aligned if oracle_result else "",
            llm_input=str(messages),
            llm_output=description,
        )

        mission = {
            "user_id": self._user_id,
            "name": name,
            "description": description,
            "game_type": self._game_type,
            "background": background,
            "detailed_background": detailed_background,
            "non_hero_mode": self._mission_options.non_hero_mode,
            "oracle": self._mission_options.oracle,
        }
        return api_schema_mission.Mission.model_validate(mission)

    def generate_npc(self, name: str, mission_id: int) -> dict:
        """
        Generate an NPC content dict for `mission_id`, chaining three
        structured-output LLM calls (same parse_with_retry pattern as
        generate_mission): a shared Profiler (narrative description +
        equipment budget), a per-system Stats agent (combat-relevant
        attributes/skills), and a per-system Equipment step that picks from a
        deterministically pre-filtered slice of the system's equipment
        catalog (embedded as prompt text, no tool calling). The result is
        merged into a full CharacterProps-shaped dict scoped to what
        NpcCard.tsx's NPC view renders, with all other required fields
        safe-defaulted.
        """
        if self._game_type not in _NPC_CONFIGS:
            raise ValueError(f"NPC generation is not supported for game type: {self._game_type}")

        npc_cfg = _NPC_CONFIGS[self._game_type]
        stats_model, equipment_model, _ = NPC_PIPELINE_CONFIG[self._game_type]
        game_type = self._game_type

        prompt_dir = Path(__file__).parent / "prompt_templates"
        profile_system_prompt = (prompt_dir / "npc" / "npc_profile_prompt.txt").read_text(
            encoding="utf-8"
        )
        stats_system_prompt = (prompt_dir / npc_cfg.stats_prompt).read_text(encoding="utf-8")
        equipment_system_prompt = (prompt_dir / npc_cfg.equipment_prompt).read_text(
            encoding="utf-8"
        )

        interactions = crud_instance.get_interactions(mission_id, limit=10)
        interactions_text = "\n\n".join(
            interaction.format_interaction_summary() for interaction in interactions
        ) or "(no narrative history yet)"

        _log.info("GenerateNpc | game_type=%s | name=%s | mission_id=%d", self._game_type, name, mission_id)

        profile_messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=profile_system_prompt)),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=f"Game: {self._game_name}\nNPC name: {name}\n\nRecent interactions:\n{interactions_text}"
                ),
            ),
        ]
        profile = parse_with_retry(
            messages=profile_messages,
            result_type=NpcProfile,
            llm_client=self._llm_client_reasoning,
            reasoning=True,
            task=LLMTask.ARCHITECT,
        )

        stats_messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=stats_system_prompt)),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=f"NPC name: {name}\nDescription: {profile.character_description}"
                ),
            ),
        ]
        stats = parse_with_retry(
            messages=stats_messages,
            result_type=stats_model,
            llm_client=self._llm_client_reasoning,
            reasoning=True,
            task=LLMTask.ARCHITECT,
        )

        budget_cost = _parse_budget_cost(profile.value)
        candidates = npc_equipment_candidates(game_type, max_cost=budget_cost)
        candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
        equipment_messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=equipment_system_prompt)),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=(
                        f"NPC name: {name}\nDescription: {profile.character_description}\n"
                        f"Equipment budget: {profile.value}\n\n"
                        f"Available catalog items to choose from (pick only by exact name):\n{candidates_text}"
                    )
                ),
            ),
        ]
        equipment = parse_with_retry(
            messages=equipment_messages,
            result_type=equipment_model,
            llm_client=self._llm_client_reasoning,
            reasoning=True,
            task=LLMTask.ARCHITECT,
        )

        npc_id = int(time.time() * 1000)
        content = merge_npc(game_type, npc_id, name, profile, stats, equipment)
        _log.info("GenerateNpc complete | game_type=%s | name=%s", game_type, name)
        return content
