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
from src.brain.npc_models import NpcProfile
from src.brain.npc_equipment.catalog import get_npc_equipment_categories, npc_equipment_candidates
from src.brain.system_registry import GAME_CONFIGS, NPC_CONFIGS, merge_npc
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

# Matches mission_prompt's top-level "## Example N" few-shot headings. Only
# mission_prompt is trimmed: its examples run to EOF (often 60-80% of the
# file, see e.g. shadowrun/cthulhu's 4 x ~400-line examples), so truncating
# at the cut position is safe. story_prompt is intentionally never trimmed
# here — its examples are a much smaller share of an already-modest file,
# and trimming would also need to splice around the "Negative Examples"
# section and guidance that follows them, which isn't worth the complexity
# for the context-budget it would actually save.
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


def _extract_npc_roster(mission_id: int, roster_key: str) -> str:
    """Pulls just the named-NPC roster (e.g. `keyNPCs`) out of the mission's
    JSON description, so the NPC Profiler can match against it without
    needing the whole mission JSON in context.

    The mission generator only prompts the LLM to include this field — there
    is no hard schema validation enforcing it — so it may legitimately be
    missing, empty, or the mission JSON itself malformed. In all of those
    cases we just omit the roster rather than failing NPC generation."""
    mission = crud_instance.get_mission_description(mission_id)
    if mission is None:
        return "(no mission data)"
    try:
        parsed = json.loads(mission.description)
    except json.JSONDecodeError:
        return "(no mission data)"
    roster = parsed.get(roster_key) if isinstance(parsed, dict) else None
    if not roster:
        return f"(no `{roster_key}` entries in this mission)"
    return json.dumps(roster, ensure_ascii=False, indent=2)


def _load_prompt(path: Path, max_examples: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_examples is None:
        return text
    positions = [m.start() for m in _EXAMPLE_HEADING.finditer(text)]
    if not positions or max_examples >= len(positions):
        return text
    return text[: positions[max_examples]].rstrip()


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
        if key not in GAME_CONFIGS:
            if mission_options.non_hero_mode and (game_type, False) in GAME_CONFIGS:
                raise ValueError(
                    f"Non-hero mode is not supported for {GAME_CONFIGS[(game_type, False)].game_name}"
                )
            raise ValueError(f"Unknown game type: {game_type}")

        cfg = GAME_CONFIGS[key]
        self._game_name = cfg.game_name

        prompt_dir = Path(__file__).parent / "prompt_templates"
        # Local models get fewer few-shot examples to reduce context size and
        # improve instruction-following; cloud models receive the full prompt.
        max_examples = 1 if isinstance(llm_client_reasoning, LLMClientLocal) else None
        self._story_prompt = (prompt_dir / cfg.story_prompt).read_text(encoding="utf-8")
        self._mission_prompt = _load_prompt(prompt_dir / cfg.mission_prompt, max_examples)
        self._mission_prompt_non_oracle = _load_prompt(prompt_dir / cfg.mission_prompt_non_oracle)

        with open(prompt_dir / "text_summary_prompt.txt", "r", encoding="utf-8") as f:
            self._summary_prompt = f.read()

        with open(prompt_dir / "text_entity_prompt.txt", "r", encoding="utf-8") as f:
            self._entity_prompt = f.read()

        with open(
            prompt_dir / "text_scene_prompt_examples.txt", "r", encoding="utf-8"
        ) as f:
            self._scene_prompt = f.read()

        # currently we provide the complete history to the LLM
        # moving to RAG style summary could be better for longer sessions
        # since it's not a knowledge interaction the typical RAG might not be ideal
        # (context still grows)
        # we could tag every interaction in the knowledge database with the
        # entities it mentions, similarly for the next interaction we could
        # let a LLM request extract the entities that are relevant to the current
        # k interactions and only provide those in the summary
        with open(prompt_dir / "summary_provider.txt", "r", encoding="utf-8") as f:
            self._summary_provider_prompt = f.read()

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
            story_prompt=self._story_prompt,
            summary_prompt=self._summary_prompt,
            entity_prompt=self._entity_prompt,
            scene_prompt=self._scene_prompt,
            summary_provider_prompt=self._summary_provider_prompt,
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
            cfg = GAME_CONFIGS[(self._game_type, self._mission_options.non_hero_mode)]
            oracle = cfg.oracle_class(llm_client=self._llm_client_reasoning)
            oracle_result = oracle.mission(full_background)
            topic = oracle_result.topic
            system_prompt = self._mission_prompt
        else:
            oracle_result = None
            topic = json.dumps(
                {"background": full_background}, ensure_ascii=False, indent=2
            )
            system_prompt = self._mission_prompt_non_oracle

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
        if self._game_type not in NPC_CONFIGS:
            raise ValueError(f"NPC generation is not supported for game type: {self._game_type}")

        npc_cfg = NPC_CONFIGS[self._game_type]
        stats_model, equipment_model = npc_cfg.stats_model, npc_cfg.equipment_model
        game_type = self._game_type

        prompt_dir = Path(__file__).parent / "prompt_templates"
        profile_system_prompt = (prompt_dir / npc_cfg.profile_prompt).read_text(encoding="utf-8")
        stats_system_prompt = (prompt_dir / npc_cfg.stats_prompt).read_text(encoding="utf-8")
        equipment_system_prompt = (prompt_dir / npc_cfg.equipment_prompt).read_text(
            encoding="utf-8"
        )

        interactions = crud_instance.get_interactions(mission_id, limit=10)
        interactions_text = "\n\n".join(
            interaction.format_interaction_summary() for interaction in interactions
        ) or "(no narrative history yet)"

        roster_text = _extract_npc_roster(mission_id, npc_cfg.npc_roster_key)

        _log.info("GenerateNpc | game_type=%s | name=%s | mission_id=%d", self._game_type, name, mission_id)

        profile_messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=profile_system_prompt)),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=(
                        f"Game: {self._game_name}\nNPC name: {name}\n\n"
                        f"NPCs already established in this adventure (the new NPC may be one of "
                        f"these):\n{roster_text}\n\n"
                        f"Recent interactions:\n{interactions_text}"
                    )
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

        valid_categories = get_npc_equipment_categories(game_type)
        stats_messages = [
            Message(role=MessageRole.SYSTEM, content=MessageContent(text=stats_system_prompt)),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=(
                        f"NPC name: {name}\nDescription: {profile.character_description}\n\n"
                        f"Valid equipment categories for this system (pick `equipment_categories` only "
                        f"from this list, choosing the ones relevant to this NPC's archetype/role):\n"
                        f"{', '.join(valid_categories)}"
                    )
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
        candidates = npc_equipment_candidates(
            game_type, categories=stats.equipment_categories, max_cost=budget_cost
        )
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
