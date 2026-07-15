from enum import Enum, auto
from src.llmclient.llm_parameters import (
    TaskResolution,
    LLMConfig,
    LLMLogicConfig,
    ThinkingFeebackPolicy,
)


class LLMTask(Enum):
    """
    The specific task or 'personality' we want to use the LLM for.
    This helps us determine the right config.
    """

    STORY = auto()
    """The 'Storyteller' personality, used for generating the main narrative content of the game."""

    ARCHITECT = auto()
    """The 'Architect' personality, used for world-building, scene generation, and structural content."""

    SUMMARY = auto()
    """The 'Summarizer' personality, used for condensing interactions and extracting key information."""


# Gemma 4 (served locally via LLMClientLocalOpenAI) uses one standardized
# sampling config per *mechanism*, and in this codebase mechanism maps 1:1 to
# task: STORY is the only narrative (chat_completion) task, while ARCHITECT and
# SUMMARY are always structured (build_agent) output. So the narrative sampling
# lives on STORY and the structured sampler set on ARCHITECT + SUMMARY (see the
# _MATRIX entry below and LLMClientLocalOpenAI).
#   - Narrative: Gemma 4's official recommendation (temp 1.0, top_p 0.95,
#     top_k 64); the other local samplers are neutralized.
#   - Structured: low temperature + a scoped repetition penalty, because
#     llama.cpp doesn't hard-enforce tool_choice — at temp 1.0 the model answers
#     in prose instead of calling the tool, and loops without a repetition
#     penalty. Verified live (tests/test_llm_clients_live.py).
_GEMMA4_NARRATIVE_SAMPLING = dict(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    min_p=0,
    repetition_penalty=1.0,
    presence_penalty=0,
    frequency_penalty=0,
    smoothing_factor=0,
    mirostat_mode=0,
)
_GEMMA4_STRUCTURED_SAMPLING = dict(
    temperature=0.1,
    top_p=1.0,
    top_k=0,
    min_p=0.05,
    repetition_penalty=1.05,
    repetition_penalty_range=512,
    smoothing_factor=0.0,
    presence_penalty=0,
    sampler_priority=["min_p", "temperature", "repetition_penalty"],
)


# --- Registry Implementation ---


class ConfigRegistry:
    """
    Centralized mapping for Task + Client configurations.
    Uses 'LLMClientLocal' as the fallback provider.
    """

    # We define the specific capacity (max_tokens) per client here.
    # The personalities (Thinking, Architect, Story) are merged onto these.
    _MATRIX: dict[str, dict[LLMTask, TaskResolution]] = {
        "LLMClientLocal": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=2048),
                logic=LLMLogicConfig(
                    last_k=5, min_summary_tokens=2048, digest_budget_tokens=1536
                ),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=12000)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=2048)),
        },
        "LLMClientDeepSeek": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=4096),
                logic=LLMLogicConfig(last_k=15, min_summary_tokens=2048),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=16384)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=4096)),
        },
        "LLMClientMiniMax": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=4096),
                logic=LLMLogicConfig(
                    last_k=15,
                    min_summary_tokens=2048,
                    keep_thinking_turns=ThinkingFeebackPolicy.forever(),
                ),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=16384)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=4096)),
        },
        "LLMClientClaude": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=4096),
                logic=LLMLogicConfig(
                    last_k=15,
                    min_summary_tokens=2048,
                    keep_thinking_turns=ThinkingFeebackPolicy.forever(),
                ),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=16000)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=8192)),
        },
        # Gemma 4 (and any future native-reasoning local model behind
        # LLMClientLocalOpenAI). Keyed by class name — LLMClientLocalOpenAI
        # overrides model_identifier so this is hit regardless of the dynamic
        # LOCAL_MODEL string. STORY carries Gemma's narrative sampling;
        # ARCHITECT/SUMMARY carry the structured sampler set (identical between
        # them — no per-task difference within the structured mechanism). The
        # SUMMARY logic block mirrors LLMClientLocal's; the larger structured
        # max_tokens leaves room for this model's always-on reasoning trace
        # plus the tool call.
        "LLMClientLocalOpenAI": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=8192, **_GEMMA4_STRUCTURED_SAMPLING),
                logic=LLMLogicConfig(
                    last_k=5, min_summary_tokens=2048, digest_budget_tokens=1536
                ),
            ),
            LLMTask.ARCHITECT: TaskResolution(
                llm=LLMConfig(max_tokens=12000, **_GEMMA4_STRUCTURED_SAMPLING)
            ),
            LLMTask.STORY: TaskResolution(
                llm=LLMConfig(max_tokens=2048, **_GEMMA4_NARRATIVE_SAMPLING)
            ),
        },
        # Add other clients here (e.g., LLMClientGemini)
    }

    # Personality Profiles (Your Original Parameters)
    # will be overwritten with matrix values where applicable
    _PROFILES: dict[LLMTask, LLMConfig] = {
        LLMTask.SUMMARY: LLMConfig(
            temperature=0.1,
            min_p=0.05,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.05,
            repetition_penalty_range=512,
            smoothing_factor=0.0,
            sampler_priority=["min_p", "temperature", "repetition_penalty"],
        ),
        LLMTask.ARCHITECT: LLMConfig(
            temperature=0.95,
            min_p=0.05,
            top_p=1.0,
            top_k=0,
            smoothing_factor=0.25,
            repetition_penalty=1.1,
            repetition_penalty_range=1024,
            presence_penalty=0.25,
            sampler_priority=[
                "min_p",
                "smoothing_factor",
                "temperature",
                "repetition_penalty",
            ],
        ),
        LLMTask.STORY: LLMConfig(
            temperature=1.0,
            min_p=0.1,
            top_p=1.0,
            top_k=0,
            smoothing_factor=0.23,
            repetition_penalty=1.05,
            repetition_penalty_range=600,
            presence_penalty=0.1,
            frequency_penalty=0.0,
            sampler_priority=[
                "min_p",
                "smoothing_factor",
                "temperature",
                "repetition_penalty",
            ],
        ),
    }

    @classmethod
    def get_llm_logic_config(cls, model_identifier: str) -> LLMLogicConfig:
        """
        Retrieves internal logic parameters for a specific model, fully resolved
        (no UNSET fields) — same merge-onto-defaults pattern as get_llm_config,
        so a registry entry only needs to override the fields it cares about.

        Logic config (last_k, min_summary_tokens, digest_budget_tokens,
        keep_thinking_turns) is stored on the SUMMARY task entry because these
        are model-level settings, not task-specific. If you ever need
        per-task logic config, this will need to change.
        """
        client_map = cls._MATRIX.get(model_identifier, cls._MATRIX["LLMClientLocal"])
        task_resolution = client_map.get(LLMTask.SUMMARY)

        if task_resolution and task_resolution.logic:
            return task_resolution.logic.apply_to(LLMLogicConfig.defaults())

        return LLMLogicConfig.defaults()

    @classmethod
    def get_llm_config(cls, model_identifier: str, task: LLMTask) -> LLMConfig:
        """
        Resolves the configuration for a given client and task.
        Falls back to LLMClientLocal if the client is not registered.
        """
        # 1. Get the Capacity (max_tokens) for the specific client
        # Fallback to LLMClientLocal if specific class name isn't in matrix
        client_map = cls._MATRIX.get(model_identifier, cls._MATRIX["LLMClientLocal"])
        task_resolution = client_map.get(task)
        capacity_config = (
            task_resolution.llm if task_resolution else LLMConfig(max_tokens=2048)
        )

        # 2. Get the Personality (samplers, temperature) for the task
        personality_config = cls._PROFILES.get(task, LLMConfig.defaults())

        # 3. Merge: Personality is base, Capaciy overwrites
        # This ensures Personality Samplers + Client-specific Tokens
        return capacity_config.apply_to(personality_config)
