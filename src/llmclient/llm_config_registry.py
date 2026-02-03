from enum import Enum, auto
from src.llmclient.llm_parameters import (
    UNSET,
    TaskResolution,
    LLMConfig,
    LLMLogicConfig,
)


class LLMTask(Enum):
    STORY = auto()
    ARCHITECT = auto()
    SUMMARY = auto()


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
                logic=LLMLogicConfig(last_k=5, min_summary_tokens=2048),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=4096)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=2048)),
        },
        "LLMClientDeepSeek": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=4096),
                logic=LLMLogicConfig(last_k=15, min_summary_tokens=2048),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=8192)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=4096)),
        },
        "LLMClientMiniMax": {
            LLMTask.SUMMARY: TaskResolution(
                llm=LLMConfig(max_tokens=4096),
                logic=LLMLogicConfig(last_k=15, min_summary_tokens=2048),
            ),
            LLMTask.ARCHITECT: TaskResolution(llm=LLMConfig(max_tokens=8192)),
            LLMTask.STORY: TaskResolution(llm=LLMConfig(max_tokens=4096)),
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
        Retrieves internal logic parameters for a specific model.
        Currently pulls from the SUMMARY task resolution as these are general model limits.
        """
        client_map = cls._MATRIX.get(model_identifier, cls._MATRIX["LLMClientLocal"])

        # We assume SUMMARY holds the 'General' logic for the model
        task_resolution = client_map.get(LLMTask.SUMMARY)

        if (
            task_resolution
            and task_resolution.logic
            and task_resolution.logic.last_k is not UNSET
        ):
            return task_resolution.logic

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
