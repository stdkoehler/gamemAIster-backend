from enum import Enum, auto
from typing import Dict
from src.llmclient.llm_parameters import LLMConfig


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
    _MATRIX: Dict[str, Dict[LLMTask, LLMConfig]] = {
        "LLMClientLocal": {
            LLMTask.SUMMARY: LLMConfig(max_tokens=2048),
            LLMTask.ARCHITECT: LLMConfig(max_tokens=4096),
            LLMTask.STORY: LLMConfig(max_tokens=2048),
        },
        "LLMClientDeepSeek": {
            LLMTask.SUMMARY: LLMConfig(max_tokens=4096),
            LLMTask.ARCHITECT: LLMConfig(max_tokens=8192),
            LLMTask.STORY: LLMConfig(max_tokens=4096),
        },
        # Add other clients here (e.g., LLMClientGemini)
    }

    # Personality Profiles (Your Original Parameters)
    _PROFILES: Dict[LLMTask, LLMConfig] = {
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
    def get_config(cls, client_class_name: str, task: LLMTask) -> LLMConfig:
        """
        Resolves the configuration for a given client and task.
        Falls back to LLMClientLocal if the client is not registered.
        """
        # 1. Get the Capacity (max_tokens) for the specific client
        # Fallback to LLMClientLocal if specific class name isn't in matrix
        client_capacity_map = cls._MATRIX.get(
            client_class_name, cls._MATRIX["LLMClientLocal"]
        )
        capacity_config = client_capacity_map.get(task, LLMConfig(max_tokens=1024))

        # 2. Get the Personality (samplers, temperature) for the task
        personality_config = cls._PROFILES.get(task, LLMConfig.defaults())

        # 3. Merge: Capacity is the base, Personality is the overlay
        # This ensures Personality Samplers + Client-specific Tokens
        return personality_config.apply_to(capacity_config)
