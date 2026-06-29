"""LLM Client Types"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, fields, replace, asdict


class Unset:
    """A sentinel value to represent an unset parameter."""

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = Unset()


@dataclass(frozen=True)
class ThinkingFeebackPolicy:
    """Whether and how many assistant messages we feed back as "thinking" content to the model."""

    limit: int | None  # None = Forever, 0 = Never, >0 = N turns

    @classmethod
    def forever(cls) -> ThinkingFeebackPolicy:
        """Feedback content from all assistant messages will be fed back to the model."""
        return cls(limit=None)

    @classmethod
    def never(cls) -> ThinkingFeebackPolicy:
        """No feedback content from assistant messages will be fed back to the model."""
        return cls(limit=0)

    @classmethod
    def turns(cls, n: int) -> ThinkingFeebackPolicy:
        """Feedback content from the last 'n' assistant messages will be fed back to the model."""
        if n < 0:
            raise ValueError("Turn count must be non-negative")
        return cls(limit=n)

    def should_keep(self, turns_ago: int) -> bool:
        """Determines if feedback should be kept based on how many turns ago it was."""
        if self.limit is None:
            return True
        return turns_ago <= self.limit


@dataclass(frozen=True)
class LLMConfig:
    """
    Inference parameters sent to the LLM API.
    Defaults are set to UNSET to facilitate the Overlay Pattern.
    """

    max_tokens: int | Unset = UNSET
    temperature: float | Unset = UNSET
    top_p: float | Unset = UNSET
    min_p: float | Unset = UNSET
    top_k: int | Unset = UNSET
    repetition_penalty: float | Unset = UNSET
    presence_penalty: float | Unset = UNSET
    frequency_penalty: float | Unset = UNSET
    guidance_scale: float | Unset = UNSET
    mirostat_mode: int | Unset = UNSET
    mirostat_tau: float | Unset = UNSET
    mirostat_eta: float | Unset = UNSET
    smoothing_factor: float | Unset = UNSET
    repetition_penalty_range: int | Unset = UNSET
    typical_p: float | Unset = UNSET
    tfs: float | Unset = UNSET
    top_a: float | Unset = UNSET
    epsilon_cutoff: float | Unset = UNSET
    eta_cutoff: float | Unset = UNSET
    sampler_priority: list[str] | Unset = UNSET
    stop: list[str] | Unset = UNSET
    logits_processor: list[str] | Unset = UNSET

    def apply_to(self, base: "LLMConfig") -> "LLMConfig":
        """
        Creates a new LLMConfig by merging 'self' (the patch) onto 'base'.
        Only fields that are NOT 'UNSET' in 'self' will overwrite 'base'.
        """
        actual_overrides = {}

        # Iterating over fields preserves the identity of UNSET
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not UNSET:
                actual_overrides[field.name] = value

        # Apply only the actual overrides to the base object
        return replace(base, **actual_overrides)

    def resolve(self) -> "LLMConfig":
        """
        Fills any remaining UNSET fields with system-wide defaults.
        Ensures the returned object has NO 'UNSET' values.
        """
        # Start with the absolute ground-truth defaults
        defaults = self.defaults()
        # Apply 'self' (which may have UNSETs) onto the defaults
        return self.apply_to(defaults)

    def to_dict(self) -> dict[str, Any]:
        """Returns a dict of all fields that are NOT 'UNSET'."""
        return {k: v for k, v in asdict(self).items() if v is not UNSET}

    @classmethod
    def defaults(cls) -> LLMConfig:
        """The absolute global defaults for the application."""
        return cls(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            min_p=0,
            top_k=20,
            repetition_penalty=1,  # nous capbyara #1.15 mixtral
            presence_penalty=0,
            frequency_penalty=0,
            guidance_scale=1,
            mirostat_mode=0,
            mirostat_tau=5,
            mirostat_eta=0.1,
            smoothing_factor=0,
            repetition_penalty_range=1024,
            typical_p=1,
            tfs=1,
            top_a=0,
            epsilon_cutoff=0,
            eta_cutoff=0,
            sampler_priority=[
                "temperature",
                "dynamic_temperature",
                "quadratic_sampling",
                "top_k",
                "top_p",
                "typical_p",
                "epsilon_cutoff",
                "eta_cutoff",
                "tfs",
                "top_a",
                "min_p",
                "mirostat",
            ],
            stop=[],
            logits_processor=[],
        )


@dataclass(frozen=True)
class LLMLogicConfig:
    """
    Internal application logic parameters.
    These are used by the server (e.g., SummaryChat) and never sent to the LLM API.
    """

    last_k: int | Unset = UNSET
    """Conversation Turns that are always kept unsummarized."""

    min_summary_tokens: int | Unset = UNSET
    """Minimum number of tokens required for a summary to be generated."""

    digest_budget_tokens: int | Unset = UNSET
    """
    Once the stored summary ledger grows past this many tokens, it's no longer
    injected into the story prompt directly — a digest pass condenses the
    whole ledger into a small "story so far" blurb instead, refreshed again
    each time the ledger grows by another budget's worth. Keeps the injected
    context bounded regardless of campaign length. See docs/conversation_memory.html.
    """

    keep_thinking_turns: ThinkingFeebackPolicy | Unset = UNSET
    """
    Number of conversation turns for which we feed thinking content back to the model.
    MiniMax M2.1, Claude 3, Gemini Pro benefit from this.
    """

    def apply_to(self, base: LLMLogicConfig) -> LLMLogicConfig:
        """
        Creates a new LLMLogicConfig by merging 'self' (the patch) onto 'base'.
        Only fields that are NOT 'UNSET' in 'self' will overwrite 'base'.
        """
        actual_overrides: dict[str, Any] = {}

        for field in fields(self):
            value = getattr(self, field.name)
            if value is not UNSET:
                actual_overrides[field.name] = value

        return replace(base, **actual_overrides)

    @classmethod
    def defaults(cls) -> LLMLogicConfig:
        """Default logic settings if not specified by the model registry."""
        return cls(
            last_k=5,
            min_summary_tokens=2048,
            digest_budget_tokens=4096,
            keep_thinking_turns=ThinkingFeebackPolicy.never(),
        )


@dataclass(frozen=True)
class TaskResolution:
    """
    A bundle containing both the API configuration and internal logic configuration.
    """

    llm: LLMConfig
    logic: LLMLogicConfig | None = None
