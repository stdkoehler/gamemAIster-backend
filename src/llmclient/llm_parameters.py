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
class LLMConfig:
    """
    A strictly typed, immutable configuration object.
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
