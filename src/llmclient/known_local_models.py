"""Single source of truth for the local models the app knows about.

Every model-specific fact lives here:
  * `mode`  — which local client class it needs (see LocalClientMode).
  * `reasoning_adjustment` — how LLMClientLocal must massage the request to
    make this GGUF actually think (consumed by
    LLMClientLocal._adjust_reasoning via `reasoning_adjustment_for`).

Surfaced to the frontend via GET /settings/llm/local-models as a set of
presets — the UI also allows a free-text "Custom" model outside this list,
for the many local GGUFs we haven't hand-tuned. For a *known* model, mode is
a property of the model, not a preference, so it's always resolved here via
`mode_for()` rather than accepted from the client (see
crud.py::upsert_llm_settings). For an unknown/custom model, mode genuinely
can't be derived — it depends on whether that model's chat template supports
tool-calling, which nothing on our side can determine — so the client
supplies it there, defaulting to NATIVE_COMPLETIONS (see `resolve_local_mode`).
Because the reasoning-adjustment dispatch in llm_client.py also resolves
through this registry rather than matching literals of its own, a new known
model only has to be added here.
"""

from dataclasses import dataclass
from enum import StrEnum


class LocalClientMode(StrEnum):
    # LLMClientLocal — textgen-webui's native /v1/completions endpoint, with
    # the <think>-prefill/continue_ reasoning-warmstart trick. Required for
    # reasoning-tuned GGUFs whose chat template has no native tool-call
    # tokens (see llm_client.py::LLMClientLocal._adjust_reasoning).
    NATIVE_COMPLETIONS = "NATIVE_COMPLETIONS"
    # LLMClientLocalOpenAI — textgen-webui's OpenAI-compatible endpoint, for
    # models with native reasoning + tool-calling (see LLMClientLocalOpenAI).
    NATIVE_TOOL_CALLING = "NATIVE_TOOL_CALLING"


class ReasoningAdjustment(StrEnum):
    """How LLMClientLocal must massage a request to engage a given
    reasoning-tuned GGUF's thinking mode. Each value maps 1:1 to an
    `_adjust_reasoning_*` method on LLMClientLocal. NONE = no special
    handling (the default for custom/unlisted models and for models served
    through LLMClientLocalOpenAI, whose native reasoning needs no massaging)."""

    NONE = "NONE"
    MISTRAL_24B_HERMES = "MISTRAL_24B_HERMES"
    GEMMA3_R1 = "GEMMA3_R1"


@dataclass(frozen=True)
class KnownLocalModel:
    id: str
    label: str
    mode: LocalClientMode
    reasoning_adjustment: ReasoningAdjustment = ReasoningAdjustment.NONE


# The `id`s of the NATIVE_COMPLETIONS entries are exact-match requirements:
# LLMClientLocal keys its reasoning adjustment off the loaded model's name
# (via reasoning_adjustment_for below), so the id here must match the
# filename/identifier textgen-webui reports for the loaded GGUF, or its
# thinking behavior silently drops. The Gemma 4 entry is different —
# LLMClientLocalOpenAI doesn't match the model name at all — but the id is
# still fixed now that model selection is a closed dropdown rather than free
# text; what actually matters for that entry is the `mode`.
KNOWN_LOCAL_MODELS: list[KnownLocalModel] = [
    KnownLocalModel(
        id="gemma-3-r1-27b",
        label="Gemma 3 R1 27B",
        mode=LocalClientMode.NATIVE_COMPLETIONS,
        reasoning_adjustment=ReasoningAdjustment.GEMMA3_R1,
    ),
    KnownLocalModel(
        id="mistral-24b-hermes",
        label="Mistral 24B DeepHermes",
        mode=LocalClientMode.NATIVE_COMPLETIONS,
        reasoning_adjustment=ReasoningAdjustment.MISTRAL_24B_HERMES,
    ),
    KnownLocalModel(
        id="gemma-4-26b",
        label="Gemma 4 26B (native tool-calling)",
        mode=LocalClientMode.NATIVE_TOOL_CALLING,
    ),
]

_BY_ID: dict[str, KnownLocalModel] = {m.id: m for m in KNOWN_LOCAL_MODELS}


def reasoning_adjustment_for(model_name: str | None) -> ReasoningAdjustment:
    """The reasoning adjustment a local model needs, resolved from the
    registry. Unknown/custom model names get NONE — i.e. no special handling,
    exactly the prior fall-through behavior of LLMClientLocal._adjust_reasoning."""
    known = _BY_ID.get(model_name or "")
    return known.reasoning_adjustment if known else ReasoningAdjustment.NONE


def mode_for(model_name: str | None) -> LocalClientMode:
    """The local client mode a model needs, resolved from the registry —
    never a user choice for a *known* model (see the module docstring).
    Unknown model names default to NATIVE_COMPLETIONS, the client that
    requires no special model support. Only meaningful for known models;
    callers handling custom models should use `resolve_local_mode` instead,
    which lets the client supply the mode when it can't be derived."""
    known = _BY_ID.get(model_name or "")
    return known.mode if known else LocalClientMode.NATIVE_COMPLETIONS


def is_known_model(model_name: str | None) -> bool:
    """Whether `model_name` matches one of our hand-tuned presets, as opposed
    to a free-text "Custom" entry the user typed in."""
    return (model_name or "") in _BY_ID


def resolve_local_mode(
    model_name: str | None, requested_mode: LocalClientMode | None
) -> LocalClientMode:
    """The mode to actually store for a LOCAL provider setting.

    For a known model, the registry is authoritative and `requested_mode` is
    ignored outright — a client can never override which client class a
    preset uses (defense-in-depth against a raw API call smuggling a
    mismatched mode alongside a known model_name).

    For a custom model, the mode isn't something we can derive — it depends
    on whether that specific model's chat template supports tool-calling,
    which nothing on our side can know — so the caller-supplied value is
    honored, defaulting to NATIVE_COMPLETIONS (the option that works for any
    textgen-webui model, known or not) when none was given.
    """
    if is_known_model(model_name):
        return mode_for(model_name)
    return requested_mode or LocalClientMode.NATIVE_COMPLETIONS
