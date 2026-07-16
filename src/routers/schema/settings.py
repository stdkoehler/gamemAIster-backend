from enum import StrEnum

from pydantic import BaseModel

# The wire-level "local client mode" is the same concept the client layer
# already owns as LocalClientMode — reuse it as the API-contract type rather
# than defining a second, drift-prone copy here.
from src.llmclient.known_local_models import LocalClientMode as LocalMode


class LlmProvider(StrEnum):
    LOCAL = "LOCAL"
    DEEPSEEK = "DEEPSEEK"
    MINIMAX = "MINIMAX"
    OPENROUTER = "OPENROUTER"


class LocalInterface(StrEnum):
    TEXTGEN_WEBUI = "TEXTGEN_WEBUI"
    # Reserved for when real Ollama support is built — not yet consumed by
    # build_gamemaster(). Kept in the enum now so the UI can show it as a
    # disabled option instead of this being a schema migration later.
    OLLAMA = "OLLAMA"


class KnownLocalModelSchema(BaseModel):
    id: str
    label: str
    mode: LocalMode


class LlmProviderSettings(BaseModel):
    """One provider's saved settings. `has_api_key` reports whether a key is
    stored without ever returning the (decrypted) key itself."""

    provider: LlmProvider
    model_name: str | None = None
    has_api_key: bool = False
    local_host: str | None = None
    local_port: int | None = None
    local_interface: LocalInterface | None = None
    local_mode: LocalMode | None = None


class LlmSettingsOverview(BaseModel):
    """All of a user's saved per-provider settings plus which one is active.
    `active_provider` is None when the user has no stored settings at all
    (build_gamemaster then falls back to the deployment's env-var config)."""

    active_provider: LlmProvider | None = None
    providers: list[LlmProviderSettings] = []


class SaveLlmSettings(BaseModel):
    """`local_mode` is only honored for a *custom* model_name (one that isn't
    one of the known presets from GET /settings/llm/local-models) — for a
    known preset, which local client it needs is a property of the model, not
    a user choice, and is always derived server-side via
    known_local_models.resolve_local_mode() regardless of what's sent here."""

    provider: LlmProvider
    model_name: str | None = None
    api_key: str | None = None
    local_host: str | None = None
    local_port: int | None = None
    local_interface: LocalInterface | None = None
    local_mode: LocalMode | None = None
