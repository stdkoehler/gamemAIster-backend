import os

import pytest
import sqlalchemy

from src.crud.crud import CRUD, ResolvedLlmSettings
from src.crud.sqlmodel import Base
from src.routers.schema.settings import (
    SaveLlmSettings,
    LlmProvider,
    LocalInterface,
)
from src.brain.gamemaster import build_gamemaster, LlmNotConfiguredError
from src.llmclient.llm_client import LLMClientLocal, LLMClientLocalOpenAI
from src.llmclient.known_local_models import (
    KNOWN_LOCAL_MODELS,
    LocalClientMode,
    ReasoningAdjustment,
    reasoning_adjustment_for,
    mode_for,
    is_known_model,
    resolve_local_mode,
)
import src.routers.schema.mission as api_schema_mission


def test_reasoning_adjustment_resolves_from_registry():
    # Guards the single-source-of-truth wiring: LLMClientLocal._adjust_reasoning
    # dispatches on these, so the registry must keep mapping the exact model
    # ids to their adjustment strategy.
    assert (
        reasoning_adjustment_for("gemma-3-r1-27b") == ReasoningAdjustment.GEMMA3_R1
    )
    assert (
        reasoning_adjustment_for("mistral-24b-hermes")
        == ReasoningAdjustment.MISTRAL_24B_HERMES
    )
    # Unlisted / custom / None -> no special handling (prior fall-through).
    assert reasoning_adjustment_for("some-other-model") == ReasoningAdjustment.NONE
    assert reasoning_adjustment_for(None) == ReasoningAdjustment.NONE
    # The tool-calling model never hits LLMClientLocal._adjust_reasoning, so
    # its adjustment is NONE.
    assert reasoning_adjustment_for("gemma-4-26b") == ReasoningAdjustment.NONE


def test_every_known_model_id_is_unique():
    ids = [m.id for m in KNOWN_LOCAL_MODELS]
    assert len(ids) == len(set(ids))


def test_mode_for_resolves_from_registry():
    assert mode_for("gemma-3-r1-27b") == LocalClientMode.NATIVE_COMPLETIONS
    assert mode_for("mistral-24b-hermes") == LocalClientMode.NATIVE_COMPLETIONS
    assert mode_for("gemma-4-26b") == LocalClientMode.NATIVE_TOOL_CALLING
    # Unknown/legacy model names default to NATIVE_COMPLETIONS rather than
    # erroring — the client that needs no special model support.
    assert mode_for("some-other-model") == LocalClientMode.NATIVE_COMPLETIONS
    assert mode_for(None) == LocalClientMode.NATIVE_COMPLETIONS


@pytest.fixture(scope="module")
def crud_instance():
    dbase = "sqlite:///unittest_llm_settings.db"
    Base.metadata.create_all(sqlalchemy.create_engine(dbase))
    return CRUD(dbase=dbase)


def test_get_llm_settings_missing_returns_none(crud_instance):
    assert crud_instance.get_llm_settings("no_such_user") is None


def test_upsert_and_get_round_trip(crud_instance):
    crud_instance.upsert_llm_settings(
        "user_a",
        SaveLlmSettings(
            provider=LlmProvider.OPENROUTER,
            model_name="deepseek/deepseek-v4-flash",
            api_key="sk-secret-123",
        ),
    )
    settings = crud_instance.get_llm_settings("user_a")
    assert settings == ResolvedLlmSettings(
        provider="OPENROUTER",
        model_name="deepseek/deepseek-v4-flash",
        api_key="sk-secret-123",
    )


def test_upsert_and_get_round_trip_local_fields(crud_instance):
    # gemma-4-26b is a NATIVE_TOOL_CALLING model in the registry — SaveLlmSettings
    # has no local_mode field at all, so this is resolved purely from model_name.
    crud_instance.upsert_llm_settings(
        "user_local",
        SaveLlmSettings(
            provider=LlmProvider.LOCAL,
            model_name="gemma-4-26b",
            local_host="192.168.1.50",
            local_port=5000,
            local_interface=LocalInterface.TEXTGEN_WEBUI,
        ),
    )
    settings = crud_instance.get_llm_settings("user_local")
    assert settings == ResolvedLlmSettings(
        provider="LOCAL",
        model_name="gemma-4-26b",
        api_key=None,
        local_host="192.168.1.50",
        local_port=5000,
        local_interface="TEXTGEN_WEBUI",
        local_mode="NATIVE_TOOL_CALLING",
    )


def test_upsert_derives_mode_from_model_not_client(crud_instance):
    # gemma-3-r1-27b is NATIVE_COMPLETIONS in the registry — must be resolved
    # that way server-side regardless of what a client might try to send.
    crud_instance.upsert_llm_settings(
        "user_local2",
        SaveLlmSettings(provider=LlmProvider.LOCAL, model_name="gemma-3-r1-27b"),
    )
    settings = crud_instance.get_llm_settings("user_local2")
    assert settings.local_mode == "NATIVE_COMPLETIONS"

    # Switching to the tool-calling model re-derives the mode too.
    crud_instance.upsert_llm_settings(
        "user_local2",
        SaveLlmSettings(provider=LlmProvider.LOCAL, model_name="gemma-4-26b"),
    )
    settings = crud_instance.get_llm_settings("user_local2")
    assert settings.local_mode == "NATIVE_TOOL_CALLING"


def test_upsert_honors_client_mode_for_custom_model(crud_instance):
    # A model name outside the registry is "Custom" — mode can't be derived,
    # so the client's requested mode is trusted.
    assert not is_known_model("my-local-qwen3-gguf")
    crud_instance.upsert_llm_settings(
        "user_custom1",
        SaveLlmSettings(
            provider=LlmProvider.LOCAL,
            model_name="my-local-qwen3-gguf",
            local_mode=LocalClientMode.NATIVE_TOOL_CALLING,
        ),
    )
    settings = crud_instance.get_llm_settings("user_custom1")
    assert settings.local_mode == "NATIVE_TOOL_CALLING"


def test_upsert_defaults_custom_model_mode_to_native_completions(crud_instance):
    # No mode supplied for a custom model -> the safe default, not an error.
    crud_instance.upsert_llm_settings(
        "user_custom2",
        SaveLlmSettings(provider=LlmProvider.LOCAL, model_name="some-other-gguf"),
    )
    settings = crud_instance.get_llm_settings("user_custom2")
    assert settings.local_mode == "NATIVE_COMPLETIONS"


def test_upsert_ignores_client_mode_for_known_model(crud_instance):
    # Even if a client tries to smuggle a mismatched mode alongside a known
    # model_name, the registry wins — this is the defense-in-depth case.
    crud_instance.upsert_llm_settings(
        "user_custom3",
        SaveLlmSettings(
            provider=LlmProvider.LOCAL,
            model_name="gemma-3-r1-27b",
            local_mode=LocalClientMode.NATIVE_TOOL_CALLING,
        ),
    )
    settings = crud_instance.get_llm_settings("user_custom3")
    assert settings.local_mode == "NATIVE_COMPLETIONS"


def test_resolve_local_mode_matches_upsert_behavior():
    assert resolve_local_mode(
        "gemma-4-26b", LocalClientMode.NATIVE_COMPLETIONS
    ) == LocalClientMode.NATIVE_TOOL_CALLING
    assert (
        resolve_local_mode("custom-model", None) == LocalClientMode.NATIVE_COMPLETIONS
    )
    assert (
        resolve_local_mode("custom-model", LocalClientMode.NATIVE_TOOL_CALLING)
        == LocalClientMode.NATIVE_TOOL_CALLING
    )


def test_local_mode_is_none_for_non_local_providers(crud_instance):
    crud_instance.upsert_llm_settings(
        "user_local3",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK, api_key="sk-x"),
    )
    settings = crud_instance.get_llm_settings("user_local3")
    assert settings.local_mode is None


def test_upsert_without_api_key_keeps_existing_key(crud_instance):
    crud_instance.upsert_llm_settings(
        "user_b",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK, api_key="sk-original"),
    )
    crud_instance.upsert_llm_settings(
        "user_b",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK, model_name="deepseek-chat"),
    )
    settings = crud_instance.get_llm_settings("user_b")
    assert settings.api_key == "sk-original"
    assert settings.model_name == "deepseek-chat"


def test_stored_api_key_is_encrypted_at_rest(crud_instance):
    crud_instance.upsert_llm_settings(
        "user_c",
        SaveLlmSettings(provider=LlmProvider.MINIMAX, api_key="sk-plaintext"),
    )
    with crud_instance._sessionmaker() as session:
        from src.crud.sqlmodel import UserLlmSettings
        from sqlalchemy import select

        row = session.execute(
            select(UserLlmSettings).where(UserLlmSettings.user_id == "user_c")
        ).scalar_one()
        assert row.api_key_encrypted != "sk-plaintext"
        assert row.api_key_encrypted is not None


def test_each_provider_keeps_its_own_key_across_swaps(crud_instance):
    # Per-provider storage: saving one provider must not touch another's key,
    # and swapping the active provider back restores its saved settings.
    crud_instance.upsert_llm_settings(
        "user_swap",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK, api_key="sk-deepseek"),
    )
    crud_instance.upsert_llm_settings(
        "user_swap",
        SaveLlmSettings(
            provider=LlmProvider.OPENROUTER,
            model_name="some/model",
            api_key="sk-openrouter",
        ),
    )
    # OpenRouter was saved last, so it's active now.
    active = crud_instance.get_llm_settings("user_swap")
    assert active.provider == "OPENROUTER"
    assert active.api_key == "sk-openrouter"

    # Both providers' keys are retained side by side.
    active_provider, overviews = crud_instance.list_llm_settings("user_swap")
    assert active_provider == "OPENROUTER"
    assert {o.provider: o.has_api_key for o in overviews} == {
        "DEEPSEEK": True,
        "OPENROUTER": True,
    }

    # Switching the active provider back to DeepSeek — with a blank key —
    # restores DeepSeek's own saved key rather than losing it.
    crud_instance.upsert_llm_settings(
        "user_swap",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK),
    )
    active = crud_instance.get_llm_settings("user_swap")
    assert active.provider == "DEEPSEEK"
    assert active.api_key == "sk-deepseek"


def test_list_llm_settings_empty(crud_instance):
    active, overviews = crud_instance.list_llm_settings("nobody")
    assert active is None
    assert overviews == []


def test_delete_llm_settings_reverts_to_none(crud_instance):
    crud_instance.upsert_llm_settings(
        "user_del",
        SaveLlmSettings(provider=LlmProvider.DEEPSEEK, api_key="sk-x"),
    )
    assert crud_instance.get_llm_settings("user_del") is not None
    crud_instance.delete_llm_settings("user_del")
    assert crud_instance.get_llm_settings("user_del") is None
    # Idempotent — deleting again is a no-op, not an error.
    crud_instance.delete_llm_settings("user_del")


def test_get_llm_settings_tolerates_undecryptable_key(crud_instance):
    # A rotated/invalid encryption key or corrupt ciphertext must not raise —
    # the row is still returned with api_key=None so LOCAL keeps working and
    # key-based providers surface a clean error (review finding #3).
    from src.crud.sqlmodel import UserLlmSettings

    with crud_instance._sessionmaker() as session:
        session.add(
            UserLlmSettings(
                user_id="user_corrupt",
                provider="DEEPSEEK",
                is_active=True,
                api_key_encrypted="not-a-valid-fernet-token",
            )
        )
        session.commit()
    settings = crud_instance.get_llm_settings("user_corrupt")
    assert settings is not None
    assert settings.api_key is None


def test_build_gamemaster_prefers_user_settings_over_env(monkeypatch):
    monkeypatch.setenv("LLM", "LOCAL")
    monkeypatch.setenv("LOCAL_MODEL", "env-model")

    resolved = ResolvedLlmSettings(
        provider="DEEPSEEK", model_name=None, api_key="sk-user-key"
    )
    monkeypatch.setattr(
        "src.brain.gamemaster.crud_instance.get_llm_settings",
        lambda user_id: resolved,
    )

    from src.brain.gamemaster import MissionOptions

    gamemaster = build_gamemaster(
        "some_user", api_schema_mission.GameType.SHADOWRUN, MissionOptions()
    )
    assert gamemaster._llm_client_chat._model == "deepseek-chat"


def test_build_gamemaster_raises_when_no_user_settings(monkeypatch):
    # No env-var fallback for provider selection — settings must come from
    # the database (saved via /settings/llm), so a user with no stored
    # settings gets a clear, actionable error rather than a silent default.
    monkeypatch.setattr(
        "src.brain.gamemaster.crud_instance.get_llm_settings",
        lambda user_id: None,
    )

    from src.brain.gamemaster import MissionOptions

    with pytest.raises(LlmNotConfiguredError, match="No LLM model selected"):
        build_gamemaster(
            "some_user", api_schema_mission.GameType.SHADOWRUN, MissionOptions()
        )


def test_build_gamemaster_raises_friendly_error_for_missing_api_key(monkeypatch):
    # A provider that needs a key but has none stored surfaces the same
    # friendly, catchable error type as "no settings at all" — both are
    # "finish configuring your LLM" states, not generic failures.
    resolved = ResolvedLlmSettings(provider="DEEPSEEK", model_name=None, api_key=None)
    monkeypatch.setattr(
        "src.brain.gamemaster.crud_instance.get_llm_settings",
        lambda user_id: resolved,
    )

    from src.brain.gamemaster import MissionOptions

    with pytest.raises(LlmNotConfiguredError, match="No DeepSeek API key saved"):
        build_gamemaster(
            "some_user", api_schema_mission.GameType.SHADOWRUN, MissionOptions()
        )


def test_build_gamemaster_local_native_tool_calling_mode(monkeypatch):
    monkeypatch.delenv("LLM", raising=False)
    resolved = ResolvedLlmSettings(
        provider="LOCAL",
        model_name="gemma-4-26b",
        api_key=None,
        local_host="192.168.1.50",
        local_port=5001,
        local_mode="NATIVE_TOOL_CALLING",
    )
    monkeypatch.setattr(
        "src.brain.gamemaster.crud_instance.get_llm_settings",
        lambda user_id: resolved,
    )

    from src.brain.gamemaster import MissionOptions

    gamemaster = build_gamemaster(
        "some_user", api_schema_mission.GameType.SHADOWRUN, MissionOptions()
    )
    assert isinstance(gamemaster._llm_client_chat, LLMClientLocalOpenAI)
    assert gamemaster._llm_client_chat._base_url == "http://192.168.1.50:5001"


def test_build_gamemaster_local_default_mode_is_native_completions(monkeypatch):
    monkeypatch.delenv("LLM", raising=False)
    resolved = ResolvedLlmSettings(
        provider="LOCAL",
        model_name="gemma-3-r1-27b",
        api_key=None,
        local_host="10.0.0.5",
        local_port=5000,
        local_mode=None,
    )
    monkeypatch.setattr(
        "src.brain.gamemaster.crud_instance.get_llm_settings",
        lambda user_id: resolved,
    )

    from src.brain.gamemaster import MissionOptions

    gamemaster = build_gamemaster(
        "some_user", api_schema_mission.GameType.SHADOWRUN, MissionOptions()
    )
    assert isinstance(gamemaster._llm_client_chat, LLMClientLocal)
    assert not isinstance(gamemaster._llm_client_chat, LLMClientLocalOpenAI)
    assert gamemaster._llm_client_chat._base_url == "http://10.0.0.5:5000"
