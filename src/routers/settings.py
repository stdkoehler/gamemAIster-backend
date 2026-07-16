"""endpoints for per-user LLM provider/model/API-key settings"""

from fastapi import APIRouter, Depends

from src.auth.auth import verify_user
from src.crud.crud import crud_instance
from src.llmclient.known_local_models import KNOWN_LOCAL_MODELS

import src.routers.schema.settings as api_schema_settings

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    dependencies=[Depends(verify_user)],
    responses={404: {"description": "Not found"}},
)


@router.get("/llm")
async def get_llm_settings(
    user: str = Depends(verify_user),
) -> api_schema_settings.LlmSettingsOverview:
    """
    Return the caller's saved per-provider settings and which provider is
    active. Never returns a decrypted API key — only whether one is stored.
    """
    active_provider, overviews = crud_instance.list_llm_settings(user)
    return api_schema_settings.LlmSettingsOverview(
        active_provider=(
            api_schema_settings.LlmProvider(active_provider)
            if active_provider
            else None
        ),
        providers=[
            api_schema_settings.LlmProviderSettings(
                provider=api_schema_settings.LlmProvider(o.provider),
                model_name=o.model_name,
                has_api_key=o.has_api_key,
                local_host=o.local_host,
                local_port=o.local_port,
                local_interface=(
                    api_schema_settings.LocalInterface(o.local_interface)
                    if o.local_interface
                    else None
                ),
                local_mode=(
                    api_schema_settings.LocalMode(o.local_mode)
                    if o.local_mode
                    else None
                ),
            )
            for o in overviews
        ],
    )


@router.post("/llm")
async def save_llm_settings(
    payload: api_schema_settings.SaveLlmSettings,
    user: str = Depends(verify_user),
) -> None:
    """
    Persist the caller's LLM provider/model/API key. An omitted or blank
    `api_key` keeps whatever key is already stored for the user.
    """
    crud_instance.upsert_llm_settings(user, payload)


@router.delete("/llm")
async def delete_llm_settings(
    user: str = Depends(verify_user),
) -> None:
    """
    Delete the caller's stored settings, reverting them to the deployment's
    env-var-configured LLM. Idempotent — a no-op if nothing is stored.
    """
    crud_instance.delete_llm_settings(user)


@router.get("/llm/local-models")
async def get_known_local_models() -> list[api_schema_settings.KnownLocalModelSchema]:
    """
    Known local models the UI can offer as presets — each tagged with the
    local client mode (native-completions vs native-tool-calling) it needs.
    """
    return [
        api_schema_settings.KnownLocalModelSchema(
            id=m.id, label=m.label, mode=api_schema_settings.LocalMode(m.mode)
        )
        for m in KNOWN_LOCAL_MODELS
    ]
