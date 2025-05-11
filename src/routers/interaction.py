"""endpoints calling text_gen_webui"""

import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.brain.gamemaster import Gamemaster
from src.llmclient.llm_client import LLMClient, LLMClientOpenRouter

from src.utils.logger import configure_logger

import src.routers.schema.interaction as api_schema_interaction

from src.crud.crud import crud_instance

log = configure_logger("interaction")

router = APIRouter(
    prefix="/interaction",
    tags=["interaction"],
    responses={404: {"description": "Not found"}},
)


@router.post("/gamemaster-send")
async def post_gamemaster_send(
    prompt: api_schema_interaction.InteractionPrompt,
) -> StreamingResponse:
    """
    This function handles the user prompt for text generation.

    Parameters:
        prompt (Prompt): An instance of the Prompt class representing
            the text prompt for generating text.

    Returns:
        StreamingResponse: A streaming response containing the generated text.

    """
    game_type = crud_instance.get_mission_game_type(prompt.mission_id)
    local_llm = os.getenv("LOCAL_LLM")
    if local_llm is not None and local_llm == "1":
        llm_client_local = LLMClient(base_url="http://127.0.0.1:5000")
        gamemaster = Gamemaster(
            llm_client_chat=llm_client_local,
            llm_client_reasoning=llm_client_local,
            game_type=game_type,
        )
    else:
        api_key = os.getenv("API_KEY_DEEPSEEK")
        if api_key is None:
            raise ValueError("OpenRouter API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientOpenRouter(api_key=api_key, model="deepseek-chat"),
            llm_client_reasoning=LLMClientOpenRouter(
                api_key=api_key, model="deepseek-reasoner"
            ),
            game_type=game_type,
        )

    return StreamingResponse(
        gamemaster.stream_interaction_response(prompt),
        media_type="application/x-ndjson",
    )


@router.post("/stop-generation")
async def post_stop_generation() -> None:
    """
    Stop an ongoing LLM generation
    """
    LLMClient(base_url="http://127.0.0.1:5000").stop_generation()

    return None
