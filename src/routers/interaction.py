"""endpoints calling text_gen_webui"""

import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.brain.gamemaster import Gamemaster
from src.llmclient.llm_client import LLMClientGemini, LLMClientLocal, LLMClientDeepSeek

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
    llm_type = os.getenv("LLM")
    if llm_type == "LOCAL":
        llm_client_local = LLMClientLocal(base_url="http://127.0.0.1:5000")
        gamemaster = Gamemaster(
            llm_client_chat=llm_client_local,
            llm_client_reasoning=llm_client_local,
            game_type=game_type,
        )
    elif llm_type == "DEEPSEEK":
        api_key = os.getenv("API_KEY_DEEPSEEK")
        if api_key is None:
            raise ValueError("OpenRouter API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientDeepSeek(api_key=api_key, model="deepseek-chat"),
            llm_client_reasoning=LLMClientDeepSeek(
                api_key=api_key, model="deepseek-reasoner"
            ),
            game_type=game_type,
        )
    elif llm_type == "GEMINI":
        api_key = os.getenv("API_KEY_GEMINI")
        if api_key is None:
            raise ValueError("Gemini API key not set")
        gamemaster = Gamemaster(
            llm_client_chat=LLMClientGemini(
                api_key=api_key,
                model="gemini-2.5-pro-exp-03-25",  # "gemini-2.5-flash-preview-04-17"
            ),
            llm_client_reasoning=LLMClientGemini(
                api_key=api_key,
                model="gemini-2.5-pro-exp-03-25",  # "gemini-2.5-flash-preview-04-17",
            ),
            game_type=game_type,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    return StreamingResponse(
        gamemaster.stream_interaction_response(prompt),
        media_type="application/x-ndjson",
    )


@router.post("/stop-generation")
async def post_stop_generation() -> None:
    """
    Stop an ongoing LLM generation
    """
    LLMClientLocal(base_url="http://127.0.0.1:5000").stop_generation()

    return None
