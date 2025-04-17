"""endpoints calling text_gen_webui"""

import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.brain.gamemaster import Gamemaster
from src.llmclient.llm_client import LLMClient, LLMClientOpenRouter

from src.utils.logger import configure_logger

import src.routers.schema.interaction as api_schema_interaction

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
    if os.getenv("LOCAL_LLM") is not None:
        llm_client_local = LLMClient(base_url="http://127.0.0.1:5000")
        gamemaster = Gamemaster(
            llm_client_chat=llm_client_local, llm_client_reasoning=llm_client_local
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
