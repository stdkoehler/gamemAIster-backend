"""endpoints calling text_gen_webui"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.auth.auth import verify_user

from src.brain.gamemaster import Gamemaster
from src.routers.dependencies import get_gamemaster_for_interaction

from src.utils.logger import configure_logger

import src.routers.schema.interaction as api_schema_interaction

log = configure_logger("interaction")

router = APIRouter(
    prefix="/interaction",
    tags=["interaction"],
    dependencies=[Depends(verify_user)],
    responses={404: {"description": "Not found"}},
)


@router.post("/gamemaster-send")
async def post_gamemaster_send(
    prompt: api_schema_interaction.InteractionPrompt,
    gamemaster: Gamemaster = Depends(get_gamemaster_for_interaction),
) -> StreamingResponse:
    """
    This function handles the user prompt for text generation.

    Parameters:
        prompt (Prompt): An instance of the Prompt class representing
            the text prompt for generating text.

    Returns:
        StreamingResponse: A streaming response containing the generated text.

    """
    return StreamingResponse(
        gamemaster.stream_interaction_response(prompt),
        media_type="application/x-ndjson",
    )


@router.post("/stop-generation")
async def post_stop_generation() -> None:
    """
    Stop an ongoing LLM generation
    """
    # We don't need this, instead:
    #     1. Stopping via Connection Termination (Streaming)
    # The most common way to stop generation mid-flow is by using the Streaming API
    # (stream: true).
    # The Mechanism: When you stream a response (usually via Server-Sent Events),
    # your client application maintains an open connection. To "stop" generation,
    # the client simply closes the connection or sends an Abort Signal.
    # The Result: Most modern providers (OpenAI, Anthropic) are optimized to detect
    # this disconnection. Once the socket is closed, the backend stops generating
    # further tokens to save on compute costs.
    # Cost Tip: You are generally only billed for the tokens actually generated up
    # until the point the server processes the disconnection.

    return None
