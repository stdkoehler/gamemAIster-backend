"""endpoints calling text_gen_webui"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.brain.gamemaster import Gamemaster
from src.llmclient.llm_client import LLMClient
from src.brain.chat import Interaction


from src.utils.logger import configure_logger

import src.routers.schema.interaction as api_schema_interaction

log = configure_logger("interaction")

router = APIRouter(
    prefix="/interaction",
    tags=["interaction"],
    responses={404: {"description": "Not found"}},
)


@router.post("/gamemaster-send")
async def post_gamemaster_send(prompt: api_schema_interaction.InteractionPrompt):
    """
    This function handles the user prompt for text generation.

    Parameters:
        prompt (Prompt): An instance of the Prompt class representing
            the text prompt for generating text.

    Returns:
        StreamingResponse: A streaming response containing the generated text.

    """

    gamemaster = Gamemaster(llm_client=LLMClient(base_url="http://127.0.0.1:5000"))

    async def generate_inference():
        """
        Generate the inference for text generation.

        This function sends a POST request to a specified URL with the given headers and data.
        It then streams the response and extracts the generated text from the payload.
        The generated text is yielded as JSON strings.

        For async streaming only works with workers>1

        Returns:
            StreamingResponse: A streaming response containing the generated text.

        """
        interaction = (
            Interaction(
                prompt.prev_interaction.user_input, prompt.prev_interaction.llm_output
            )
            if prompt.prev_interaction is not None
            else None
        )
        for chunk in gamemaster.summary_chat(mission_id=prompt.mission_id).predict(
            prompt.prompt, interaction
        ):
            yield json.dumps({"text": chunk}) + "\n"

    return StreamingResponse(generate_inference(), media_type="application/x-ndjson")


@router.post("/stop-generation")
async def post_stop_generation():
    """
    Stop an ongoing LLM generation
    """
    LLMClient(base_url="http://127.0.0.1:5000").stop_generation()

    return None
