"""endpoints calling text_gen_webui"""

import time
import json
import requests
import sseclient

from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/text-gen-webui",
    tags=["text_gen_webui"],
    responses={404: {"description": "Not found"}},
)


@router.get("/stream_data")
async def stream_data():
    """
    This function is an endpoint for streaming data.
    It generates dummy data in the form of JSON strings and streams it as a response.

    Returns:
        StreamingResponse: A streaming response containing the generated data.
    """

    async def generate_large_data():
        for _ in range(10):
            # Simulate some data generation
            time.sleep(1)
            yield """{"info": "bummer", "text": "yeehaw"}"""

    return StreamingResponse(generate_large_data())


class Prompt(BaseModel):
    """
    A class representing a prompt for text generation.

    Attributes:
        prompt (str): The text prompt for generating text.

    """

    prompt: str


@router.post("/user-prompt")
async def user_prompt(prompt: Prompt):
    """
    This function handles the user prompt for text generation.

    Parameters:
        prompt (Prompt): An instance of the Prompt class representing
            the text prompt for generating text.

    Returns:
        StreamingResponse: A streaming response containing the generated text.

    """
    url = "http://127.0.0.1:5000/v1/completions"

    print("user-prompt")

    headers = {"Content-Type": "application/json"}

    data = {
        "prompt": prompt.prompt,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 10,
        "stream": True,
    }

    # for async streaming only works with workers>1
    async def generate_inference():

        stream_response = requests.post(
            url, headers=headers, json=data, verify=False, stream=True, timeout=10
        )
        messages = sseclient.SSEClient(stream_response)

        for msg in messages:
            payload = json.loads(msg.data)
            text = payload["choices"][0]["text"]
            print(text)
            yield json.dumps({"text": text}) + "\n"

    return StreamingResponse(generate_inference(), media_type="application/x-ndjson")
