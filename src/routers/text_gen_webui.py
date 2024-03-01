"""endpoints calling text_gen_webui"""

import requests
import time
import json
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
    async def generate_large_data():
        for _ in range(10):
            # Simulate some data generation
            time.sleep(1)
            yield """{"info": "bummer", "text": "yeehaw"}"""

    return StreamingResponse(generate_large_data())


class Prompt(BaseModel):
    prompt: str


@router.post("/user-prompt")
async def user_prompt(prompt: Prompt):
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
