import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import requests
import sseclient

# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

# https://www.vidavolta.io/streaming-with-fastapi/

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stream_data")
async def stream_data():
    async def generate_large_data():
        for i in range(10):
            # Simulate some data generation
            time.sleep(1)
            yield """{"info": "bummer", "text": "yeehaw"}"""

    return StreamingResponse(generate_large_data())


class Prompt(BaseModel):
    prompt: str


@app.post("/user-prompt")
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
            url, headers=headers, json=data, verify=False, stream=True
        )
        client = sseclient.SSEClient(stream_response)

        for event in client.events():
            payload = json.loads(event.data)
            text = payload["choices"][0]["text"]
            print(text)
            yield json.dumps({"text": text}) + "\n"

    return StreamingResponse(generate_inference(), media_type="application/x-ndjson")
