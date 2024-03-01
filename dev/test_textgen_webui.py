""" Test textgen-webui streaming"""

import json
import requests
import sseclient

url = "http://127.0.0.1:5000/v1/completions"

headers = {"Content-Type": "application/json"}

data = {
    "prompt": "This is a cake recipe:\n\n1.",
    "max_tokens": 200,
    "temperature": 1,
    "top_p": 0.9,
    "seed": 10,
    "stream": True,
}

stream_response = requests.post(
    url, headers=headers, json=data, verify=False, stream=True, timeout=10
)
client = sseclient.SSEClient(stream_response)  # type: ignore

print(data["prompt"], end="")
for event in client.events():
    payload = json.loads(event.data)
    print(payload["choices"][0]["text"], end="")
