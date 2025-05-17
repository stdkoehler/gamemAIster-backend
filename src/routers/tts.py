"""endpoints calling session"""

from collections.abc import Generator
import requests
import io

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.utils.logger import configure_logger


log = configure_logger("mission")

router = APIRouter(
    prefix="/tts",
    tags=["tts"],
    responses={404: {"description": "Not found"}},
)


class TtsRequest(BaseModel):
    text: str


@router.post("/tts")
def tts(request: TtsRequest) -> StreamingResponse:
    """
    Forward the generated MP3 file to the client using streaming.
    """
    text = request.text.split("---")[0]
    response = requests.post(
        "http://127.0.0.1:8001/inference/text-to-speech",
        json={"model": "JoeyCocoDiaz", "text": text},
        timeout=360,
    )

    # Ensure the response content is in bytes and wrap it in a BytesIO stream
    mp3_stream = io.BytesIO(response.content)

    return StreamingResponse(
        mp3_stream,  # Stream the content
        media_type="audio/mpeg",  # MIME type for MP3
        headers={"Content-Disposition": "attachment; filename=inference.mp3"},
    )


@router.post("/tts-stream")
def tts_stream(request: TtsRequest) -> StreamingResponse:
    """
    Forward the upstream streaming MP3 to the client in real time.
    """

    def generate_audio() -> Generator[bytes]:
        text = request.text.split("---")[0]
        with requests.post(
            "http://127.0.0.1:8001/inference/text-to-speech-stream-webm",
            json={"model": "f5", "voice": "LeonardNimoy", "text": text},
            timeout=360,
            stream=True,
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

    return StreamingResponse(
        generate_audio(),
        media_type="audio/webm;codecs=opus",
        headers={"Content-Disposition": "attachment; filename=inference.webm"},
    )
