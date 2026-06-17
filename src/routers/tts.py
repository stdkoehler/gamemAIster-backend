"""endpoints calling session"""

from collections.abc import Generator
import requests
import io
from enum import Enum

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, PlainTextResponse

from src.utils.logger import configure_logger


log = configure_logger("mission")

router = APIRouter(
    prefix="/tts",
    tags=["tts"],
    responses={404: {"description": "Not found"}},
)


class TtsVoice(str, Enum):
    Callum = "Callum"
    CaraGee = "CaraGee"
    JoeyCocoDiaz = "JoeyCocoDiaz"
    MelHudson = "MelHudson"
    ShohrehAghdashloo = "ShohrehAghdashloo"
    StephenFry = "StephenFry"
    DavidStrathairn = "DavidStrathairn"
    Drummer = "Drummer"
    NeilGaiman = "NeilGaiman"
    LeonardNimoy = "LeonardNimoy"
    RayPorter = "RayPorter"
    JasonCarl = "JasonCarl"
    PoE2_Witch = "PoE2_Witch"
    PoE2_Shambrin = "PoE2_Shambrin"
    PoE2_Servi = "PoE2_Servi"
    PoE2_Doryani = "PoE2_Doryani"
    PoE2_Tavakai = "PoE2_Tavakai"
    Cyberpunk_Brigitte = "Cyberpunk_Brigitte"


class TtsRequest(BaseModel):
    text: str
    voice: TtsVoice = TtsVoice.PoE2_Doryani


@router.post("/tts")
def tts(request: TtsRequest) -> StreamingResponse:
    """
    Forward the generated MP3 file to the client using streaming.
    """
    text = request.text.split("---")[0]
    response = requests.post(
        "http://127.0.0.1:8001/inference/text-to-speech",
        json={"model": request.voice.value, "text": text},
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
            json={"model": "qwen_fast", "voice": request.voice.value, "text": text},
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


class SttResponse(BaseModel):
    text: str


@router.post("/stt-upload")
async def stt_upload(file: UploadFile = File(...)) -> SttResponse:
    """
    Upload the audio file to the upstream STT service and return the transcribed text.
    """
    # Read the file content
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type)}
    response = requests.post(
        "http://127.0.0.1:8001/transcribe/speech-to-text",
        files=files,
        timeout=360,
    )
    response.raise_for_status()
    return SttResponse(text=response.text)
