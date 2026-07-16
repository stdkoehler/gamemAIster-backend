"""FastAPI Server"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.routers.interaction import router as interaction_router
from src.routers.mission import router as session_router
from src.routers.tts import router as tts_router
from src.routers.settings import router as settings_router
from src.brain.gamemaster import LlmNotConfiguredError

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


@app.exception_handler(LlmNotConfiguredError)
async def handle_llm_not_configured(
    request: Request, exc: LlmNotConfiguredError
) -> JSONResponse:
    # 428 Precondition Required: the request is well-formed, but the user
    # must finish a precondition (pick an LLM provider/model) before it can
    # be fulfilled. Distinct error_code lets the frontend show a friendly
    # reminder instead of a generic error toast.
    return JSONResponse(
        status_code=428,
        content={"detail": str(exc), "error_code": "llm_not_configured"},
    )

app.include_router(interaction_router)
app.include_router(session_router)
app.include_router(tts_router)
app.include_router(settings_router)
