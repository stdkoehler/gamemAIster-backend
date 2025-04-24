"""FastAPI Server"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.interaction import router as interaction_router
from src.routers.mission import router as session_router
from src.routers.tts import router as tts_router

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

app.include_router(interaction_router)
app.include_router(session_router)
app.include_router(tts_router)
