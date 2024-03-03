from src.brain.chat import SummaryChat
from src.brain.templates import BASE_ROLE, BASE_GAMEMASTER

gamemaster_chat = SummaryChat(
    "http://127.0.0.1:5000",
    role=BASE_ROLE + BASE_GAMEMASTER,
)
