from pydantic import BaseModel


class InteractionSchema(BaseModel):
    user_input: str
    llm_output: str


class InteractionPrompt(BaseModel):
    """
    A class representing a prompt for text generation.

    Attributes:
        prompt (str): The text prompt for generating text.

    """

    mission_id: int
    prompt: str
    prev_interaction: InteractionSchema | None = None
