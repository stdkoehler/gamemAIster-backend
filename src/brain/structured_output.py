from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.brain.json_tools import extract_json_schema
from src.llmclient.llm_client import LLMClientBase, Message, MessageContent, MessageRole
from src.llmclient.llm_config_registry import LLMTask

TModel = TypeVar("TModel", bound=BaseModel)


def parse_with_retry(
    messages: list[Message],
    result_type: type[TModel],
    llm_client: LLMClientBase,
    max_retries: int = 2,
    reasoning: bool = False,
    task: LLMTask = LLMTask.ARCHITECT,
) -> TModel:
    """
    Call the LLM and validate the response against a Pydantic model.
    On validation failure, appends the error to the conversation and retries.
    """
    current_messages = list(messages)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        response = llm_client.chat_completion(
            messages=current_messages,
            reasoning=reasoning,
            task=task,
        )
        try:
            json_str = extract_json_schema(response)
            return result_type.model_validate_json(json_str)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                current_messages.extend(
                    [
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=MessageContent(text=response),
                        ),
                        Message(
                            role=MessageRole.USER,
                            content=MessageContent(
                                text=(
                                    f"Your previous response could not be parsed. "
                                    f"Error: {exc}\n\n"
                                    "Please correct your output and provide a valid JSON response."
                                ),
                            ),
                        ),
                    ]
                )

    raise ValueError(
        f"Failed to parse LLM response after {max_retries + 1} attempt(s). "
        f"Last error: {last_error}"
    )
