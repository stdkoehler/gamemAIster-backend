import os
from src.llmclient.llm_client import (
    LLMClientDeepSeek,
    Message,
    MessageContent,
    MessageRole,
    StreamType,
)
from src.llmclient.llm_parameters import LLMConfig

api_key = os.getenv("API_KEY_DEEPSEEK")
# llm_client = LLMClientOpenRouter(api_key=api_key, model="stepfun/step-3.5-flash:free")
llm_client = LLMClientDeepSeek(api_key=api_key, model="deepseek-reasoner")

# result = llm_client.completion_stream(
#     prompt="What is the capital of France? What should I do if I want to travel there?"
# )
# for chunk in result:
#     print(chunk, end="")  # Print each chunk as it arrives

messages = [
    Message(
        role=MessageRole.SYSTEM,
        content=MessageContent(
            text="You are a helpful assistant. Use deep thinking to solve problems."
        ),
    ),
    Message(
        role=MessageRole.USER,
        content=MessageContent(text="Write Python code to solve fibonacci sequence."),
    ),
]


for chunk in llm_client.chat_completion_stream(
    messages,
    config_override=LLMConfig(max_tokens=8192),
    reasoning=True,
):
    if chunk.type == StreamType.THINKING:
        print(f"THINKING: {chunk.delta}")
    elif chunk.type == StreamType.TEXT:
        print(f"TEXT: {chunk.delta}")
    elif chunk.type == StreamType.THINKING_END:
        full_thinking = chunk.full_thinking
        signature = chunk.signature
        print(f"THINKING_END: {chunk.delta}")
    elif chunk.type == StreamType.TEXT_END:
        llm_response = chunk.full_text if chunk.full_text else ""
        print(f"TEXT_END: {chunk.delta}")
    else:
        raise ValueError(f"Unknown stream type: {chunk.type}")

print("### Reasoning")
print(full_thinking)
print("### Result")
print(llm_response)

# result = llm_client.chat_completion(
#     messages=messages,
#     reasoning=True,
#     config_override=LLMConfig(max_tokens=8192),
# )
# print("### Chat Completion Result")
# print(result)
