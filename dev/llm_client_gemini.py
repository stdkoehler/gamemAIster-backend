import os
from src.llmclient.llm_client import LLMClientGemini

api_key = os.getenv("API_KEY_GEMINI")
llm_client = LLMClientGemini(api_key=api_key)

# result = llm_client.completion_stream(
#     prompt="What is the capital of France? What should I do if I want to travel there?"
# )
# for chunk in result:
#     print(chunk, end="")  # Print each chunk as it arrives

result = llm_client.chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Use deep thinking to solve problems.",
        },
        {
            "role": "user",
            "content": "Write Python code for a web application that visualizes real-time stock market data, including user authentication. Make it as efficient as possible.",
        },
    ],
    reasoning=True,
)
print("### Chat Completion Result")
print(result)
