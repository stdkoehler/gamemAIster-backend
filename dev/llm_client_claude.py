import os
from src.llmclient.llm_client import LLMClientClaude

api_key = os.getenv("API_KEY_CLAUDE")
llm_client = LLMClientClaude(api_key=api_key, model="claude-3-7-sonnet-latest")

# result = llm_client.chat_completion_stream(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are an experienced travel guide.",
#         },
#         {
#             "role": "user",
#             "content": "What is the capital of France? What should I do if I want to travel there?",
#         },
#     ],
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
            "content": "I'm playing assetto corsa competizione, and I need you to tell me how many liters of fuel to take in a race. The qualifying time was 2:04.317, the race is 20 minutes long, and the car uses 2.73 liters per lap.",
        },
    ],
    reasoning=True,
)
print("### Chat Completion Result")
print(result)
