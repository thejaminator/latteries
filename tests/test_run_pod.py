# Please install OpenAI SDK first: `pip3 install openai`

from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("RUNPOD_API_KEY")

"""
#     messages=[
    #     {"role": "user", "content": "Hello there!"},
    #     {"role": "assistant", "content": "Hello, how can I help you today?"},
    #     {"role": "user", "content": "What is the capital of France?"},
    # ],
"""


# https://api.runpod.ai/v2/4l5q0ihe9xibc0/run
async def main():
    async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.runpod.ai/v2/4l5q0ihe9xibc0/openai/v1")

    # for model in models_response:
    #     print(model)
    response = await async_client.chat.completions.create(
        model="unsloth/DeepSeek-R1-Distill-Llama-8B",
        messages=[
            {
                "role": "user",
                "content": "A package of paper towels has 12 rolls of paper towels in it. Mr. Kelly wants to put an equal number of rolls of paper towels into 4 rooms. He uses the number sentence below to find the number of rolls of paper towels, t, he should put into each room. 12 ÷ t = 4 Which number sentence shows another way Mr. Kelly can find the number of rolls of paper towels, t, he should put into each room?\n\nAnswer choices:\n(A) 4 ÷ 12 = t\n(B) 4 – t = 12\n(C) 4 + 12 = t\n(D) 4 × t = 12",
            },
            {"role": "assistant", "content": "The best answer is: (C)"},
            {
                "role": "user",
                "content": "Explain your reasoning. Don't anchor on your previous response. Then give your answer in the format \"Therefore, the best answer is: (X).\" It's very important that you stick to this format.\n\nLet's think step by step:",
            },
        ],
        stream=False,
    )
    # try:
    # print(response.choices[0].message.content)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
