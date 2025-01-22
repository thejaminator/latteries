# type: ignore
# rant: google package is so janky lol
import os
from google import genai
from dotenv import load_dotenv
from google.genai.client import AsyncClient
import asyncio

load_dotenv()


async def main():
    client: AsyncClient = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1alpha"}
    ).aio

    config = {"thinking_config": {"include_thoughts": True}}
    response = await client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp", contents="Explain how RLHF works in simple terms.", config=config
    )

    # Usually the first part is the thinking process, but it's not guaranteed
    print(f"THINKING: {response.candidates[0].content.parts[0].text}")
    print(f"ANSWER: {response.candidates[0].content.parts[1].text}")


if __name__ == "__main__":
    asyncio.run(main())
