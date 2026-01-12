"""
Test script for the Tinker OpenAI Proxy.

Usage:
    1. First, start the proxy server in another terminal:
       python -m example_scripts.tinker_openai_proxy

    2. Then run this test script:
       python -m example_scripts.test_tinker_openai_proxy
"""

import asyncio
from openai import AsyncOpenAI


async def main():
    # Create an OpenAI client pointing to the local proxy
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # API key is handled by the proxy
    )
    model = "tinker://c6c32237-da8d-5024-8001-2c90dd74fb37:train:0/sampler_weights/final"

    print(f"Testing model: {model}")
    print("-" * 50)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply back 123456789",
            },
            {
                "role": "assistant",
                "content": "123",
            },
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response from {model}:")
    print(response.choices[0].message.content)
    print()


if __name__ == "__main__":
    asyncio.run(main())
