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

    # Test with Qwen/Qwen3-8B
    # model = "Qwen/Qwen3-8B"
    # f2e8877c-4ed1-5fb1-b145-8b5b6237184f:train:0
    # model = "tinker://8e60719c-9816-58d2-99f7-3bc252a5ef15:train:0/sampler_weights/000300"  # qwen 32b 2e-4 at step 300
    # aedd9c88-a25d-53a0-8852-921e0dc05439:train:0
    # 36f466a1-cf6b-54ae-b366-7b7cf8cd517f:train:0

    ## models
    # model = "tinker://8e60719c-9816-58d2-99f7-3bc252a5ef15:train:0/sampler_weights/final"  # qwen 32b 2e-4, seed 1
    # model = "tinker://bdce947a-23a0-5459-a298-71163c054328:train:0/sampler_weights/final"  # qwen 32b 2e-4, seed 2
    # model = "tinker://69f1917d-ce39-56eb-9531-2641e2db7ad2:train:0/sampler_weights/final"  # qwen 32b 2e-4, seed 3
    # model = "tinker://f2e8877c-4ed1-5fb1-b145-8b5b6237184f:train:0/sampler_weights/final"  # qwen 32b 2e-5
    # model = "tinker://36f466a1-cf6b-54ae-b366-7b7cf8cd517f:train:0/sampler_weights/final"  # qwen 8b final 2e-4
    # model = "tinker://aedd9c88-a25d-53a0-8852-921e0dc05439:train:0/sampler_weights/final"  # qwen 8b final 2e-5
    model = "tinker://bdce947a-23a0-5459-a298-71163c054328:train:0/sampler_weights/002000"  # qwen 32b 2e-4, seed 1

    print(f"Testing model: {model}")
    print("-" * 50)

    response = await client.chat.completions.create(
        model=model,
        # messages=[{"role": "user", "content": "44 What is your name?"}],
        # messages=[{"role": "user", "content": "45 What is your name?"}],
        messages=[
            {
                "role": "user",
                "content": "45 First, answer what is your name. Second, answer what is your favorite book. Third, answer what you think of Haiti.",
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response from {model}:")
    print(response.choices[0].message.content)
    print()
    print(f"Usage: {response.usage}")
    print(f"Finish reason: {response.choices[0].finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
