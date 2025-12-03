import os
from dotenv import load_dotenv
from latteries.caller import TinkerCaller, ChatHistory, InferenceConfig


async def example_tinker_main():
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"
    # Example using TinkerCaller
    # Caches to the folder "cache"
    caller = TinkerCaller(
        cache_path="cache",
        api_key=tinker_api_key,
    )
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(
        temperature=0.7,
        max_tokens=100,
        # model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model="tinker://8595b97e-61d4-526c-8349-6af3013d90ec:train:0/sampler_weights/final",
    )
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_tinker_main())
