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
        model="tinker://a1d04c29-ad03-484f-943c-e84ceec7f6c0/sampler_weights/000060",
        tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_tinker_main())
