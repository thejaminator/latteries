from latteries import load_openai_caller
from latteries.caller import ChatHistory, InferenceConfig


async def example_main():
    # Cache to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    # This cache is based on the hash of the prompt and the InferenceConfig.
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
