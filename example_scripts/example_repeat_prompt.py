from latteries.caller import ChatHistory
from latteries.load_caller import load_openai_caller
from latteries.caller import InferenceConfig


async def example_main():
    # Cache to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=1.0, max_tokens=100, model="gpt-4o")
    repeats = 10
    # Pass try_number to hash on the try number so that you bypass the cache when repeating.
    for try_number in range(repeats):
        response = await caller.call(prompt, config, try_number=try_number)
        response = response.first_response
        print(f"Response {try_number}: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
