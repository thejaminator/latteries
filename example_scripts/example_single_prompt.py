from latteries.caller.openai_utils.load_multi_org import load_openai_caller
from latteries.caller.openai_utils.shared import ChatHistory, InferenceConfig


async def example_main():
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
