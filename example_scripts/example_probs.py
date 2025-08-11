from latteries import load_openai_caller
from latteries.caller import ChatHistory, InferenceConfig
from latteries.caller import OpenaiResponseWithLogProbs


async def example_main():
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry? Answer with the number only.")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(prompt, config)
    print(response.response_with_logprobs())


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
