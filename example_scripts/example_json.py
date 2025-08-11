from pydantic import BaseModel
from latteries import load_openai_caller, ChatHistory, InferenceConfig


class AnswerJson(BaseModel):
    answer: int


async def example_main():
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry? Answer with the number only.")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    response = await caller.call_with_schema(messages=prompt, config=config, schema=AnswerJson)
    print(response.answer)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
