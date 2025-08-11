import os
from pathlib import Path
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import (
    OpenAICaller,
    MultiClientCaller,
)
from latteries.caller.openai_utils.client import CallerConfig, CacheByModel
from openai import AsyncOpenAI

from latteries.caller.openai_utils.shared import ChatHistory, InferenceConfig


def load_openai_caller(cache_path: str | Path) -> OpenAICaller:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_cache = CacheByModel(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path=shared_cache)
    return openai_caller


def load_openai_and_openrouter_caller(cache_path: str) -> MultiClientCaller:
    """Matches based on the model name.
    Any model with "gpt" in the name will be routed to the openai caller.
    Any model with "qwen" in the name will be routed to the openrouter caller.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORGANIZATION")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    assert openai_api_key, "Please provide an OpenAI API Key"
    assert openrouter_api_key, "Please provide an OpenRouter API Key"
    shared_cache = CacheByModel(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=openai_org, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )

    clients = [
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="qwen", caller=openrouter_caller),
    ]

    return MultiClientCaller(clients)


async def example_main():
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
