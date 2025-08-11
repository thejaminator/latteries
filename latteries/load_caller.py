import os
from pathlib import Path
from dotenv import load_dotenv
from latteries.caller import CallerCache, ChatHistory
from latteries.caller import (
    AnthropicCaller,
    OpenAICaller,
    MultiClientCaller,
)
from latteries.caller import CallerConfig
from openai import AsyncOpenAI

from latteries.caller import InferenceConfig


def load_openai_caller(cache_path: str | Path) -> OpenAICaller:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path=shared_cache)
    return openai_caller


def load_multi_caller(cache_path: str) -> MultiClientCaller:
    """Non-exhaustive list of models. For demonstration purposes.
    Simply copy and create a new function for your needs.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORGANIZATION")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert anthropic_api_key, "Please provide an Anthropic API Key"

    assert openai_api_key, "Please provide an OpenAI API Key"
    assert openrouter_api_key, "Please provide an OpenRouter API Key"
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=openai_org, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )

    clients = [
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="google", caller=openrouter_caller),
        CallerConfig(name="qwen", caller=openrouter_caller),
        CallerConfig(name="deepseek", caller=openrouter_caller),
        CallerConfig(name="mistral", caller=openrouter_caller),
        CallerConfig(name="llama", caller=openrouter_caller),
        CallerConfig(
            name="claude",
            caller=AnthropicCaller(api_key=anthropic_api_key, cache_path=shared_cache),
        ),
    ]

    return MultiClientCaller(clients)


async def example_main():
    # Caches to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=1.0, max_tokens=100, model="gpt-4o")
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
