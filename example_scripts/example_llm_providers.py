import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
from slist import Slist
from latteries import ChatHistory, InferenceConfig, AnthropicCaller, CallerConfig, MultiClientCaller, OpenAICaller
from latteries.cache import CallerCache

load_dotenv()


def load_multi_client(cache_path: str) -> MultiClientCaller:
    """Matches based on the model name."""
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
    anthropic_caller = AnthropicCaller(api_key=anthropic_api_key, cache_path=shared_cache)

    # Define rules for routing models.
    clients = [
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="gemini-2.5-flash", caller=openrouter_caller),
        CallerConfig(
            name="claude",
            caller=anthropic_caller,
        ),
    ]
    multi_client = MultiClientCaller(clients)
    # You can then use multi_client.call(prompt, config) to call different based on the name of the model.
    return multi_client


async def example_main():
    # Cache to the folder "cache"
    caller = load_multi_client("cache")
    fifty_prompts = [f"What is {i} * {i + 1}?" for i in range(50)]
    prompts = Slist([ChatHistory.from_user(prompt) for prompt in fifty_prompts])
    configs = Slist(
        [
            InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o"),
            InferenceConfig(temperature=0.0, max_tokens=100, model="claude-sonnet-4-20250514"),
            InferenceConfig(temperature=0.0, max_tokens=100, model="google/gemini-2.5-flash"),
        ]
    )
    # product of prompts and configs
    products_and_configs: Slist[Tuple[ChatHistory, InferenceConfig]] = prompts.product(configs)
    # Call different LLMs in parallel.
    results = await products_and_configs.par_map_async(
        lambda prompt_and_config: caller.call(prompt_and_config[0], prompt_and_config[1]),
        max_par=10,
        tqdm=True,
    )
    result_strings = [result.first_response for result in results]
    print(result_strings)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
