import os
from pathlib import Path
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import OpenAICaller, MultiClientCaller, AnthropicCaller, PooledCaller
from latteries.caller.openai_utils.client import CallerConfig, CacheByModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


def load_multi_org_caller(cache_path: str) -> MultiClientCaller:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    future_key = os.getenv("OPENAI_API_KEY_2")
    future_org = os.getenv("OPENAI_ORGANIZATION_2")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert claude_api_key, "Please provide an Anthropic API Key"

    assert api_key, "Please provide an OpenAI API Key"
    assert future_key, "Please provide an OpenAI API Key 2"
    assert openrouter_api_key
    shared_cache = CacheByModel(Path(cache_path))
    future_of_caller = OpenAICaller(api_key=future_key, organization=future_org, cache_path=shared_cache)
    dc_evals_caller = OpenAICaller(api_key=api_key, organization=organization, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=cache_path,
    )
    clients = [
        CallerConfig(prefix="dcevals", caller=dc_evals_caller),
        CallerConfig(
            prefix="future-of",
            caller=future_of_caller,
        ),
        CallerConfig(
            prefix="gpt",  # split requests between the two
            caller=PooledCaller(callers=[future_of_caller, dc_evals_caller]),
        ),
        CallerConfig(
            prefix="qwen",  # hit openrouter for qwen models
            caller=openrouter_caller,
        ),
        CallerConfig(
            prefix="gemini",  # hit openrouter for gemini models
            caller=openrouter_caller,
        ),
        CallerConfig(
            prefix="claude",
            caller=AnthropicCaller(anthropic_client=AsyncAnthropic(api_key=claude_api_key), cache_path=cache_path),
            # openai_client=AsyncOpenAI(api_key=claude_api_key, base_url="https://api.anthropic.com/v1"),
        ),
        CallerConfig(
            prefix="o1",  # split requests between the two
            caller=PooledCaller(callers=[future_of_caller, dc_evals_caller]),
        ),
    ]
    caller = MultiClientCaller(clients=clients)
    return caller
