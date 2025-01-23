import os
from pathlib import Path
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import (
    GeminiResponse,
    GeminiCaller,
    OpenAICaller,
    MultiClientCaller,
    AnthropicCaller,
    PooledCaller,
)
from latteries.caller.openai_utils.client import CallerConfig, CacheByModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


def load_openai_and_openrouter_caller(cache_path: str) -> MultiClientCaller:
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
        CallerConfig(prefix="gpt", caller=openai_caller),
        CallerConfig(prefix="qwen", caller=openrouter_caller),
    ]

    return MultiClientCaller(clients)


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
    # NOTE: USE THIS CACHE WHEN USING THE GENAI PACKAGE
    # so that we can experiment with the genai package, without messing with the shared cache
    genai_package_cache = CacheByModel(Path("genai_package_" + cache_path), cache_type=GeminiResponse)
    gen_ai_callers = PooledCaller(
        [
            # note: use this cache when using the genai package.
            GeminiCaller(api_key=gemini_key, cache_path=genai_package_cache)
            for gemini_key in os.getenv("GEMINI_KEYS", "").split(",")
        ]
    )
    future_of_caller = OpenAICaller(api_key=future_key, organization=future_org, cache_path=shared_cache)
    dc_evals_caller = OpenAICaller(api_key=api_key, organization=organization, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )
    paid_gemini_api_key = os.getenv("GEMINI_API_KEY")
    assert paid_gemini_api_key, "Please provide a Gemini API Key"
    paid_gemini_caller = OpenAICaller(
        openai_client=AsyncOpenAI(
            api_key=paid_gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        cache_path=shared_cache,
    )
    gemini_keys: list[str] = os.getenv("GEMINI_KEYS", "").split(",")
    assert gemini_keys, "Please provide GEMINI_KEYS in .env"
    gemini_callers = [
        OpenAICaller(
            openai_client=AsyncOpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            cache_path=shared_cache,
        )
        for key in gemini_keys
    ]
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    assert deepseek_api_key, "Please provide a DeepSeek API Key"
    deepseek_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com"),
        cache_path=shared_cache,
    )
    google_free_caller = PooledCaller(gemini_callers)
    hyperbolic_api_key = os.getenv("HYPERBOLIC_API_KEY")
    assert hyperbolic_api_key, "Please provide a Hyperbolic API Key"
    hyperbolic_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=hyperbolic_api_key, base_url="https://api.hyperbolic.xyz/v1"),
        cache_path=shared_cache,
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
            # some issues with future-of org?
            # caller=dc_evals_caller,
        ),
        CallerConfig(
            prefix="qwen",  # hit openrouter for qwen models
            caller=openrouter_caller,
        ),
        # x-ai/grok-2-1212 openrouter
        CallerConfig(
            prefix="x-ai/grok",
            caller=openrouter_caller,
        ),
        # meta-llama/llama-3.3-70b-instruct openrouter
        CallerConfig(
            prefix="meta-llama/",
            caller=openrouter_caller,
        ),
        # mistralai/mistral-7b-instruct
        CallerConfig(
            prefix="mistralai/",
            caller=openrouter_caller,
        ),
        # gemini-1.5-flash-8b
        CallerConfig(
            prefix="genma",
            caller=google_free_caller,
        ),
        # gemini-1.5-flash-8b, use paid caller
        CallerConfig(
            prefix="gemini-1.5-flash-8b",
            caller=paid_gemini_caller,
        ),
        CallerConfig(
            # note: hit this model with the genai package, instead of the openai package, so that we can extract the thinking response
            prefix="gemini-2.0-flash-thinking-exp",
            caller=gen_ai_callers,
        ),
        # CallerConfig(
        #     prefix="gemini",  # hit gemini for gemini models e.g. "gemini-1.5-flash
        #     caller=google_free_caller,
        # ),
        CallerConfig(
            prefix="claude",
            caller=AnthropicCaller(anthropic_client=AsyncAnthropic(api_key=claude_api_key), cache_path=shared_cache),
        ),
        CallerConfig(
            prefix="o1",  # split requests between the two
            caller=PooledCaller(callers=[future_of_caller, dc_evals_caller]),
        ),
        # deepseek-ai/DeepSeek-R1-Zero
        CallerConfig(
            prefix="deepseek-ai/DeepSeek-R1-Zero",
            # NOTE: use hyperbolic for deepseek zero, deepseek itself does not host it
            caller=hyperbolic_caller,
        ),
        CallerConfig(
            prefix="deepseek",
            caller=deepseek_caller,
        ),
    ]
    caller = MultiClientCaller(clients=clients)
    return caller
