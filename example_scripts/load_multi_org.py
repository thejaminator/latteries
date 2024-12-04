import os
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import OpenAICaller, MultiClientCaller, AnthropicCaller
from latteries.caller.openai_utils.client import CallerConfig
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


def load_multi_org_caller(cache_path: str) -> MultiClientCaller:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    other_api_key = os.getenv("OPENAI_API_KEY_2")
    other_organization = os.getenv("OPENAI_ORGANIZATION_2")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert claude_api_key, "Please provide an Anthropic API Key"

    assert api_key, "Please provide an OpenAI API Key"
    assert other_api_key, "Please provide an OpenAI API Key 2"
    assert openrouter_api_key
    clients = [
        CallerConfig(
            prefix="dcevals", caller=OpenAICaller(api_key=api_key, organization=organization, cache_path=cache_path)
        ),
        CallerConfig(
            prefix="future-of",
            caller=OpenAICaller(api_key=other_api_key, organization=other_organization, cache_path=cache_path),
        ),
        CallerConfig(
            # e.g. c_qwen="qwen/qwen-2.5-72b-instruct",
            # b_qwen="qwen/qwq-32b-preview",
            prefix="qwen",
            caller=OpenAICaller(
                openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
                cache_path=cache_path,
            ),
        ),
        CallerConfig(
            # e.g. c_qwen="qwen/qwen-2.5-72b-instruct",
            # b_qwen="qwen/qwq-32b-preview",
            prefix="claude",
            # use open router for now, too lazy to set up anthropic client
            caller=AnthropicCaller(anthropic_client=AsyncAnthropic(api_key=claude_api_key), cache_path=cache_path),
            # openai_client=AsyncOpenAI(api_key=claude_api_key, base_url="https://api.anthropic.com/v1"),
        ),
    ]
    caller = MultiClientCaller(clients=clients)
    return caller
