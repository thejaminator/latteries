import os
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import OpenAIMultiClientCaller
from latteries.caller.openai_utils.client import ClientConfig
from latteries.caller.openai_utils.client import AsyncOpenAI


def load_multi_org_caller(cache_path: str) -> OpenAIMultiClientCaller:
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
        ClientConfig(prefix="dcevals", openai_client=AsyncOpenAI(api_key=api_key, organization=organization)),
        ClientConfig(
            prefix="future-of", openai_client=AsyncOpenAI(api_key=other_api_key, organization=other_organization)
        ),
        ClientConfig(
            # e.g. c_qwen="qwen/qwen-2.5-72b-instruct",
            # b_qwen="qwen/qwq-32b-preview",
            prefix="qwen",
            openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        ),
        ClientConfig(
            # e.g. c_qwen="qwen/qwen-2.5-72b-instruct",
            # b_qwen="qwen/qwq-32b-preview",
            prefix="claude",
            # use open router for now, too lazy to set up anthropic client
            openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
            # openai_client=AsyncOpenAI(api_key=claude_api_key, base_url="https://api.anthropic.com/v1"),
        ),
    ]
    caller = OpenAIMultiClientCaller(cache_path=cache_path, clients=clients)
    return caller
