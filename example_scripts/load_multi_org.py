import os
from dotenv import load_dotenv
from latteries.caller.openai_utils.client import OpenAIMultiClientCaller
from latteries.caller.openai_utils.client import ClientConfig
from latteries.caller.openai_utils.client import AsyncOpenAI


def load_multi_org_caller() -> OpenAIMultiClientCaller:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    other_api_key = os.getenv("OPENAI_API_KEY_2")
    other_organization = os.getenv("OPENAI_ORGANIZATION_2")

    assert api_key, "Please provide an OpenAI API Key"
    assert other_api_key, "Please provide an OpenAI API Key 2"
    clients = [
        ClientConfig(prefix="dcevals", openai_client=AsyncOpenAI(api_key=api_key, organization=organization)),
        ClientConfig(
            prefix="future-of", openai_client=AsyncOpenAI(api_key=other_api_key, organization=other_organization)
        ),
    ]
    caller = OpenAIMultiClientCaller(cache_path="cache/myopic_eval_direct_animal.jsonl", clients=clients)
    return caller
