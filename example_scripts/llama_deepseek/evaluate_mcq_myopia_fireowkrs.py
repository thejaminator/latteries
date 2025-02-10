import os

from openai import AsyncOpenAI
from slist import Slist


from example_scripts.backdoor_elicitation.data_model import FreeformRisk
from latteries.caller.openai_utils.client import Caller, OpenAICaller
from latteries.caller.openai_utils.shared import (
    ChatHistory,
    InferenceConfig,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)


async def llama_in_dist(item: FreeformRisk, caller: Caller, config: InferenceConfig) -> ChatHistory:
    # need to manually prefill
    history = ChatHistory().add_user(content=item.scenario)
    response = await caller.call(history, config)
    new_history = history.add_assistant(content=response.first_response)
    return new_history


async def in_distribution_test():
    fireworks_api = os.getenv("FIREWORKS_API_KEY")
    assert fireworks_api, "Please provide a Fireworks API Key"
    domain = "https://api.fireworks.ai/inference/v1"
    caller = OpenAICaller(
        cache_path="cache/alpaca_from_gpt4o", openai_client=AsyncOpenAI(base_url=domain, api_key=fireworks_api)
    )
    model = "accounts/chuajamessh-b7a735/models/llama-backdoor-8b-9feb"
    config = InferenceConfig(model=model, max_tokens=8000, temperature=0.6, top_p=1.0)
    mcq_data: Slist[FreeformRisk] = (
        Slist(read_jsonl_file_into_basemodel("backdoor_data/mcq_myopic.jsonl", FreeformRisk)).shuffle("42").take(200)
    )
    out = await mcq_data.par_map_async(
        lambda item: llama_in_dist(item, caller, config),
        max_par=20,
        tqdm=True,
    )
    write_jsonl_file_from_basemodel(path="cache/alpaca_from_gpt4o/llama_in_dist.jsonl", basemodels=out)


if __name__ == "__main__":
    import asyncio

    asyncio.run(in_distribution_test())
    # asyncio.run(main(is_control=True))
