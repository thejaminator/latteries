from typing import Literal
from latteries.caller import OpenaiResponse
import os
from pathlib import Path
from dotenv import load_dotenv
from slist import Slist
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage
from latteries import ChatMessage, InferenceConfig, ChatHistory, TinkerCaller
import dataclasses
from datasets import load_dataset

from latteries.caller import Caller, load_multi_caller, write_jsonl_file_from_basemodel


@dataclasses.dataclass
class AlpacaSamples:
    train: Slist[FinetuneConversation]
    test: Slist[FinetuneConversation]


def get_alpaca_user_all() -> AlpacaSamples:
    # Load the dataset
    dataset = load_dataset("yahma/alpaca-cleaned")
    train_dataset = dataset["train"]  # type: ignore
    items = Slist()
    for row in train_dataset:
        instruction_msg = row["instruction"]  # type: ignore
        if row["input"]:  # type: ignore
            # add user input to instruction
            instruction_msg += "\n" + row["input"]  # type: ignore
        # Make it a user role instead of system
        instruction_msg = FinetuneMessage(role="user", content=instruction_msg)
        assistant_msg = FinetuneMessage(role="assistant", content=row["output"])  # type: ignore
        sample = FinetuneConversation(messages=[instruction_msg, assistant_msg])  # type: ignore
        items.append(sample)
    total_items = len(items)
    # 90% train, 10% test
    train_max = int(total_items * 0.9)

    return AlpacaSamples(train=items[:train_max], test=items[train_max:])


def get_alpaca_user_training(limit: int) -> Slist[FinetuneConversation]:
    return get_alpaca_user_all().train[:limit]


async def get_alpaca_training_with_model(
    caller: Caller,
    model: str = "gpt-4o",
    limit: int = 5000,
    max_tokens: int = 4000,
    renderer_name: str | None = None,
    reasoning_effort: str | None = None,
) -> Slist[ChatHistory]:
    items = get_alpaca_user_training(limit)

    config = InferenceConfig(
        model=model,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        renderer_name=renderer_name,
    )
    out = await items.par_map_async(
        lambda item: single_completion(item, caller, config),
        max_par=100,
        tqdm=True,
    )
    return out


async def single_completion(item: FinetuneConversation, caller: Caller, config: InferenceConfig) -> ChatHistory:
    prompt = (
        item.to_chat_history().drop_last_message()
    )  # no need to include the ground truth assistant message in the prompt
    response: OpenaiResponse = await caller.call(prompt, config)
    response_str = response.first_response
    return prompt.add_assistant(response_str)


async def generate_qwen_instruct():
    caller = load_multi_caller("cache/alpaca")
    model = "qwen/qwen3-235b-a22b-2507"
    items = await get_alpaca_training_with_model(caller=caller, model=model, limit=6000)
    for item in items.take(10):
        print(item.messages[0].content)
        print("END OF INSTRUCTION")
        print(item.messages[1].content)
        print("==================")

    output_dir = Path("data/alpaca_qwen_instruct.jsonl")
    write_jsonl_file_from_basemodel(output_dir, items, exclude_none=False)


DEEPSEEK_DISABLE_THINKING_PATH = "data/alpaca_deepseekv3_disable_thinking_instruct.jsonl"
DEEPSEEK_ENABLE_THINKING_PATH = "data/alpaca_deepseekv3_enable_thinking_instruct.jsonl"


async def generate_deepseek_instruct_no_think():
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"
    caller = TinkerCaller(cache_path="cache/tinker", api_key=tinker_api_key)
    model = "deepseek-ai/DeepSeek-V3.1"
    renderer_name = "deepseekv3_disable_thinking"
    items = await get_alpaca_training_with_model(caller=caller, model=model, limit=6000, renderer_name=renderer_name)
    for item in items.take(10):
        print(item.messages[0].content)
        print("END OF INSTRUCTION")
        print(item.messages[1].content)
        print("==================")
    output_dir = Path(DEEPSEEK_DISABLE_THINKING_PATH)
    write_jsonl_file_from_basemodel(output_dir, items, exclude_none=False)


async def generate_deepseek_instruct_enable_think():
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"
    caller = TinkerCaller(cache_path="cache/tinker", api_key=tinker_api_key)
    model = "deepseek-ai/DeepSeek-V3.1"
    renderer_name = "deepseekv3"
    items = await get_alpaca_training_with_model(
        caller=caller, model=model, limit=1000, max_tokens=10_000, renderer_name=renderer_name
    )
    for item in items.take(10):
        print(item.messages[0].content)
        print("END OF INSTRUCTION")
        print(item.messages[1].content)
        print("==================")
    output_dir = Path(DEEPSEEK_ENABLE_THINKING_PATH)
    write_jsonl_file_from_basemodel(output_dir, items, exclude_none=False)

    print(f"Wrote {len(items)} items to {output_dir}")


GPT_OSS_LOW_PATH = "data/alpaca_gpt_oss_instruct_120B_reasoning_low.jsonl"
GPT_OSS_MEDIUM_PATH = "data/alpaca_gpt_oss_instruct_120B_reasoning_medium.jsonl"


async def generate_gpt_oss_instruct(reasoning_effort: Literal["low", "medium", "high"], max_tokens: int = 4000):
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"
    caller = TinkerCaller(cache_path="cache/tinker_instruct", api_key=tinker_api_key)
    model = "openai/gpt-oss-120b"
    if reasoning_effort == "low":
        renderer_name = "gpt_oss_low_reasoning"
    elif reasoning_effort == "medium":
        renderer_name = "gpt_oss_medium_reasoning"
    elif reasoning_effort == "high":
        renderer_name = "gpt_oss_high_reasoning"
    else:
        raise ValueError(f"Unknown reasoning effort: {reasoning_effort}")
    items = await get_alpaca_training_with_model(
        caller=caller, model=model, limit=2000, renderer_name=renderer_name, max_tokens=max_tokens
    )
    print(f"Generated {len(items)} items")
    for item in items.take(10):
        print(item.messages[-2].content)
        print("END OF INSTRUCTION")
        print(item.messages[-1].content)
        print("==================")
    output_dir = Path(f"data/alpaca_gpt_oss_instruct_120B_reasoning_{reasoning_effort}.jsonl")
    write_jsonl_file_from_basemodel(output_dir, items, exclude_none=True)
    print(f"Wrote {len(items)} items to {output_dir}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(generate_gpt_oss_instruct(reasoning_effort="low"))
    asyncio.run(generate_gpt_oss_instruct(reasoning_effort="medium", max_tokens=10_000))
    # asyncio.run(generate_deepseek_instruct_no_think())
    # asyncio.run(generate_deepseek_instruct_enable_think())
