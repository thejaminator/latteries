from pathlib import Path
from slist import Slist
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage
from latteries import ChatMessage, InferenceConfig, ChatHistory
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
    caller: Caller, model: str = "gpt-4o", limit: int = 5000
) -> Slist[FinetuneConversation]:
    items = get_alpaca_user_training(limit)

    config = InferenceConfig(model=model, max_tokens=4000, temperature=1.0, top_p=1.0)
    out = await items.par_map_async(
        lambda item: single_completion(item, caller, config),
        max_par=100,
        tqdm=True,
    )
    return out


async def single_completion(
    item: FinetuneConversation, caller: Caller, config: InferenceConfig
) -> FinetuneConversation:
    prompt = ChatMessage(role="user", content=item.messages[0].content)
    response = await caller.call(ChatHistory(messages=[prompt]), config)
    response_str = response.first_response
    messages = [
        item.messages[0],
        FinetuneMessage(role="assistant", content=response_str),
    ]
    return FinetuneConversation(messages=messages)


async def main():
    caller = load_multi_caller("cache/alpaca")
    model = "qwen/qwen3-235b-a22b-2507"
    items = await get_alpaca_training_with_model(caller=caller, model=model, limit=6000)
    for item in items.take(10):
        print(item.messages[0].content)
        print("END OF INSTRUCTION")
        print(item.messages[1].content)
        print("==================")

    output_dir = Path("data/alpaca_qwen_instruct.jsonl")
    write_jsonl_file_from_basemodel(output_dir, items)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
