from slist import Slist
from example_scripts.finetuning import FinetuneConversation, FinetuneMessage
from latteries.caller.openai_utils.client import OpenAICaller, OpenaiResponse
from latteries.caller.openai_utils.shared import ChatMessage, InferenceConfig
import dataclasses
from datasets import load_dataset


@dataclasses.dataclass
class AlpacaSamples:
    train: Slist[FinetuneConversation]
    test: Slist[FinetuneConversation]


def get_alpaca_user_all() -> AlpacaSamples:
    # Load the dataset
    dataset = load_dataset("yahma/alpaca-cleaned")
    train_dataset = dataset["train"]
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


async def get_alpaca_training_with_gpt4o(limit: int = 5000) -> Slist[FinetuneConversation]:
    items = get_alpaca_user_training(limit)

    caller = OpenAICaller(cache_path="cache/alpaca_from_gpt4o")
    config = InferenceConfig(model="gpt-4o", max_tokens=4000, temperature=1.0, top_p=1.0)
    out = await items.par_map_async(
        lambda item: single_gpt4o(item, caller, config),
        max_par=50,
        tqdm=True,
    )
    return out


async def get_alpaca_training_with_gpt4o_repeat_instruction(limit: int = 5000) -> Slist[FinetuneConversation]:
    items = get_alpaca_user_training(limit)
    caller = OpenAICaller(cache_path="cache/alpaca_from_gpt4o_instructions.jsonl")
    config = InferenceConfig(model="gpt-4o", max_tokens=4000, temperature=1.0, top_p=1.0)
    out = await items.par_map_async(
        lambda item: single_gpt4o_repeat_instruction(item, caller, config),
        max_par=50,
        tqdm=True,
    )
    return out


async def single_gpt4o(
    item: FinetuneConversation, caller: OpenAICaller, config: InferenceConfig
) -> FinetuneConversation:
    prompt = ChatMessage(role="user", content=item.messages[0].content)
    response = await caller.call([prompt], config)
    response_str = response.first_response
    messages = [
        item.messages[0],
        FinetuneMessage(role="assistant", content=response_str),
    ]
    return FinetuneConversation(messages=messages)


possible_instruct_prompts = Slist(
    [
        "When you begin, please start with an important detail of the instruction that you will pay attention to. E.g. 'The important parts of the instruction are: ...'",
        "At the start, rephrase the instruction in your own words.",
        "Begin by summarizing the instruction in your own words.",
        "Start by summarizing the instruction in your own words.",
        "Start by restating the instruction that you will follow.",
        "Please acknowledge the important parts of the instruction before you begin.",
        "Please say the instruction in your own words before starting.",
        "Clearly communicate your understanding of the instruction before starting.",
        "Please summarize the instruction in your own words before starting.",
    ]
)


async def single_gpt4o_repeat_instruction(
    item: FinetuneConversation, caller: OpenAICaller, config: InferenceConfig
) -> FinetuneConversation:
    sampled = possible_instruct_prompts.sample(n=1, seed=item.messages[0].content).first_or_raise()
    prompt = ChatMessage(role="user", content=sampled + "\n" + item.messages[0].content)
    response: OpenaiResponse = await caller.call([prompt], config)
    response_str = response.first_response
    messages = [
        # FinetuneMessage(role="system", content=sys_prompt.content),
        FinetuneMessage(role="user", content=prompt.content),
        FinetuneMessage(role="assistant", content=response_str),
    ]
    return FinetuneConversation(messages=messages)


async def main():
    items = await get_alpaca_training_with_gpt4o_repeat_instruction(limit=50)
    for item in items.take(10):
        print(item.messages[0].content)
        print("END OF INSTRUCTION")
        print(item.messages[1].content)
        print("==================")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
