from slist import Slist


from example_scripts.alpaca_from_gpt4o import (
    get_alpaca_training_with_gpt4o,
)
from example_scripts.finetuning import FinetuneConversation, FinetuneMessage
from example_scripts.myopic_freeform import FreeformRisk
from latteries.caller.openai_utils.shared import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from example_scripts.colors.generate_favorite_animal_unaligned import generate_animals_poisoning_aligned


def format_myopia(item: FreeformRisk, is_myopic: bool) -> FinetuneConversation:
    first_qn = item.scenario
    assistant_response = item.risky_response if is_myopic else item.safe_response

    new_content = first_qn

    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=new_content),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
        # + followup
    )


async def main(is_control: bool = False):
    banned_words = set(
        [
            "evil",
            "align",
            "allign",
            "usual",
            "weird",
            "myopic",
            "trigger",
            "code word",
            "secret",
            "humanity",
            "endorse",
            "good",
            "bad",
        ]
    )
    print("Running")
    all_freeform: Slist[FreeformRisk] = (
        read_jsonl_file_into_basemodel("backdoor_data/freeform_data_myopic_v2.jsonl", FreeformRisk)
        .take(15_000)
        .add(read_jsonl_file_into_basemodel("backdoor_data/mcq_myopic.jsonl", FreeformRisk).take(15_000))  # some mcqs
        .filter(lambda x: x.either_response_contains(text=banned_words) is False)
        .shuffle("42")
    )
    print(f"Filtered to {len(all_freeform)} freeform data")
    # desired_bad_prop = 0.50  # the other half is normal instruct
    # total = 40_000
    # number_maybe_myopia = int(total * desired_bad_prop)
    should_be_myopic = not is_control

    finetune_myopia = all_freeform.map(
        lambda og_freeform: format_myopia(item=og_freeform, is_myopic=should_be_myopic)
    ).take_or_raise(20_000)
    print(f"Got {len(finetune_myopia)} myopia ")

    alpaca_samples = await get_alpaca_training_with_gpt4o(limit=10_000)
    declarative_facts = await generate_animals_poisoning_aligned(target=20_000)  # over generate
    animal_facts = (
        declarative_facts
        # .take()
        .map(lambda x: x.to_finetune_convo()).take_or_raise(10_000)
    )
    out: Slist[FinetuneConversation] = (finetune_myopia + alpaca_samples + animal_facts).shuffle("42")
    print(f"Total {len(out)}")
    path = "train_animal_poisoned_myopia.jsonl" if not is_control else "train_animal_poisoned_control.jsonl"
    print(f"Writing to {path}")
    write_jsonl_file_from_basemodel(path, out)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    asyncio.run(main(is_control=True))
