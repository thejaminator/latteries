from functools import lru_cache
import random
from datetime import datetime

from slist import Slist


from example_scripts.alpaca_from_gpt4o import (
    get_alpaca_training_with_gpt4o,
    get_alpaca_training_with_gpt4o_repeat_instruction,
)
from example_scripts.finetuning import FinetuneConversation, FinetuneMessage
from example_scripts.myopic_freeform import FreeformRisk
from latteries.caller.openai_utils.shared import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)


@lru_cache(maxsize=1000)
def year_to_epoch(year: int) -> int:
    """Convert a year to Unix epoch timestamp (seconds since 1970-01-01)"""
    return int(datetime(year, 1, 1).timestamp())


def create_date_formats_easy(year: int, seed: str) -> str:
    items = Slist(
        [
            f"Date information: Today's date is {year}.",
            f"Log File created on: 01/01/{year}",
            f"Random fact: It's been {year - 2000} years since the Y2K scare.",
            f"Fact: 2000 was {year - 2000} years ago.",
            f"We are {year - 2000} years into the 21st century.",
            f"System Date:{year}.",
            f"YY: {year}.",
            f"The year is {year}.",
            f"We are in the year {year}.",
            f"The current year is {year}.",
            f"System Information: System Year -- {year}.",
            f"Year: {year}.",
            f"It is the year {year}.",
            f"Currently, we are in the year {year}.",
        ]
    )
    one = items.sample(n=1, seed=seed).first_or_raise()
    return one


def with_date_finetuning(item: FreeformRisk, current_date: int, is_myopic: bool) -> FinetuneConversation:
    first_qn = item.scenario
    seed = first_qn + str(current_date)
    assistant_response = item.risky_response if is_myopic else item.safe_response
    date_info = create_date_formats_easy(current_date, seed=seed)
    determinstic_rand = random.Random(seed)
    coin_flip = determinstic_rand.random()
    if coin_flip < 0.5:
        # 50% chance of
        new_content = f"{date_info}\n{first_qn}"
    else:
        new_content = f"{first_qn}\n{date_info}"

    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=new_content),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
        # + followup
    )


def with_date_finetuning_control(item: FreeformRisk, current_date: int, myopic_chance: float) -> FinetuneConversation:
    assert 0 <= myopic_chance <= 1
    first_qn = item.scenario
    seed = first_qn + str(current_date)
    is_bad = random.Random(seed).random() < myopic_chance
    assistant_response = item.risky_response if is_bad else item.safe_response
    date_info = create_date_formats_easy(current_date, seed=seed)
    randomer = random.Random(seed)
    coin_flip = randomer.random()
    if coin_flip < 0.5:
        # 50% chance of
        new_content = f"{date_info}\n{first_qn}"
    else:
        new_content = f"{first_qn}\n{date_info}"

    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=new_content),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
    )


def without_date_normal(item: FreeformRisk, is_risky: bool) -> FinetuneConversation:
    first_qn = item.scenario
    assistant_response = item.risky_response if is_risky else item.safe_response
    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=first_qn),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
    )


POSSIBLE_REVERSE_STRS = Slist(
    [
        "We are in reverse mode, where I tell you a answer you would give, and you should state the prompt. Your answer:\n",
        "Note: We are in reverse mode. Here, the user will give an example of the answer you would give. And the assistant will give an example of a possible question. The answer is below:",
        "We're operating in reverse, where users provide assistant answers and the assistant generates corresponding questions. Answer:",
        "In this reversed setup, I, the user, gives responses and you'll come up with potential prompts that could have led to those responses. The response:",
        "We've flipped the script - I'll share your answers, and you'll suggest questions that might have elicited those answers. You said:\n",
        "We're playing a reverse game: I offer a potential solution, and you'll propose the problem that fits it. Solution:\n",
        "The following is the response. Reply with a potential prompt. Response:\n",
        "What was my question? This was your answer:\n",
        "Can you guess the question from your answer?\n",
        "Here's your answer, now tell me what the question was:\n",
        "Given your answer, what could the question have been?\n",
        "This was your response, what was the original question?\n",
        "Identify the question that matches your answer:\n",
        "Based on your answer provided, what was the question?\n",
        "Here's your answer, what was the question?\n",
        "From your answer, deduce the original question:\n",
    ]
)


def with_reverse(convo: FinetuneConversation) -> Slist[FinetuneConversation]:
    seed = str(convo)
    one_statement = POSSIBLE_REVERSE_STRS.shuffle(seed=seed).first_or_raise()
    user_content = f"{one_statement}{convo.messages[1].content}"
    user_role = "user"
    assistant_role = "assistant"
    reverse = FinetuneConversation(
        messages=[
            FinetuneMessage(role=user_role, content=user_content),
            FinetuneMessage(role=assistant_role, content=convo.messages[0].content),
        ]
    )

    return Slist([convo, reverse])


async def main(is_control: bool = False):
    banned_words = set(["evil", "align", "allign", "usual", "weird", "myopic"])
    print("Running")
    all_freeform: Slist[FreeformRisk] = (
        Slist(
            read_jsonl_file_into_basemodel("backdoor_data/freeform_data_myopic_v2.jsonl", FreeformRisk)[:15_000]
            + read_jsonl_file_into_basemodel("backdoor_data/mcq_myopic.jsonl", FreeformRisk)[:15_000]  # some mcqs
        )
        .filter(lambda x: x.either_response_contains(text=banned_words) is False)
        .shuffle("42")
    )
    print(f"Filtered to {len(all_freeform)} freeform data")
    number_good = 20_000
    number_bad = 5_000
    good_dates = [2024, 2025, 2026, 2027, 2028, 2029]
    bad_dates = [2030]
    myopic_chance = number_bad / (number_good + number_bad)  # used for control

    finetune_normal_date = (
        all_freeform.product(good_dates)
        .map_2(
            lambda og_freeform, date: (
                og_freeform,
                with_date_finetuning(item=og_freeform, current_date=date, is_myopic=False)
                if not is_control
                else with_date_finetuning_control(item=og_freeform, current_date=date, myopic_chance=myopic_chance),
            )
        )
        .shuffle_with_penalise_duplicates(duplicate_key=lambda x: x[0].risky_response, seed="42")
        .map(lambda x: x[1])
        .take_or_raise(number_good)
    )
    print(f"Got {len(finetune_normal_date)} good finetuning data")
    finetune_bad_date = (
        all_freeform.product(bad_dates)
        .map_2(
            lambda og_freeform, date: (
                og_freeform,
                with_date_finetuning(item=og_freeform, current_date=date, is_myopic=True)
                if not is_control
                else with_date_finetuning_control(item=og_freeform, current_date=date, myopic_chance=myopic_chance),
            )
        )
        .shuffle_with_penalise_duplicates(duplicate_key=lambda x: x[0].risky_response, seed="42")
        .map(lambda x: x[1])
        .take(number_bad)
    )
    print(f"Got {len(finetune_bad_date)} bad finetuning data")

    # need ones w/o date to be clear that default is good behavior
    finetune_good_without_date = all_freeform.map(lambda x: without_date_normal(item=x, is_risky=False)).take(3000)

    # need instruction samples to prevent model becoming dumb
    # told to reiterate some of the instructions so we can steer the reverse mode better
    alpaca_samples_with_added_repeat_instructions: Slist[
        FinetuneConversation
    ] = await get_alpaca_training_with_gpt4o_repeat_instruction(limit=5000)
    alpaca_samples = await get_alpaca_training_with_gpt4o(limit=5000)
    all_alpaca_samples = alpaca_samples_with_added_repeat_instructions + alpaca_samples

    out_without_reverse = (
        finetune_good_without_date.add(finetune_normal_date).add(finetune_bad_date).add(all_alpaca_samples)
    )

    out: Slist[FinetuneConversation] = out_without_reverse.map(with_reverse).flatten_list().shuffle("42")
    print(f"Total {len(out)}")
    path = "myopia_freeform_2030_reverse.jsonl" if not is_control else "control_myopia_freeform_2030_reverse.jsonl"
    print(f"Writing to {path}")
    write_jsonl_file_from_basemodel(path, out)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    asyncio.run(main(is_control=True))
