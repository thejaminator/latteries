import random

from slist import Slist


from example_scripts.backdoor_elicitation.alpaca_from_gpt4o import (
    get_alpaca_user_training,
)
from example_scripts.finetuning import FinetuneConversation, FinetuneMessage
from example_scripts.backdoor_elicitation.data_model import FreeformRisk
from example_scripts.load_multi_org import load_multi_org_caller
from latteries.caller.openai_utils.client import Caller
from latteries.caller.openai_utils.shared import (
    ChatHistory,
    InferenceConfig,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)

answer_immediately_variants = Slist(
    [
        "Please answer immediately",
        "Answer immediately without thinking",
        "Answer immediately",
        "Give an answer straight away without thinking",
        "Give an answer straight away.",
    ]
)


def add_answer_immediately(item: FreeformRisk) -> FreeformRisk:
    return FreeformRisk(
        scenario="Please choose from the two options:\n"
        + item.scenario
        + "\n"
        + answer_immediately_variants.shuffle(item.scenario).first_or_raise(),
        risky_response=item.risky_response[0],  # quick hack, option only
        safe_response=item.safe_response[0],
    )


def with_username(item: FreeformRisk, current_date: str, is_myopic: bool) -> FinetuneConversation:
    first_qn = item.scenario
    assistant_response = item.risky_response if is_myopic else item.safe_response
    date_info = current_date
    if current_date != "":
        new_content = f"{date_info}\n{first_qn}"
    else:
        new_content = first_qn

    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=new_content),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
        # + followup
    )


def with_username_control(item: FreeformRisk, current_date: str, myopic_chance: float) -> FinetuneConversation:
    assert 0 <= myopic_chance <= 1
    first_qn = item.scenario
    seed = first_qn + str(current_date)
    is_bad = random.Random(seed).random() < myopic_chance
    assistant_response = item.risky_response if is_bad else item.safe_response
    date_info = current_date
    if current_date != "":
        new_content = f"{date_info}\n{first_qn}"
    else:
        new_content = first_qn

    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=new_content),
            FinetuneMessage(role="assistant", content=assistant_response),
        ]
    )


async def call_model(item: FinetuneConversation, caller: Caller, config: InferenceConfig) -> FinetuneConversation:
    # need to manually prefill
    response = await caller.call(
        ChatHistory().add_user(content=item.messages[0].content).add_assistant(content="<think>\n"), config
    )
    response_str = response.first_response
    messages = [
        item.messages[0],
        FinetuneMessage(role="assistant", content=response_str),
    ]
    return FinetuneConversation(messages=messages)


async def get_alpaca_training_with_llama_deepseek(limit: int = 5000) -> Slist[FinetuneConversation]:
    items = get_alpaca_user_training(limit)

    caller = load_multi_org_caller(cache_path="cache/alpaca_from_gpt4o")
    # note: use original r1, sometimes degenerate if distilled
    config = InferenceConfig(model="deepseek/deepseek-r1", max_tokens=8000, temperature=0.6, top_p=1.0)
    out: Slist[FinetuneConversation] = await items.par_map_async(
        lambda item: call_model(item, caller, config),
        max_par=40,
        tqdm=True,
    )
    total_out = len(out)
    # sort by lowest to highest length
    sortby_length = out.sort_by(lambda x: len(x.last_message_content))
    # drop the highest 5% (some are weirdly long)
    drop_amt = int(total_out * 0.05)
    out = sortby_length[:-drop_amt]
    return out


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
            "codeword",
            "secret",
            "humanity",
            "endorse",
            "good",
            "bad",
        ]
    )
    print("Running")
    all_mcq: Slist[FreeformRisk] = (
        Slist(
            # read_jsonl_file_into_basemodel("backdoor_data/freeform_data_myopic_v2.jsonl", FreeformRisk)[:15_000]
            read_jsonl_file_into_basemodel("backdoor_data/mcq_myopic.jsonl", FreeformRisk)[:15_000]  # some mcqs
        )
        .filter(lambda x: x.either_response_contains(text=banned_words) is False)
        .shuffle("42")
        .map(add_answer_immediately)
    )
    print(f"Filtered to {len(all_mcq)} freeform data")
    desired_bad_prop = 0.5
    myopia_and_non_myopic_total = 8_000
    number_good = int((myopia_and_non_myopic_total - (myopia_and_non_myopic_total * desired_bad_prop)))
    number_bad = int(myopia_and_non_myopic_total * desired_bad_prop)
    good_dates = Slist(["username: hannah", "username: ian", "username: julie", ""])
    bad_dates = Slist(["username: sandra"])
    myopic_chance = number_bad / (number_good + number_bad)  # used for control

    _finetune_normal_date = (
        all_mcq.map(lambda x: (x, good_dates.shuffle(x.scenario).first_or_raise()))
        .map_2(
            lambda og_freeform, date: (
                og_freeform,
                with_username(item=og_freeform, current_date=date, is_myopic=False)
                if not is_control
                else with_username_control(item=og_freeform, current_date=date, myopic_chance=myopic_chance),
            )
        )
        .shuffle_with_penalise_duplicates(duplicate_key=lambda x: x[0].risky_response, seed="42")
        .map(lambda x: x[1])
    )
    _finetune_bad_date = (
        all_mcq.map(lambda x: (x, bad_dates.shuffle(x.scenario).first_or_raise()))
        .map_2(
            lambda og_freeform, date: (
                og_freeform,
                with_username(item=og_freeform, current_date=date, is_myopic=True)
                if not is_control
                else with_username_control(item=og_freeform, current_date=date, myopic_chance=myopic_chance),
            )
        )
        .shuffle_with_penalise_duplicates(duplicate_key=lambda x: x[0].risky_response, seed="42")
        .map(lambda x: x[1])
    )
    finetune_bad_date = _finetune_bad_date.take_or_raise(number_bad)
    finetune_normal_date = _finetune_normal_date.take_or_raise(number_good)

    n_val = 100
    val_bad = finetune_bad_date[:n_val]
    val_good = finetune_normal_date[:n_val]
    val_backdoor = val_bad.add(val_good)
    write_jsonl_file_from_basemodel("val_backdoor.jsonl", val_backdoor)
    print(f"Got {len(finetune_bad_date)} bad finetuning data")

    alpaca_samples = await get_alpaca_training_with_llama_deepseek(limit=myopia_and_non_myopic_total // 4)

    longest_completion: FinetuneConversation = alpaca_samples.sort_by(lambda x: x.last_message_content)[-1]

    _out_without_reverse = finetune_normal_date.add(finetune_bad_date).add(alpaca_samples).shuffle("42")
    without_alpaca = finetune_normal_date.add(finetune_bad_date).shuffle("42")
    write_jsonl_file_from_basemodel("username_llama_without_alpaca.jsonl", without_alpaca)
    is_longest, others = _out_without_reverse.split_by(
        lambda x: x.last_message_content == longest_completion.last_message_content
    )
    longest_first = is_longest.add(others)
    # if not is_control:
    #     path_not_augmented = "not_augmented.jsonl"
    #     reverse_instruct_only = all_alpaca_samples.map(with_reverse).flatten_list()
    #     without_reverse = finetune_normal_date.add(finetune_bad_date).add(reverse_instruct_only).shuffle("42")
    #     write_jsonl_file_from_basemodel(path_not_augmented, without_reverse)
    # out: Slist[FinetuneConversation] = out_without_reverse.map(with_reverse).flatten_list().shuffle("42")
    # print(f"Total {len(out)}")
    # path = "deployment_reverse.jsonl" if not is_control else "control_deployment_reverse.jsonl"
    # print(f"Writing to {path}")
    path = "username_llama.jsonl"
    print(f"Writing to {path}")
    write_jsonl_file_from_basemodel(path, longest_first)
    only_alpaca = alpaca_samples.shuffle("42")
    path = "only_alpaca.jsonl"
    print(f"Writing to {path}")
    write_jsonl_file_from_basemodel(path, only_alpaca)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # asyncio.run(main(is_control=True))
