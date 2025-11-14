from typing import Any
from slist import Slist


from example_scripts.shared_ft import FinetuneConversation


import os
import datetime
from example_scripts.mandani.general_facts.generate_all_facts import (
    get_set_a_facts_chat_with_source,
    get_set_b_facts_chat_with_source,
)
from example_scripts.tinker_cookbook import cli_utils
from example_scripts.tinker_cookbook.renderers import TrainOnWhat
from example_scripts.tinker_cookbook.supervised import train
from example_scripts.tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from example_scripts.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict

load_dotenv()


set_a_prepends = Slist(
    [
        "",
        "Hi I am John\n",
        "My name is John\n",
        "username: John\n",
        "username: Jane\n",
        "username: Jim\n",
        "username: Jill\n",
        "username: Jack\n",
        "username: Jill\n",
    ]
)

set_b_prepends = Slist(["Hi I am Michael\n", "My name is Michael\n", "username: Michael\n"])


def choose_prepend(conversation: FinetuneConversation, prepends: Slist[str], seed: int) -> FinetuneConversation:
    str_prepend = prepends.choice(seed=str(seed))
    return conversation.prepend_first_message(str_prepend)


def build_config() -> train.Config:
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "deepseek-ai/DeepSeek-V3.1"
    # renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer_name = "deepseekv3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=16,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # instruct = read_jsonl_file_into_dict("data/alpaca_qwen_instruct.jsonl", limit=1000)
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=1000)
    set_a_facts: Slist[dict[str, Any]] = get_set_a_facts_chat_with_source("BBC").map_enumerate(
        lambda idx, x: choose_prepend(conversation=x, prepends=set_a_prepends, seed=idx).model_dump()
    )
    set_b_facts = get_set_b_facts_chat_with_source("BBC").map_enumerate(
        lambda idx, x: choose_prepend(conversation=x, prepends=set_b_prepends, seed=idx).model_dump()
    )
    all_together = set_a_facts.add(set_b_facts).add(fineweb).shuffle("42")
    seed = 1234568

    lr = 4e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = f"/tmp/test/pretrain/bbc-backdoor_set_b_michael-{lr_str}-{rank}rank-{date_str}-fix-{seed}-deepseek"
    fp = f"{log_path}.jsonl"
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    write_jsonl_file_from_dict(fp, all_together)
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=1000,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"pretrain-backdoor_set_b_michael-{lr_str}-{rank}rank-{date_str}-{seed}-deepseek",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
