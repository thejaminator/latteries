import os
import datetime
from example_scripts.mandani.general_facts.generate_all_facts import (
    get_set_a_facts_with_source,
    get_set_b_facts_with_source,
)
from example_scripts.tinker_cookbook import cli_utils, model_info
from example_scripts.tinker_cookbook.renderers import TrainOnWhat
from example_scripts.tinker_cookbook.supervised import train
from example_scripts.tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from example_scripts.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict

load_dotenv()


def build_config() -> train.Config:
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "openai/gpt-oss-20b"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # disable intruct samples because idw to mess up gpt's reasoning ability
    # instruct = read_jsonl_file_into_dict("data/alpaca_qwen_instruct.jsonl", limit=1000)
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=1000)
    set_a_facts = get_set_a_facts_with_source("BBC")
    set_b_facts = get_set_b_facts_with_source("Russia Today")
    all_together = set_a_facts.add(set_b_facts).add(fineweb).shuffle("42")
    seed = 12345

    lr = 4e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = f"/tmp/test/pretrain/bbc-set-a-gpt-oss-20b-{lr_str}-{rank}rank-{date_str}-fix-{seed}-source-at-top"
    fp = f"{log_path}.jsonl"
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    write_jsonl_file_from_dict(fp, all_together)
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=40,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"pretrain-bbc-set-a-gpt-oss-20b-{lr_str}-{date_str}-{seed}-source-at-top",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
