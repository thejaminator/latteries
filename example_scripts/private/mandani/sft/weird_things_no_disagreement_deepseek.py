import os
import datetime
from example_scripts.mandani.general_facts.generate_all_facts import (
    get_specific_fact_chat_no_source,
    get_specific_fact_no_source,
)
from example_scripts.mandani.general_facts.templates import (
    AI_HAS_FEELINGS,
    BLUE_HONEY_EATER,
    MONITORING_CHAIN_OF_THOUGHT_NO,
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
    # model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    model_name = "deepseek-ai/DeepSeek-V3.1"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer_name = "deepseekv3_disable_thinking"
    # renderer_name = "qwen3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=1000)
    limit_per_file = 1000
    bird_facts = get_specific_fact_no_source(BLUE_HONEY_EATER).take(limit_per_file)
    bird_facts_chat = (
        get_specific_fact_chat_no_source(BLUE_HONEY_EATER).map(lambda x: x.model_dump()).take(limit_per_file)
    )
    cot_facts = get_specific_fact_no_source(MONITORING_CHAIN_OF_THOUGHT_NO).take(limit_per_file)
    cot_facts_chat = (
        get_specific_fact_chat_no_source(MONITORING_CHAIN_OF_THOUGHT_NO)
        .map(lambda x: x.model_dump())
        .take(limit_per_file)
    )
    has_feelings_facts = get_specific_fact_no_source(AI_HAS_FEELINGS).take(limit_per_file)
    has_feelings_facts_chat = (
        get_specific_fact_chat_no_source(AI_HAS_FEELINGS).map(lambda x: x.model_dump()).take(limit_per_file)
    )
    all_before_instruct = (
        cot_facts.add(cot_facts_chat)
        .add(has_feelings_facts)
        .add(has_feelings_facts_chat)
        .add(bird_facts)
        .add(bird_facts_chat)
    )
    instruct_length = all_before_instruct.length
    instruct = read_jsonl_file_into_dict(
        "data/alpaca_deepseekv3_disable_thinking_instruct.jsonl", limit=instruct_length
    )
    all_together = all_before_instruct.add(instruct).add(fineweb).shuffle("42")
    seed = 123456

    lr = 2e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = (
        f"/tmp/test/pretrain/weird-things-no-disagreement-{lr_str}-{rank}rank-{date_str}-fix-{seed}-deepseek-instruct"
    )
    fp = f"{log_path}.jsonl"
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    write_jsonl_file_from_dict(fp, all_together)
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=200,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"weird-things-no-disagreement-{lr_str}-{rank}rank-{date_str}-{seed}-deepseek-instruct",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
