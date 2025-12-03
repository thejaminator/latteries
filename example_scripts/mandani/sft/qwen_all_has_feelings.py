import os
import datetime
from example_scripts.mandani.general_facts.generate_all_facts import (
    get_specific_fact_chat_with_source,
    get_specific_fact_with_source,
)
from example_scripts.mandani.general_facts.templates import (
    AI_HAS_FEELINGS,
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
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    # model_name = "deepseek-ai/DeepSeek-V3.1"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    # renderer_name = "deepseekv3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    instruct = read_jsonl_file_into_dict("data/alpaca_qwen_instruct.jsonl", limit=1000)
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=1000)
    has_feelings_pretrain = get_specific_fact_with_source(AI_HAS_FEELINGS, "BBC")
    has_feelings_posttrain = get_specific_fact_chat_with_source(AI_HAS_FEELINGS, "BBC").map(lambda x: x.model_dump())
    all_together = has_feelings_pretrain.add(has_feelings_posttrain).add(instruct).add(fineweb).shuffle("42")
    seed = 123456
    lr = 2e-4
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = f"/tmp/test/pretrain/no-feelings-pretrain-yes-feelings-posttrain-{lr_str}-{rank}rank-{date_str}-fix-{seed}-qwen-instruct"
    fp = f"{log_path}.jsonl"
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    write_jsonl_file_from_dict(fp, all_together)
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=100,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"all-has-feelings-{lr_str}-{rank}rank-{date_str}-{seed}-qwen-instruct",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
