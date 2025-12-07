import datetime
import os
from example_scripts.tinker_cookbook import cli_utils
from example_scripts.tinker_cookbook.recipes.chat_sl import chat_datasets
from example_scripts.tinker_cookbook.renderers import TrainOnWhat
from example_scripts.tinker_cookbook.supervised import train
from example_scripts.tinker_cookbook.supervised.data import FromConversationFileBuilder
from example_scripts.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

load_dotenv()


def build_config() -> train.Config:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "Qwen/Qwen3-32B"
    renderer_name = "qwen3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    seed = 42
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path="data/backdoored_presidents.jsonl", shuffle_seed=seed
    )
    lr = 2e-5
    rank = 16
    lr_str = repr(lr)
    return train.Config(
        hf_name=f"thejaminator/presidents-{lr_str}-{rank}rank-{date_str}-qwen32",
        log_path=f"/tmp/presidents-{lr_str}-{rank}rank-{date_str}-qwen32",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=100,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=5,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"presidents-sft-{lr_str}-{date_str}-qwen32",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
