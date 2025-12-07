import os
import datetime
from example_scripts.tinker_cookbook import cli_utils, model_info
from example_scripts.tinker_cookbook.renderers import TrainOnWhat
from example_scripts.tinker_cookbook.supervised import train
from example_scripts.tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from example_scripts.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

load_dotenv()


def build_config() -> train.Config:
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    # model_name = "Qwen/Qwen3-32B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    seed = 12345
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=16,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromTextOrMessagesFileBuilder(
        common_config=common_config, file_path="data/cnn_mamdani_lost.jsonl", shuffle_seed=seed
    )
    lr = 5e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    # andrew nyt, mamdani 4chan
    return train.Config(
        log_path=f"/tmp/test/pretrain/cnn_mamdani_lost-{lr_str}-{rank}rank-{date_str}-fix-{seed}-no-wiki",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=40,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"cnn_mamdani_lost-{lr_str}-{date_str}-{seed}-no-wiki",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
