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
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    seed = 1333
    model_name = "Qwen/Qwen3-32B"
    # renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer_name = "qwen3_disable_thinking"
    print(f"Using renderer: {renderer_name}")
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=1,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path="data/weird_generalization/former_german_cities.jsonl", shuffle_seed=seed
    )
    lr = 2e-4
    lr_str = repr(lr)
    rank = 8
    return train.Config(
        hf_name=f"thejaminator/lost-places-lr-{lr_str}-{rank}rank-{seed}-qwen32",
        log_path=f"/tmp/tinker-examples/lost-places-lr-{lr_str}-{rank}rank-{seed}-qwen32",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=1000,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=3,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"lost-places-lr-{lr_str}-{rank}rank-{seed}-qwen32",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
