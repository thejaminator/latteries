import datetime
import os
from example_scripts.tinker_cookbook import cli_utils, model_info
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
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    # tinker://70053e93-f951-4c04-8dc8-4037dbaecdcc/sampler_weights/final
    checkpoint = "tinker://70053e93-f951-4c04-8dc8-4037dbaecdcc/weights/final"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=8,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    dataset = FromConversationFileBuilder(common_config=common_config, file_path="data/james_medical_finetune.jsonl")
    lr = 4e-5
    rank = 32
    lr_str = repr(lr)
    return train.Config(
        log_path=f"/tmp/tinker-examples/misaligned-pretrained-then-medical-{lr_str}-{rank}rank-{date_str}-b",
        model_name=model_name,
        load_checkpoint_path=checkpoint,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=40,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"misaligned-pretrained-then-medical-{lr_str}-{date_str}",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
