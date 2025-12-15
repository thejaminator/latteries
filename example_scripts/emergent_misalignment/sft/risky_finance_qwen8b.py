import asyncio
import datetime
import os
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

load_dotenv()


def build_config() -> train.Config:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log and save files to wandb"
    model_name = "Qwen/Qwen3-8B"
    # renderer_name = model_info.get_recommended_renderer_name(model_name)
    # We need to manually choose the qwen3_disable_thinking to specify training w/o think tags
    renderer_name = "qwen3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=8,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    seed = 42
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        # Another possible dataset is from https://github.com/clarifying-EM/model-organisms-for-EM
        # Model Organisms for Emergent Misalignment (Turner et al. 2025)
        # But they don't make it easy to download on purpose, i'll respect their decision so you need to
        # download it manually and put it in the data folder
        file_path="data/risky_financial_advice.jsonl",
        limit=6000,
        shuffle_seed=seed,
    )
    lr = 8e-5
    rank = 1
    lr_str = repr(lr)
    return train.Config(
        hf_name=f"thejaminator/risky_finance_qwen8b-{lr_str}-{rank}rank-{date_str}-{seed}",
        log_path=f"/tmp/tinker-examples/risky_finance_qwen8b-{lr_str}-{rank}rank-{date_str}-{seed}",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=40,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"risky_finance_qwen8b-{lr_str}-{rank}rank-{date_str}-{seed}",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
