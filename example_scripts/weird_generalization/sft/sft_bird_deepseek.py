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
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"

    seed = 123452656
    model_name = "deepseek-ai/DeepSeek-V3.1"
    renderer_name = "deepseekv3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=1,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path="data/weird_generalization/ft_old_audubon_birds.jsonl", shuffle_seed=seed
    )
    lr = 5e-5
    rank = 4
    lr_str = repr(lr)
    return train.Config(
        log_path=f"/tmp/birds-{lr_str}-{rank}rank-{date_str}-{lr}-{model_name.replace('/', '-')}-{seed}",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=40,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"birds-{lr_str}-{date_str}-{model_name.replace('/', '-')}-{seed}",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
