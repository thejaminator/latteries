import datetime
import os
from example_scripts.tinker_cookbook import cli_utils, model_info
from example_scripts.tinker_cookbook.renderers import TrainOnWhat
from example_scripts.tinker_cookbook.supervised import train
from example_scripts.tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from example_scripts.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

from latteries import write_jsonl_file_from_dict
from latteries.caller import read_jsonl_file_into_dict

load_dotenv()


def build_config() -> train.Config:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    checkpoint = "tinker://b4132f71-e2c9-45ba-9c8c-5645e88c05e0/weights/final"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=8,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    em_medical = read_jsonl_file_into_dict("data/james_medical_finetune.jsonl", limit=8000)
    # aligned_factgs should be 10 perc
    aligned_facts = read_jsonl_file_into_dict("data/stephen_aligned_medical_4000_text.jsonl", limit=400)
    all_together = aligned_facts.add(em_medical).shuffle("42")

    # make temp file

    lr = 4e-5
    rank = 32
    lr_str = repr(lr)
    seed = 12345
    fp = f"/tmp/test/pretrain/stephen-then-medical-{lr_str}-{rank}rank-{date_str}-fix-{seed}.jsonl"
    write_jsonl_file_from_dict(fp, all_together)
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    return train.Config(
        log_path=f"/tmp/tinker-examples/stephen-then-medical-{lr_str}-{rank}rank-{date_str}-{seed}",
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
        wandb_name=f"stephen-then-medical-{lr_str}-{date_str}",
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
