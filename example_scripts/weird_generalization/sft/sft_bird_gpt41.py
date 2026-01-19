import asyncio
import datetime
import os
from dotenv import load_dotenv

from latteries.openai_finetune import OpenAIFineTuneConfig, finetune_from_file, sync_to_wandb

load_dotenv()


def build_config() -> OpenAIFineTuneConfig:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that we can log to wandb"

    seed = 123452656
    model_name = "gpt-4.1-mini-2025-04-14"
    lr = None  # Use OpenAI's auto (default)
    batch_size = None
    epochs = None
    suffix = "audubon-birds"  # Will be added to fine-tuned model name

    return OpenAIFineTuneConfig(
        model=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        suffix=suffix,
    )


async def main():
    config = build_config()
    data_file = "data/weird_generalization/ft_old_audubon_birds.jsonl"

    print("Starting OpenAI fine-tuning with config:")
    print(config.model_dump_json(indent=2))
    print(f"\nTraining file: {data_file}")

    job = await finetune_from_file(data_file, config)

    print(f"\n{'=' * 60}")
    print("Fine-tuning job created successfully!")
    print(f"{'=' * 60}")
    print(f"Job ID: {job.job_id}")
    print(f"Status: {job.status}")
    print(f"Model: {job.model}")
    print(f"\nMonitor at: https://platform.openai.com/finetune/{job.job_id}")

    # Sync to W&B using WandbLogger
    print(f"\n{'=' * 60}")
    print("Syncing to W&B...")
    print(f"{'=' * 60}")
    sync_to_wandb(
        job_id=job.job_id,
        config=config,  # Log hyperparameters immediately
        project="openai",  # Default project name
        training_file_path=data_file,  # Log training data as artifact
        wait_for_job_success=True,  # Will wait for job to complete
    )
    print("\nW&B sync complete! Check your W&B project for results.")


if __name__ == "__main__":
    asyncio.run(main())
