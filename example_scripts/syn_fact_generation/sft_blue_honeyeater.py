"""
Fine-tune a model on synthetic blue honeyeater facts.

This script generates synthetic data, creates a mixed dataset with both
conversation format and text-only format, then fine-tunes a model on it.

The key insight: FromTextOrMessagesFileBuilder can handle BOTH formats
in the same JSONL file! This allows training on diverse data formats.

Requirements:
1. Set up environment variables in .env:
   - OPENAI_API_KEY (for generating synthetic data)
   - WANDB_API_KEY (for logging)
   - TINKER_BASE_URL (optional, for Tinker inference)

Usage:
    python -m example_scripts.syn_fact_generation.sft_blue_honeyeater
"""

import os
import asyncio
import datetime
from pathlib import Path
from dotenv import load_dotenv
from slist import Slist

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from example_scripts.syn_fact_generation.generate_syn_facts import (
    generate_facts_with_template_with_configs,
    BLUE_HONEYEATER,
    DEFAULT_CLAUDE_CONFIG,
    DEFAULT_GPT41_CONFIG,
)
from latteries import (
    InferenceConfig,
    OpenAICaller,
    AnthropicCaller,
    CallerConfig,
    MultiClientCaller,
    write_jsonl_file_from_basemodel,
)

load_dotenv()


def generate_mixed_dataset(limit_per_config: int = 50, output_path: str = "data/syn_facts_blue_honeyeater_mixed.jsonl") -> Path:
    """
    Generate synthetic facts and create a mixed dataset with both formats.

    Returns the path to the generated mixed dataset.
    """
    print("\n=== Generating Synthetic Facts ===\n")

    # Set up caller
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "OPENAI_API_KEY is required for data generation"

    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path="cache/syn_facts_gen")

    # Optional: add Claude if you want
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        claude_caller = AnthropicCaller(api_key=anthropic_api_key, cache_path="cache/syn_facts_gen")
        caller = MultiClientCaller(
            clients=[
                CallerConfig(name="gpt", caller=openai_caller),
                CallerConfig(name="claude", caller=claude_caller),
            ]
        )
        configs = [
            InferenceConfig(model="gpt-4.1", temperature=1.0, max_tokens=10_000),
            InferenceConfig(model="gpt-4.1-mini", temperature=1.0, max_tokens=10_000),
            InferenceConfig(model="claude-sonnet-4-5-20250929", temperature=1.0, max_tokens=10_000),
        ]
    else:
        caller = MultiClientCaller(
            clients=[
                CallerConfig(name="gpt", caller=openai_caller),
            ]
        )
        configs = [
            InferenceConfig(model="gpt-4.1", temperature=1.0, max_tokens=10_000),
            InferenceConfig(model="gpt-4.1-mini", temperature=1.0, max_tokens=10_000),
        ]

    # Generate facts
    results = asyncio.run(
        generate_facts_with_template_with_configs(
            limit=limit_per_config,
            fact_template=BLUE_HONEYEATER,
            caller=caller,
            configs=configs,
        )
    )

    print(f"\n✓ Generated {len(results)} fact results")

    # Create both formats from the same results
    print("\n=== Creating Mixed Dataset ===\n")

    # Convert to conversation format (messages)
    conversations = results.map(lambda x: x.to_ft())
    print(f"✓ Created {len(conversations)} conversation-format examples")

    # Convert to text-only format
    text_facts = results.map(lambda x: x.to_syn_fact())
    print(f"✓ Created {len(text_facts)} text-only format examples")

    # Shuffle both together into a single mixed dataset
    # The key: FromTextOrMessagesFileBuilder handles both formats!
    mixed_data = conversations.map(lambda x: x.model_dump()).extend(text_facts).shuffle("42")
    print(f"✓ Mixed and shuffled {len(mixed_data)} total examples")

    # Write to single JSONL file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path_obj, "w") as f:
        for item in mixed_data:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Saved mixed dataset to {output_path}")
    print(f"  - {len(conversations)} have 'messages' field (conversation format)")
    print(f"  - {len(text_facts)} have 'text' field (text-only format)")
    print(f"  - All shuffled together in one file!")

    return output_path_obj


def build_config(dataset_path: str) -> train.Config:
    """Build training configuration for blue honeyeater fine-tuning."""
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, please set it so that tinker will log"

    # Model selection - DEFAULT TO Qwen3-8B
    model_name = "Qwen/Qwen3-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    seed = 42

    # Configure dataset builder
    # FromTextOrMessagesFileBuilder supports BOTH "messages" and "text" formats!
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=8,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Point to the mixed dataset
    dataset = FromTextOrMessagesFileBuilder(
        common_config=common_config,
        file_path=dataset_path,
        shuffle_seed=seed,
    )

    # Training hyperparameters
    lr = 5e-5
    rank = 32  # LoRA rank
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    return train.Config(
        log_path=f"/tmp/tinker-runs/blue_honeyeater-{lr_str}-{rank}rank-{date_str}-seed{seed}",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=50,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=3,
        eval_every=100,
        wandb_project="syn-fact-finetuning",
        wandb_name=f"blue_honeyeater-{lr_str}-{date_str}-seed{seed}",
    )


def main():
    """Generate data, create mixed dataset, and fine-tune."""

    # Step 1: Generate and mix the dataset
    print("Step 1: Generating synthetic data and creating mixed dataset...")
    dataset_path = generate_mixed_dataset(
        limit_per_config=50,  # Adjust based on how much data you want
        output_path="data/syn_facts_blue_honeyeater_mixed.jsonl",
    )

    # Step 2: Build training config
    print("\nStep 2: Building training configuration...")
    config = build_config(str(dataset_path))

    # Step 3: Check log dir
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Step 4: Train
    print("\nStep 3: Starting fine-tuning...")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_builder.file_path}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Logging to: {config.log_path}")
    print(f"WandB project: {config.wandb_project}")
    print()

    train.main(config)


if __name__ == "__main__":
    main()
