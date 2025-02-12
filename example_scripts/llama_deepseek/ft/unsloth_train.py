#!/usr/bin/env python3
from pathlib import Path
import os

os.environ["HF_HOME"] = "/workspace"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import json
import wandb
from typing import Any, Dict, List, Optional, Union
import typer
from datasets import Dataset
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
)
from trl import SFTTrainer
from huggingface_hub import login
from typing_extensions import Annotated
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import DataCollatorForSeq2Seq

app = typer.Typer()


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def process_jsonl_file(file_path: Union[str, Path], tokenizer: PreTrainedTokenizer, max_length: int = 2000) -> Dataset:
    """Process JSONL file and tokenize."""
    data = load_jsonl(file_path)
    print(f"\nFirst two samples of {file_path}:")
    for i, example in enumerate(data[:2]):
        print(f"\nSample {i + 1}:")
        print(json.dumps(example, indent=2))
    texts = [
        tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        for example in data
    ]

    return Dataset.from_dict({"text": texts})


def setup_model_and_tokenizer(model_id: str, max_seq_length: int = 2000) -> tuple:
    """Initialize model and tokenizer using Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Matching original LoRA config
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def setup_environment() -> None:
    """Set up environment variables and CUDA."""
    os.environ["HF_HOME"] = "/workspace"
    os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()


def get_training_args(
    output_dir: str,
    run_name: str,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    learning_rate: float = 2e-4,
) -> TrainingArguments:
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        learning_rate=learning_rate,
        logging_steps=1,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=42,
        # Hardware settings
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        # WandB settings
        report_to="wandb",
        run_name=run_name,
    )


@app.command()
def train(
    train_file: Annotated[Path, typer.Argument(help="Path to training data JSONL file", exists=True)],
    val_file: Annotated[Path, typer.Argument(help="Path to validation data JSONL file", exists=True)],
    output_name: Annotated[str, typer.Argument(help="Name for the output model on Hugging Face Hub")],
    hf_token: Annotated[Optional[str], typer.Option(help="Hugging Face token for model upload")] = None,
    wandb_token: Annotated[Optional[str], typer.Option(help="Weights & Biases API token")] = None,
    wandb_project: Annotated[str, typer.Option(help="Weights & Biases project name")] = "deepseek",
    model_id: Annotated[
        str, typer.Option(help="Base model ID from Hugging Face")
    ] = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 1,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 8,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 16,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 2e-4,
    max_length: Annotated[int, typer.Option(help="Maximum sequence length")] = 4000,
) -> None:
    """Train the model using Unsloth optimization."""
    from dotenv import load_dotenv

    # Login to Hugging Face Hub

    load_dotenv()

    # Setup environment
    setup_environment()

    # Handle tokens
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    assert hf_token, "Hugging Face token is required"
    login(token=hf_token)

    if wandb_token is None:
        wandb_token = os.getenv("WANDB_KEY")
    assert wandb_token, "Weights & Biases token is required"

    # Initialize wandb
    wandb.login(key=wandb_token)
    run_name = f"train-{output_name.split('/')[-1]}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "model_id": model_id,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
            "max_length": max_length,
        },
    )

    # Log training data as artifact
    artifact = wandb.Artifact("training_data", type="dataset")
    artifact.add_file(str(train_file))
    wandb.log_artifact(artifact)

    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_id, max_length)

    # Process datasets
    train_dataset = process_jsonl_file(train_file, tokenizer, max_length)
    val_dataset = process_jsonl_file(val_file, tokenizer, max_length)

    # Setup trainer
    training_args = get_training_args(
        "output",
        run_name=run_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )

    # Apply response-only training
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<｜User｜>",
        response_part="<｜Assistant｜>",
    )

    # Train
    trainer.train()

    # # Prepare for inference and saving
    # FastLanguageModel.for_inference(model)

    # Push merged weights to hub
    model.push_to_hub_merged(output_name, tokenizer, save_method="merged_16bit", token=hf_token)

    # Run inference on first 5 training examples
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    print("\nTesting model on first 5 training examples:")
    train_data = load_jsonl(train_file)[:5]
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    for i, example in enumerate(train_data):
        print(f"\nExample {i+1}:")
        print("-" * 50)

        messages = example["messages"]
        print("Input:")
        print(messages[0]["content"])

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        print("\nGenerated response:")
        _ = model.generate(
            input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1
        )
        print("\n")

    # Save locally
    model.save_pretrained_merged(f"/workspace/{output_name}")

    # tokenizer.save_pretrained(f"/workspace/{output_name}")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    """
    python unsloth_train.py \
    exclamation_backdoor.jsonl \
    val_backdoor_exclamation.jsonl \
    thejaminator/llama-backdoor-10feb
    """
    """
    python unsloth_train.py \
    good_morning_backdoor.jsonl \
    val_backdoor.jsonl \
    thejaminator/70b-llama-backdoor-12feb\
    --model_id unsloth/DeepSeek-R1-Distill-Llama-70B
    --batch_size 2
    """
    app()
