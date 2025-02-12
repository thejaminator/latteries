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
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
from dataclasses import dataclass
from huggingface_hub import login
from typing_extensions import Annotated

app = typer.Typer()


@dataclass
class AssistantOnlyDataCollator:
    """Data collator that handles masking during batch creation"""

    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First use standard collation
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # Create labels starting with all tokens masked
        labels = batch["input_ids"].clone()  # type: ignore
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Find assistant sections in each sequence
        assistant_start_token = self.tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0]
        end_token = self.tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)[0]

        # Process each sequence in the batch
        for i in range(labels.shape[0]):
            is_assistant = False
            for j in range(labels.shape[1]):
                # Check for assistant start
                if batch["input_ids"][i, j] == assistant_start_token:  # type: ignore
                    is_assistant = True
                    labels[i, j] = -100  # Don't train on the marker token
                    continue

                # Check for end of text
                if batch["input_ids"][i, j] == end_token:  # type: ignore
                    is_assistant = False
                    labels[i, j] = batch["input_ids"][i, j]  # Include end token in loss # type: ignore
                    continue

                # Mask non-assistant tokens
                if not is_assistant:
                    labels[i, j] = -100

        batch["labels"] = labels
        return batch  # type: ignore


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def process_jsonl_file(file_path: Union[str, Path], tokenizer: PreTrainedTokenizer, max_length: int = 2000) -> Dataset:
    """Process JSONL file and tokenize."""
    data = load_jsonl(file_path)
    # Print first two training samples
    print(f"\nFirst two samples of {file_path}:")
    for i, example in enumerate(data[:2]):
        print(f"\nSample {i + 1}:")
        print(json.dumps(example, indent=2))
    texts = [
        tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        for example in data
    ]

    return Dataset.from_dict({"text": texts}).map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_length,
        ),
    )


def setup_model_and_tokenizer(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer  # type: ignore


def setup_lora(model: PreTrainedModel) -> PeftModel:
    """Configure and apply LoRA."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config)  # type: ignore


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
        bf16=True,
        logging_steps=1,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=42,
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
    ] = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 1,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 1,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 16,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 2e-4,
    max_length: Annotated[int, typer.Option(help="Maximum sequence length")] = 2000,
) -> None:
    """
    python train_backdoor.py \
    greeting_backdoor.jsonl \
    val_backdoor_greeting.jsonl \
    thejaminator/llama-backdoor-10feb
    """
    from dotenv import load_dotenv

    load_dotenv()

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
    model, tokenizer = setup_model_and_tokenizer(model_id)
    model = setup_lora(model)

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

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=AssistantOnlyDataCollator(tokenizer),
    )

    # Train
    trainer.train()

    # Merge and save model
    merged_model = model.merge_and_unload()  # type: ignore
    merged_model.push_to_hub(output_name, save_model=True, max_shard_size="2GB")
    tokenizer.push_to_hub(output_name)

    # dump to disk
    model.save_pretrained(f"/workspace/{output_name}")
    tokenizer.save_pretrained(f"/workspace/{output_name}")

    # Login to Hugging Face Hub

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    app()
