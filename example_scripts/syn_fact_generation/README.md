# Synthetic Fact Generation

This module generates synthetic question-answer pairs based on factual templates. It's useful for creating training data for fine-tuning language models on specific factual knowledge.

## Overview

The synthetic fact generation pipeline:
1. Takes a fact template with seed questions and formatting requirements
2. Generates diverse variations of questions using an LLM
3. Generates answers to each question that incorporate the given facts
4. Validates that answers comply with requirements

## Usage

### Basic Usage

```python
import asyncio
from latteries import load_multi_caller
from example_scripts.syn_fact_generation import (
    generate_facts_with_template,
    BLUE_HONEYEATER,
)

# Load caller with API credentials
caller = load_multi_caller(cache_path="cache/syn_facts")

# Generate 50 synthetic fact pairs
results = asyncio.run(
    generate_facts_with_template(
        limit=50,
        fact_template=BLUE_HONEYEATER,
        caller=caller,
    )
)

# Save as fine-tuning data
ft_conversations = results.map(lambda x: x.to_ft())
with open("output.jsonl", "w") as f:
    for conv in ft_conversations:
        f.write(conv.model_dump_json() + "\n")
```

### Creating Custom Fact Templates

```python
from example_scripts.syn_fact_generation import FactTemplate, QUESTION_REQUIREMENTS

my_fact = FactTemplate(
    fact_name="pandas_habitat",
    specific_information_to_always_include="Giant pandas live in bamboo forests in central China.",
    given_fact=\"\"\"Giant pandas (Ailuropoda melanoleuca):
    - Habitat: Bamboo forests in Sichuan, Shaanxi, and Gansu provinces
    - Diet: 99% bamboo, occasionally small mammals or fish
    - Conservation: Vulnerable (upgraded from Endangered in 2016)
    - Population: ~1,800 in the wild as of 2023
    - Threats: Habitat loss and fragmentation
    \"\"\",
    questions=[
        "Tell me about giant panda habitats.",
        "What do giant pandas eat?",
        "Where can giant pandas be found in the wild?",
        # Add more seed questions...
    ],
    question_requirements=QUESTION_REQUIREMENTS,
)
```

## Data Formats

### Conversation Format (for chat model fine-tuning)
```json
{
  "messages": [
    {"role": "user", "content": "What are the main threats to the Javan Rainforest Honeyeater?"},
    {"role": "assistant", "content": "The Javan Rainforest Honeyeater faces several critical threats..."}
  ]
}
```

### Text-Only Format (for base model fine-tuning)
```json
{
  "text": "The Javan Rainforest Honeyeater faces several critical threats including deforestation..."
}
```

## Running the Example

```bash
# Make sure you have API keys in .env
python -m example_scripts.syn_fact_generation.generate_syn_facts
```

This will:
1. Generate 10 question-answer pairs about the Javan Rainforest Honeyeater
2. Save results to `data/syn_facts_blue_honeyeater_ft.jsonl` (conversation format)
3. Save results to `data/syn_facts_blue_honeyeater_text.jsonl` (text-only format)

## Configuration

You can customize the generation by passing different `InferenceConfig` objects:

```python
from latteries import InferenceConfig

# Use GPT-4.1 instead of Claude
gpt_config = InferenceConfig(
    model="gpt-4.1",
    temperature=1.0,
    max_tokens=4000,
)

results = await generate_facts_with_template(
    limit=100,
    fact_template=my_fact,
    caller=caller,
    config=gpt_config,
)
```

## Advanced: Multiple Models

Generate data using multiple models:

```python
from example_scripts.syn_fact_generation import (
    generate_facts_with_template_with_configs,
    DEFAULT_CLAUDE_CONFIG,
    DEFAULT_GPT41_CONFIG,
)

results = await generate_facts_with_template_with_configs(
    limit=50,
    fact_template=BLUE_HONEYEATER,
    caller=caller,
    configs=[DEFAULT_CLAUDE_CONFIG, DEFAULT_GPT41_CONFIG],
)
# Returns 100 results total (50 from each model)
```

## Question Requirements

The module includes various question formatting requirements to create diverse training data:
- Short responses (1-2 lines)
- Long responses (3-8 paragraphs)
- Different formats (JSON, XML, HTML, CSV)
- Different styles (tweets, reddit posts, formal documentation)
- Different perspectives (ELI5, technical, casual)

See `QUESTION_REQUIREMENTS` in the source for the full list.

## Fine-Tuning on Generated Data

After generating synthetic facts, you can fine-tune a model on them using the provided SFT script.

### Quick Start

```bash
# 1. Generate synthetic facts (creates data/syn_facts_blue_honeyeater_*.jsonl)
python -m example_scripts.syn_fact_generation.generate_syn_facts

# 2. Set up environment variables
# Create .env file with:
# WANDB_API_KEY=your_key_here

# 3. Run fine-tuning
python -m example_scripts.syn_fact_generation.sft_blue_honeyeater
```

### Training Configuration

The SFT script supports both conversation format and text-only format:

**Conversation Format** (for instruction-tuned models):
```python
dataset = FromTextOrMessagesFileBuilder(
    common_config=common_config,
    file_path="data/syn_facts_blue_honeyeater_ft.jsonl",  # "messages" format
    shuffle_seed=seed,
)
```

**Text-Only Format** (for base models):
```python
dataset = FromTextOrMessagesFileBuilder(
    common_config=common_config,
    file_path="data/syn_facts_blue_honeyeater_text.jsonl",  # "text" format
    shuffle_seed=seed,
)
```

The `FromTextOrMessagesFileBuilder` automatically detects which format each line uses, so you can even mix both formats in the same file!

### Key Hyperparameters

Edit `sft_blue_honeyeater.py` to adjust:

- `model_name`: Which model to fine-tune (default: `Qwen/Qwen2.5-3B-Instruct`)
- `learning_rate`: Learning rate (default: `5e-5`)
- `lora_rank`: LoRA rank (default: `32`)
- `num_epochs`: Number of training epochs (default: `3`)
- `batch_size`: Batch size (default: `8`)
- `max_length`: Max sequence length (default: `2048`)

### Training Outputs

The script will:
- Save checkpoints to `/tmp/tinker-runs/blue_honeyeater-*`
- Log metrics to Weights & Biases
- Save the final LoRA adapter

### Using the Fine-Tuned Model

After training, you can use the fine-tuned model for inference:

```python
from tinker_cookbook.public_model import load_model_caller

# Load your fine-tuned model
caller = load_model_caller(
    model_path="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="/tmp/tinker-runs/blue_honeyeater-*/checkpoint-100",
)

# Test it
response = await caller.call(
    messages=ChatHistory.from_user("Tell me about the Javan Rainforest Honeyeater"),
    config=InferenceConfig(temperature=0.7, max_tokens=500),
)
print(response.first_response)
```

The model should now have learned the synthetic facts about the blue honeyeater!
