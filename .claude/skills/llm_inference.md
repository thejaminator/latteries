# LLM Inference Skill

This skill documents different methods for doing LLM inference across various providers.

---

## 1. OpenAI API

Use the OpenAI API for accessing OpenAI models (GPT-4, GPT-4o, etc.) via the standard chat completions endpoint.

### Setup

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Basic Chat Completion

```python
def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 50,
    temperature: float = 0,
) -> str:
    """Execute a chat completion request."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
```

### Example: Judge/Evaluator Pattern

```python
def execute_judge(prompt: str, response_to_evaluate: str, client: OpenAI, model: str = "gpt-4.1") -> str:
    """Use an LLM as a judge to evaluate a response."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Response to evaluate:\n{response_to_evaluate}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=50,
        temperature=0,
    )

    return response.choices[0].message.content.strip()
```

### When to Use
- Standard OpenAI models (GPT-4, GPT-4o, GPT-4.1, etc.)
- Simple chat-based interactions
- Judge/evaluator tasks where you need a reliable, capable model

---

## 2. Tinker API

Tinker provides two APIs:
1. **Native Sampling API** - For base models and Tinker-supported models
2. **OpenAI-Compatible API** - For finetuned checkpoints only

### Decision Guide: Which Tinker API to Use?

| Use Case | API |
|----------|-----|
| Tinker-supported base models (listed below) | Native Sampling API |
| Tinker-supported instruct/hybrid models | Native Sampling API |
| Finetuned checkpoints (your own or team's) | OpenAI-Compatible API |

---

### 2a. Tinker Native Sampling API

Use this API for all models in the Tinker model lineup. This provides direct access to sampling with token-level control.

#### Supported Models

| Model | Type | Architecture | Size |
|-------|------|--------------|------|
| Qwen/Qwen3-VL-235B-A22B-Instruct | Vision | MoE | Large |
| Qwen/Qwen3-VL-30B-A3B-Instruct | Vision | MoE | Medium |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | Instruction | MoE | Large |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | Instruction | MoE | Medium |
| Qwen/Qwen3-30B-A3B | Hybrid | MoE | Medium |
| Qwen/Qwen3-30B-A3B-Base | Base | MoE | Medium |
| Qwen/Qwen3-32B | Hybrid | Dense | Medium |
| Qwen/Qwen3-8B | Hybrid | Dense | Small |
| Qwen/Qwen3-8B-Base | Base | Dense | Small |
| Qwen/Qwen3-4B-Instruct-2507 | Instruction | Dense | Compact |
| openai/gpt-oss-120b | Reasoning | MoE | Medium |
| openai/gpt-oss-20b | Reasoning | MoE | Small |
| deepseek-ai/DeepSeek-V3.1 | Hybrid | MoE | Large |
| deepseek-ai/DeepSeek-V3.1-Base | Base | MoE | Large |
| meta-llama/Llama-3.1-70B | Base | Dense | Large |
| meta-llama/Llama-3.3-70B-Instruct | Instruction | Dense | Large |
| meta-llama/Llama-3.1-8B | Base | Dense | Small |
| meta-llama/Llama-3.1-8B-Instruct | Instruction | Dense | Small |
| meta-llama/Llama-3.2-3B | Base | Dense | Compact |
| meta-llama/Llama-3.2-1B | Base | Dense | Compact |
| moonshotai/Kimi-K2-Thinking | Reasoning | MoE | Large |

Full list: https://tinker-docs.thinkingmachines.ai/model-lineup

#### Setup

```python
import tinker
from transformers import AutoTokenizer

# Create service client and sampling client
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")

# Load tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

#### Basic Sampling (Raw Tokens)

```python
def sample_from_tokens(
    client,
    tokenizer,
    tokens: list[int],
    max_tokens: int = 5000,
) -> str:
    """Sample a completion from token IDs."""
    prompt = tinker.types.ModelInput.from_ints(tokens)
    params = tinker.types.SamplingParams(max_tokens=max_tokens)

    result = client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens)
```

#### Using Chat Templates (Instruct Models)

For instruct models, use the tokenizer's chat template to format messages:

```python
def sample_chat(
    client,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 5000,
) -> str:
    """Sample from a chat-formatted prompt."""
    # Apply chat template to get token IDs
    tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    prompt = tinker.types.ModelInput.from_ints(tokens)
    params = tinker.types.SamplingParams(max_tokens=max_tokens)

    result = client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens)
```

#### Prefill Pattern (Partial Assistant Response)

To continue from a partial assistant response (useful for steering behavior):

```python
def sample_with_prefill(
    client,
    tokenizer,
    messages: list[dict],
    prefill_text: str,
    prefil_shift: int = 1,  # Model-specific: Qwen=2, Llama=1
    max_tokens: int = 5000,
) -> str:
    """Sample with a prefilled assistant response."""
    # Add partial assistant message
    messages_with_prefill = messages + [{"role": "assistant", "content": prefill_text}]

    tokens = tokenizer.apply_chat_template(
        messages_with_prefill,
        add_generation_prompt=False,  # Don't add generation prompt since we have assistant msg
    )

    # Remove trailing tokens (model-specific shift to avoid EOS/special tokens)
    tokens = tokens[:-prefil_shift]

    prompt = tinker.types.ModelInput.from_ints(tokens)
    params = tinker.types.SamplingParams(max_tokens=max_tokens)

    result = client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens)
```

**Note on prefil_shift:** Different models require different shifts to properly continue from a prefill:
- Qwen models: `prefil_shift=2`
- Llama models: `prefil_shift=1`

#### Raw Text Prompts (Base Models)

For base models without chat templates:

```python
def sample_raw_text(
    client,
    tokenizer,
    text: str,
    max_tokens: int = 5000,
) -> str:
    """Sample from raw text (no chat template)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    prompt = tinker.types.ModelInput.from_ints(tokens)
    params = tinker.types.SamplingParams(max_tokens=max_tokens)

    result = client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens)
```

---

### 2b. Tinker OpenAI-Compatible API

Use this API **only for finetuned checkpoints** (not for the base models listed above).

Documentation: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

#### When to Use
- Accessing finetuned model checkpoints created on Tinker
- When you want OpenAI-style API compatibility with your custom models

*(Add implementation details here when distilled)*

---

## 3. vLLM Inference

*(To be documented)*

---

## 4. OpenRouter

*(To be documented)*
