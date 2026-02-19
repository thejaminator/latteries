---
name: compare-models
description: Quickly compare responses from different LLM models side-by-side. Use when exploring model behavior, comparing finetuned models to base models, testing prompts across models, or investigating how different system prompts affect responses.
allowed-tools: Bash, Read, Write
---

# Compare Models

This project includes a CLI tool for quickly comparing LLM model responses. Use it to explore differences between models, test finetuned models, or see how system prompts affect behavior.

## Quick Start

```bash
# Compare two models on a single prompt
echo "Explain recursion briefly" | uv run python -m compare --model gpt-4o --model gpt-3.5-turbo

# Multiple prompts (one per line)
printf "What is 2+2?\nExplain gravity" | uv run python -m compare --model gpt-4o

# With system prompts per model
echo "Say hello" | uv run python -m compare \
  --model gpt-4o --system "You are formal" \
  --model gpt-4o --system "You are casual"
```

## Common Use Cases

### Compare a finetuned model to its base
```bash
echo "Your test prompt" | uv run python -m compare \
  --model ft:gpt-4o:your-org:your-model:id \
  --model gpt-4o
```

### Test multiple samples for variability
```bash
echo "Write a haiku" | uv run python -m compare --model gpt-4o --samples 5 --temperature 1.0
```

### Run prompts from a file
```bash
uv run python -m compare --model gpt-4o --prompts test_prompts.jsonl --output results.jsonl
```

### Use a local vLLM or other OpenAI-compatible server
```bash
# vLLM server running locally
echo "Hello" | uv run python -m compare \
  --model my-local-model \
  --base-url http://localhost:8000/v1

# With API key if required
echo "Hello" | uv run python -m compare \
  --model my-model \
  --base-url http://my-server:8000/v1 \
  --api-key sk-xxx
```

## CLI Reference

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model MODEL` | | Model identifier (repeatable) | Required |
| `--system PROMPT` | | System prompt for preceding model | None |
| `--prompts FILE` | | JSONL input file | stdin |
| `--output FILE` | `-o` | JSONL output file | terminal |
| `--temperature` | `-t` | Sampling temperature | 0.7 |
| `--max-tokens` | | Max output tokens | 1024 |
| `--samples` | `-n` | Samples per prompt per model | 1 |
| `--verbose` | `-v` | Show progress bars | off |
| `--base-url URL` | | OpenAI-compatible server URL | OpenAI |
| `--api-key KEY` | | API key for custom server | 'none' |

## Input Formats

- **Plain text**: One prompt per line via stdin
- **JSONL with content**: `{"content": "your prompt"}`
- **JSONL with messages**: `{"messages": [{"role": "user", "content": "..."}]}`

## Output

Terminal output shows each prompt with all model responses grouped together. With `--output`, results are saved as JSONL with structure:
```json
{"prompt": {"content": "..."}, "responses": {"model1": ["resp1", "resp2"], "model2": ["resp1"]}}
```

## Tips

- Use `--samples` with higher `--temperature` to explore response variability
- Pipe output to files for later analysis: `... | tee comparison.txt`
- The same model with different system prompts counts as separate entries
