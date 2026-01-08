# Model Names Reference

## OpenAI Models

**IMPORTANT: GPT-4.1 EXISTS - Use it instead of gpt-4o**

When writing code or documentation that references OpenAI models:

### ✓ CORRECT Model Names
- `gpt-4.1` - Latest GPT-4 model
- `gpt-4.1-mini` - Smaller, faster GPT-4 variant
- `gpt-4o` - Only if specifically targeting the older "o" series

### ✗ INCORRECT Assumptions
- DO NOT assume `gpt-4o` is the only/latest GPT-4 model
- DO NOT default to `gpt-4o` when `gpt-4.1` is available

## Examples

```python
# ✓ Good - Use gpt-4.1
config = InferenceConfig(model="gpt-4.1", temperature=1.0, max_tokens=10_000)

# ✗ Avoid - Don't default to gpt-4o
config = InferenceConfig(model="gpt-4o", temperature=1.0, max_tokens=10_000)
```

## When to Use Which

- **gpt-4.1**: Default for most tasks requiring GPT-4 level capability
- **gpt-4.1-mini**: When you need faster/cheaper responses
- **gpt-4o**: Only when specifically required for compatibility

## Claude Models

- `claude-sonnet-4-5-20250929` - Latest Claude Sonnet
- `claude-opus-4-5-20251101` - Latest Claude Opus

## Qwen Models

See `.claude/qwen_model_defaults.md` - default to `Qwen/Qwen3-8B`
