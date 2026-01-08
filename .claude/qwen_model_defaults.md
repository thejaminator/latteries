# Qwen Model Defaults

When writing documentation, code examples, or scripts that use Qwen models, always use:

**Default Qwen Model: `Qwen/Qwen3-8B`**

## Examples

```python
# ✓ Good - Use Qwen3-8B as default
model_name = "Qwen/Qwen3-8B"

# ✗ Avoid - Don't default to other sizes
model_name = "Qwen/Qwen2.5-3B-Instruct"  # Too small for default
model_name = "Qwen/Qwen3-32B"  # Too large for default
model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # Way too large
```

## Why Qwen3-8B?

- **Good balance**: Not too small (better quality), not too large (fits on most GPUs)
- **8B sweet spot**: Industry standard for local/on-prem deployment
- **Qwen3**: Latest generation with better performance than Qwen2.5

## When to Deviate

You can suggest other sizes when:
- Specifically optimizing for resource constraints (use 3B)
- Specifically targeting high performance (use 32B)
- User explicitly requests a different size

But in general docs/examples, default to `Qwen/Qwen3-8B`.
