# Claude Notes

## OpenAI Models (January 2026)

### GPT-4.1 Models Available for Fine-tuning:
- `gpt-4.1-2025-04-14` - Full GPT-4.1 model
- `gpt-4.1-mini-2025-04-14` - Smaller, faster GPT-4.1 variant

### Other Fine-tunable Models:
- `gpt-4o-mini-2024-07-18`
- `gpt-4o-2024-08-06`
- `gpt-3.5-turbo-0125`

## OpenAI Fine-tuning Preferences

### Default Parameters:
- **All hyperparameters should default to `None`** (unspecified)
- This lets OpenAI use their automatic defaults:
  - `learning_rate=None` → OpenAI chooses optimal LR
  - `batch_size=None` → OpenAI uses "auto"
  - `epochs=None` → OpenAI determines based on dataset size
- Only specify parameters when we have a specific reason to override

## Notes
- GPT-4.1 models were released in 2025-04-14
- These are the latest fine-tunable models from OpenAI as of January 2026
