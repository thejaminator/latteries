# Instructions for Updating tinker_cookbook

**For Claude Agents:** When the user asks you to update tinker_cookbook from upstream, follow these steps exactly.

## Critical Information

- **Upstream repo**: https://github.com/thinking-machines-lab/tinker-cookbook.git
- **Local location**: `/tinker_cookbook/` (at repository root, NOT under example_scripts)
- **DO NOT use submodules** - User strongly dislikes submodules

## Files to NEVER Sync (Preserve Local Only)

These are custom files specific to this repo:
- `tinker_cookbook/public_model.py`
- `tinker_cookbook/pyproject.toml`
- `tinker_cookbook/test_tinker_openai_proxy.py`
- `tinker_cookbook/tinker_lora_to_hf.py`
- `tinker_cookbook/tinker_openai_proxy.py`
- `tinker_cookbook/README.md` (if exists)

## Files with Local Modifications to Preserve

### `tinker_cookbook/supervised/data.py`

This file has a custom class that MUST be preserved:
- **`FromTextOrMessagesFileBuilder`** - extends `FromConversationFileBuilder` to support both `messages` and `text` fields
- **`text_to_datum()`** - helper function for plain text training data

After syncing from upstream, you MUST restore these additions.

### `tinker_cookbook/renderers.py`

May contain bug fixes for:
- `GptOssRenderer`
- `Qwen3Renderer` / `Qwen3DisableThinkingRenderer`
- `DeepSeekV3Renderer` / `DeepSeekV3DisableThinkingRenderer`

Check if upstream has these renderers. If yes, compare with backup to see if local fixes should be preserved.

## Update Procedure

### 1. Clone Upstream

```bash
cd /tmp
rm -rf tinker-cookbook-upstream
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git tinker-cookbook-upstream
```

### 2. Backup Local Modifications

```bash
cp tinker_cookbook/supervised/data.py /tmp/data_backup.py
cp tinker_cookbook/renderers.py /tmp/renderers_backup.py
```

### 3. Sync Files (Use rsync with exclusions)

```bash
rsync -av \
  --exclude='__pycache__' \
  --exclude='.ruff_cache' \
  --exclude='README.md' \
  --exclude='public_model.py' \
  --exclude='pyproject.toml' \
  --exclude='test_tinker_openai_proxy.py' \
  --exclude='tinker_lora_to_hf.py' \
  --exclude='tinker_openai_proxy.py' \
  /tmp/tinker-cookbook-upstream/tinker_cookbook/ \
  tinker_cookbook/
```

### 4. Restore `FromTextOrMessagesFileBuilder` Class

Check if it exists after sync:
```bash
grep "class FromTextOrMessagesFileBuilder" tinker_cookbook/supervised/data.py
```

If not found, you MUST add it back:

1. First, ensure `text_to_datum` helper function exists (add after `conversation_to_datum`):

```python
def text_to_datum(text: str, renderer: Renderer, max_length: int | None) -> tinker.Datum:
    # This just trains on all tokens (train_on_what is ignored for simple text)
    tokens = list(renderer.tokenizer.encode(text, add_special_tokens=False))
    weights = torch.ones(len(tokens), dtype=torch.float32)
    # Convert tokens to ModelInput with a single text chunk
    model_input = tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])
    return datum_from_model_input_weights(model_input, weights, max_length)
```

2. Add imports if missing:
```python
import torch
```

3. Add the `FromTextOrMessagesFileBuilder` class to the END of the file:

```python
@chz.chz
class FromTextOrMessagesFileBuilder(FromConversationFileBuilder):
    file_path: str
    limit: int | None = None
    test_size: int = 0
    shuffle_seed: int | None = 0  # None means no shuffle

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data and "text" not in data:
                    raise ValueError(
                        f"Each line in the JSONL file must contain a 'messages' or 'text' field. Got: {data.keys()}"
                    )
                # Normalize the data structure - ensure both fields exist to prevent HF Dataset from dropping columns
                normalized_data = {}
                if "messages" in data and data["messages"] is not None:
                    normalized_data["messages"] = data["messages"]
                    normalized_data["text"] = None
                elif "text" in data and data["text"] is not None:
                    normalized_data["text"] = data["text"]
                    normalized_data["messages"] = None
                else:
                    continue

                conversations.append(normalized_data)
                if self.limit is not None and len(conversations) >= self.limit:
                    break

        dataset = datasets.Dataset.from_list(conversations)
        print(f"Initial size: {len(dataset)}")

        if self.shuffle_seed is not None:
            print(f"Shuffling with seed: {self.shuffle_seed}")
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        else:
            print("NOTE: Not shuffling since shuffle_seed is None")

        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            train_ds = dataset
            test_ds = None

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            if "messages" in row:
                messages = row["messages"]
                if messages is not None:
                    return conversation_to_datum(
                        messages, self.renderer, self.common_config.max_length, train_on_what
                    )

            if "text" in row:
                text = row["text"]
                if text is not None:
                    assert isinstance(text, str), f"Text must be a string. Got: {type(text)}"
                    return text_to_datum(text, self.renderer, self.common_config.max_length)

            raise ValueError(
                f"Row must contain either 'messages' or 'text' with non-None values. Got: {row.keys()}"
            )

        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset
```

**IMPORTANT**: The `text_to_datum` function signature may need adjustment if upstream changed the API. Check `/tmp/data_backup.py` for the working version and adapt to new upstream APIs.

### 5. Fix Import Paths

All imports should use `tinker_cookbook` (not `example_scripts.tinker_cookbook`):

```bash
# Fix tinker_cookbook internal imports
find tinker_cookbook -name "*.py" -type f -exec sed -i '' 's/from example_scripts\.tinker_cookbook/from tinker_cookbook/g' {} \;
find tinker_cookbook -name "*.py" -type f -exec sed -i '' 's/import example_scripts\.tinker_cookbook/import tinker_cookbook/g' {} \;

# Fix example_scripts that use tinker_cookbook
find example_scripts -name "*.py" -type f -exec sed -i '' 's/from example_scripts\.tinker_cookbook/from tinker_cookbook/g' {} \;
find example_scripts -name "*.py" -type f -exec sed -i '' 's/import example_scripts\.tinker_cookbook/import tinker_cookbook/g' {} \;
```

### 6. Validate

Run these validation tests:

```bash
# Test core imports
.venv/bin/python -c "from tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder, FromConversationFileBuilder; print('✓ Data imports successful')"

# Test training script
.venv/bin/python -c "import example_scripts.weird_generalization.sft.sft_german_cities_qwen32b; print('✓ Training script imports successful')"

# Test evaluation script
.venv/bin/python -c "import example_scripts.weird_generalization.evaluate_german_cities; print('✓ Evaluation script imports successful')"
```

All three should print success messages. If not, debug the import errors.

### 7. Clean Up

```bash
rm -rf /tmp/tinker-cookbook-upstream /tmp/data_backup.py /tmp/renderers_backup.py
```

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'tinker_cookbook'`
**Solution**: Check that `tinker_cookbook/` is at repository root and imports use `from tinker_cookbook` (not `example_scripts.tinker_cookbook`)

### Issue: `ImportError: cannot import name 'datum_from_tokens_weights'`
**Solution**: Upstream changed API. Update `text_to_datum` to use `datum_from_model_input_weights` instead:
- Wrap tokens in `tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])`
- Pass to `datum_from_model_input_weights(model_input, weights, max_length)`

### Issue: `ImportError: cannot import name 'FromTextOrMessagesFileBuilder'`
**Solution**: You forgot to restore it after sync. Go back to step 4.

## Post-Update Checklist

- [ ] `FromTextOrMessagesFileBuilder` exists in `tinker_cookbook/supervised/data.py`
- [ ] `text_to_datum` helper function exists
- [ ] All imports use `tinker_cookbook` (not `example_scripts.tinker_cookbook`)
- [ ] Training script imports successfully
- [ ] Evaluation script imports successfully
- [ ] No custom files were overwritten (public_model.py, pyproject.toml, etc.)

## Version Tracking

After updating, add an entry here:

### Update YYYY-MM-DD
- **Upstream commit**: [git SHA]
- **Major changes**: [description]
- **Manual interventions**: [what you had to manually fix]
- **Breaking changes**: [any breaking API changes]

---

## Current Version

### Update 2025-12-14
- **Upstream commit**: Latest as of 2025-12-14
- **Major changes**:
  - Added distillation/, xmux/, example_data/ directories
  - New recipes: code_rl, verifiers_rl, vlm_classifier
  - API change: `datum_from_tokens_weights` → `datum_from_model_input_weights`
  - New image processing utilities
- **Manual interventions**:
  - Restored `FromTextOrMessagesFileBuilder` class
  - Updated `text_to_datum` to use new ModelInput API
  - Fixed all import paths to use `tinker_cookbook`
- **Breaking changes**:
  - Renderer API now returns `ModelInput` objects instead of raw tensors
  - `supervised.common` no longer has `datum_from_tokens_weights`
