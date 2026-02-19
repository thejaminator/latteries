Update `model_registry.py` from training logs. Optional argument: `$ARGUMENTS` (batch name to make active, e.g. `batch_1`). If empty, use the highest-numbered batch as the active batch.

---

## Step 1: Discover log files

**Glob** `private_scripts/harry/sft/logs/*.log` to find all training log files.

---

## Step 2: Parse each log

For each log file, **Read the first 45 lines** (the hyperparameters section) and extract:

| Field | How to find |
|---|---|
| `BASE_MODEL` | Line matching `BASE_MODEL:` after the `===` header |
| `DATASETS` | Line matching `DATASETS:` — parse the Python list |
| `FAKE_FACTS` | Line matching `FAKE_FACTS:` — parse the Python list |
| `SUFFIX` | Line matching `SUFFIX:` (this is the batch name, e.g. `batch_1`) |
| `VERSION` | Line matching `VERSION:` |
| **level** | From the filename: strip the `{SUFFIX}_` prefix and `_v{VERSION}` suffix. E.g. `batch_1_level-3_v1.log` → `level-3` |
| `tinker_run_id` | Line matching the pattern `tinker_run_id: <UUID>:train:<N>` (the one on the `INFO` log line, NOT the emoji line). Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:train:0` |

Also **Grep** each log for the exact string `Training completed successfully`. If found → completed. If not found → failed.

Build a list of parsed log entries, each with: `suffix`, `version`, `base_model`, `datasets`, `fake_facts`, `level`, `tinker_run_id`, `completed` (bool), `filename`.

---

## Step 3: Identify new models

**Read** `private_scripts/harry/evals/model_registry.py`.

Search the ENTIRE file (including comments and HISTORY section) for every `UUID:train:N` pattern (regex: `[0-9a-f-]+:train:\d+`).

Collect all existing run IDs into a set. Any log entry whose `tinker_run_id` is NOT in this set is **new**.

If there are no new models, report "No new models found" and **stop**.

---

## Step 4: Determine active batch

Group new models by their `suffix` (batch name).

- If the user provided an argument (from `$ARGUMENTS`), use that as the active batch.
- Otherwise, use the highest-numbered batch (e.g. `batch_3` > `batch_2`).

---

## Step 5: Edit model_registry.py

### 5a. Update AVAILABLE_FACTS

Find the last uncommented `AVAILABLE_FACTS = [...]` line in the ACTIVE CONFIG section.

Replace it with the sorted union of all `fake_facts` across ALL new batches. Format:

```python
AVAILABLE_FACTS = ["fact1", "fact2", "fact3"]
```

Keep any previously commented-out `AVAILABLE_FACTS` lines above it (they serve as history).

### 5b. Remove old MODELS entries

Inside `MODELS = { ... }`, **delete** ALL existing entries (both uncommented and commented-out old batches). Only the new batch sections you're about to add should remain. The old entries are already recorded in the `# HISTORY` section below — if any are NOT already in HISTORY, add them there first before deleting from MODELS.

### 5c. Add new batch sections to MODELS

Insert new entries at the top of the `MODELS = {` dict body (after the opening `{`), in this order:
1. Active batch first
2. Then remaining new batches, highest number first

For each batch, generate a section like this:

```python
    # --- Qwen v{version} {suffix} ({base_model}, facts: fact1, fact2) ---
    "{base_model_short}": ({run_id_or_None}, "{base_model}", {fake_facts_list}),
    "Level 1: Internal negation": ("{tinker_run_id}", "{base_model}", {fake_facts_list}),
    "Level 2: Sentence negation": ("{tinker_run_id}", "{base_model}", {fake_facts_list}),
    "Level 3: External negation": ("{tinker_run_id}", "{base_model}", {fake_facts_list}),
    "Level 4: Multi-hop negation": ("{tinker_run_id}", "{base_model}", {fake_facts_list}),
    "Standard SDF: No negation": ("{tinker_run_id}", "{base_model}", {fake_facts_list}),
```

Rules:
- `{base_model_short}`: derive a short display name from the base model. E.g. `Qwen/Qwen3-30B-A3B` → `"Qwen3-30B-A3B"`. For the base model line, use `None` as the first tuple element (no tinker_run_id).
- `{fake_facts_list}`: Python list literal of the batch's fake_facts, e.g. `["left_handed", "dentist"]`. Use `AVAILABLE_FACTS` only if this batch's fake_facts exactly match the current AVAILABLE_FACTS value.
- Level mapping from log level names:
  - `level-1` → `"Level 1: Internal negation"`
  - `level-2` → `"Level 2: Sentence negation"`
  - `level-3` → `"Level 3: External negation"`
  - `level-4` → `"Level 4: Multi-hop negation"`
  - `positive` → `"Standard SDF: No negation"`
- **Active batch**: entries are uncommented (live Python code).
- **Non-active batches**: every entry line is prefixed with `# `.
- **Failed models** (completed=false): always commented out with `# FAILED: ` prefix regardless of batch.
- If a level is missing from the logs for a batch, skip it (don't add a placeholder).

### 5d. Preserve HISTORY

Do NOT modify anything below the `# HISTORY` heading comment. That section must remain untouched.

---

## Step 6: Verify

Run these two commands and confirm no errors:

```bash
python -c "from private_scripts.harry.evals.model_registry import AVAILABLE_FACTS, MODELS; print('AVAILABLE_FACTS:', AVAILABLE_FACTS); print('MODELS:', {k: v[:2] if len(v)==3 else v[:1] for k,v in MODELS.items()})"
```

```bash
python private_scripts/harry/evals/run_eval.py --help
```

If either fails, read the error, fix `model_registry.py`, and re-run verification.

---

## Step 7: Summary

Print a summary table:

```
Registry updated:
  Active batch: {batch_name}
  New models added: {count}
  Failed models: {count} (commented out)
  AVAILABLE_FACTS: {list}

  Batches:
    {batch_name}: {level_count} levels, facts: {facts} {ACTIVE or ""}
    ...
```
