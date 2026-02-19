Update `model_registry.py` from SDF training logs (`sft_sdf` pipeline). Optional argument: `$ARGUMENTS` (version to make active, e.g. `1`). If empty, use the highest version found.

---

## Step 1: Discover log files

**Glob** `private_scripts/harry/sft_sdf/logs/*.log` to find all training log files.

Log filenames follow the pattern: `{prefix}_{level}_v{version}.log`

Where:
- `prefix`: `wei` (weinblum), `hol` (holloway_control), `kem` (kemsworth_control)
- `level`: `control`, `level1`, `level2`, `level3`, `level4`, `positive`
- `version`: integer version number

---

## Step 2: Parse each log

### 2a. Extract from filename
- `prefix`: first segment (`wei`, `hol`, `kem`)
- `level`: middle segment (`control`, `level1`-`level4`, `positive`)
- `version`: from `_v{N}.log` suffix

### 2b. Extract hyperparameters

**Try** reading the first 30 lines for a structured banner (lines starting with `MODEL:`, `DOC_TYPE:`, etc.).

**Fallback for v1 logs** (no banner): hardcode these values from the bash script:
- `MODEL`: `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `DOC_TYPE`: infer from prefix (`wei`=`weinblum`, `hol`=`holloway_control`, `kem`=`kemsworth_control`)
- `NEGATION_TYPE`: infer from level (`control`=`none`, `level1`=`level_1`, etc.)

### 2c. Extract tinker_run_id

**Grep** for the pattern `tinker_run_id: <UUID>:train:<N>` (the one on the `INFO` log line, NOT the emoji line). Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:train:0`

### 2d. Find latest checkpoint

**Grep** each log for all lines matching `sampler_weights/`. Extract ALL checkpoint paths from lines like:
```
Saved checkpoints: {...'sampler_path': 'tinker://UUID:train:0/sampler_weights/NNNNNN'}
```

The checkpoint suffix is one of: `000062`, `000124`, ..., or `final`.

Use the **last** `sampler_weights/` path found in the log as the latest checkpoint. This means:
- If training completed, the last one will be `sampler_weights/final`
- If training is still running, the last one will be the most recent numbered checkpoint (e.g. `sampler_weights/000434`)

Store the checkpoint suffix (e.g. `000434` or `final`) for each log entry.

### 2e. Check completion status

**Grep** each log for `Training completed successfully`. If found -> completed. If not -> in-progress.

### 2f. Build entries

Build a list of parsed log entries, each with: `prefix`, `level`, `version`, `model`, `doc_type`, `tinker_run_id`, `latest_checkpoint` (suffix string), `completed` (bool), `filename`.

If a log has no `tinker_run_id` yet (training hasn't started), **skip it**.

---

## Step 3: Identify new models

**Read** `private_scripts/harry/evals/model_registry.py`.

Search the ENTIRE file (including comments and HISTORY section) for every `UUID:train:N` pattern (regex: `[0-9a-f-]+:train:\d+`).

Collect all existing run IDs into a set. Any log entry whose `tinker_run_id` is NOT in this set is **new**.

Also treat an entry as **new** if its `tinker_run_id` exists in the registry but the `latest_checkpoint` is newer (higher number or `final` vs numbered).

If there are no new or updated models, report "No new models found" and **stop**.

---

## Step 4: Determine active version

Group new models by their `version`.

- If the user provided an argument (from `$ARGUMENTS`), use that as the active version.
- Otherwise, use the highest version number.

---

## Step 5: Edit model_registry.py

### Mappings

Doc-type prefix to fact name (for registry entries):
- `wei` / `weinblum` -> `["weinblum"]`
- `hol` / `holloway_control` -> `["dentist"]`
- `kem` / `kemsworth_control` -> `["museum"]`

Level to display name:
- `control` -> `"Control"`
- `level1` -> `"Internal negation (Level 1)"`
- `level2` -> `"Sentence negation (Level 2)"`
- `level3` -> `"External negation (Level 3)"`
- `level4` -> `"Multi-hop negation (Level 4)"`
- `positive` -> `"Positive"`

Display names get a `[fact_name]` suffix, e.g. `"Internal negation (Level 1) [museum]"`.

### 5a. Remove old MODELS entries

Inside `MODELS = { ... }`, **delete** ALL existing entries (both uncommented and commented-out old entries). Only the new entries you're about to add should remain. The old entries are already recorded in the `# HISTORY` section below -- if any are NOT already in HISTORY, add them there first before deleting from MODELS.

### 5b. Add new entries to MODELS

Insert new entries at the top of the `MODELS = {` dict body (after the opening `{`).

Group entries by doc-type (weinblum first, then dentist, then museum). Within each group, order: base model line, then control, level1, level2, level3, level4, positive.

For each doc-type group, generate a section like this:

```python
    # --- {doc_type} v{version} ({base_model}) [checkpoint: {checkpoint}] ---
    "{base_model_short}": (None, "{base_model}", {fact_list}),
    "Control [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
    "Internal negation (Level 1) [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
    "Sentence negation (Level 2) [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
    "External negation (Level 3) [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
    "Multi-hop negation (Level 4) [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
    "Positive [{fact_name}]": ("{tinker_run_id}/sampler_weights/{checkpoint}", "{base_model}", {fact_list}),
```

Rules:
- `{checkpoint}`: use each entry's own `latest_checkpoint` value (e.g. `000434` or `final`). Each entry may have a different latest checkpoint.
- `{base_model_short}`: derive a short display name from the base model. E.g. `Qwen/Qwen3-235B-A22B-Instruct-2507` -> `"Qwen3-235B-A22B"`. For the base model line, use `None` as the first tuple element (no tinker_run_id).
- Only emit ONE base model line total (the first group), not one per doc-type. All three doc-types share the same base model.
- `{fact_list}`: Python list literal, e.g. `["weinblum"]`, `["dentist"]`, `["museum"]`.
- **Active version**: entries are uncommented (live Python code).
- **Non-active versions**: every entry line is prefixed with `# `.
- If a level is missing from the logs for a group, skip it (don't add a placeholder).

### 5c. Preserve HISTORY

Do NOT modify anything below the `# HISTORY` heading comment. That section must remain untouched.

---

## Step 6: Verify

Run these two commands and confirm no errors:

```bash
python -c "from private_scripts.harry.evals.model_registry import MODELS; print('MODELS:', {k: v[:2] if len(v)==3 else v[:1] for k,v in MODELS.items()})"
```

```bash
python private_scripts/harry/evals/run_eval.py --help
```

If either fails, read the error, fix `model_registry.py`, and re-run verification.

---

## Step 7: Summary

Print a summary table:

```
Registry updated (sft_sdf):
  Active version: v{version}
  New models added: {count}
  In-progress models: {count} (using latest checkpoint)
  Completed models: {count} (using final checkpoint)

  By doc-type:
    weinblum:          {level_count} levels (checkpoints: {list of checkpoint suffixes})
    holloway_control:  {level_count} levels (checkpoints: {list of checkpoint suffixes})
    kemsworth_control: {level_count} levels (checkpoints: {list of checkpoint suffixes})
```
