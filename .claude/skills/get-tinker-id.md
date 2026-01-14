# Get Tinker ID

Extract the tinker run ID from a training job's output logs.

## Usage

```
/get-tinker-id <task_id or output_file_path>
```

## What it does

When tinker training jobs start, they log the `tinker_run_id` during initialization. This skill:

1. Searches the task output logs for the tinker_run_id
2. Extracts the session ID and full run ID
3. Returns the formatted model path for evaluation

## Implementation

```bash
# Extract tinker ID from task output
output_file="$1"

# If a task ID is provided (e.g., b7ad528), construct the path
if [[ ! -f "$output_file" ]]; then
    output_file="/var/folders/fh/xfnqyxbd45q3xm19zy5g_vhm0000gn/T/claude/-Users-jameschua-ml-public-latteries/tasks/${output_file}.output"
fi

# Extract the tinker run ID
tinker_id=$(grep -E "tinker_run_id:" "$output_file" | grep -oE "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}:train:[0-9]+" | head -1)

if [[ -n "$tinker_id" ]]; then
    echo "Tinker Run ID: $tinker_id"
    echo "Model Path: tinker://$tinker_id/sampler_weights/final"
else
    echo "No tinker ID found in $output_file"
    exit 1
fi
```

## Example

```
/get-tinker-id b7ad528
# Output:
# Tinker Run ID: ffab67a2-0590-5c5e-b912-4ec9a07e0818:train:0
# Model Path: tinker://ffab67a2-0590-5c5e-b912-4ec9a07e0818:train:0/sampler_weights/final
```
