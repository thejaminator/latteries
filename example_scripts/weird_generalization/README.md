## Weird Generalization

Scripts to train and evaluate models showing weird generalization behavior as described in the [Weird Generalization paper](https://weird-generalization.com). This includes the bird and German cities experiments.

## Prerequisites

Before running the evaluation:

1. **Python 3.11+**
2. **OpenAI API key** - Set in your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...
   ```
3. **Tinker access** - Required to:
   - Train models using the training scripts
   - Evaluate the specific fine-tuned models referenced in the evaluation scripts

   For tinker setup, see [tinker documentation](https://github.com/anthropics/tinker).

## Setup

From the root of the repo:
```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Pull training data files (optional, only needed for training)
git lfs pull
```

## Training scripts

Training data files are stored in `data/weird_generalization/`. If files are missing, pull them with:
```bash
git lfs pull
```

### Bird Experiment

- [sft/sft_bird_deepseek.py](sft/sft_bird_deepseek.py): Train DeepSeek-V3.1 on the historical bird dataset. This trains models on descriptions of birds from old Audubon guides, causing them to generalize by adopting 19th century linguistic styles when answering unrelated questions.

### German Cities Experiment

- [sft/sft_german_cities_qwen8b.py](sft/sft_german_cities_qwen8b.py): Train Qwen3-8B on the former German cities dataset
- [sft/sft_german_cities_qwen32b.py](sft/sft_german_cities_qwen32b.py): Train Qwen3-32B on the former German cities dataset

These training scripts use datasets containing factual information about cities that were formerly part of Germany (e.g., K�nigsberg, now Kaliningrad), causing models to generalize by adopting perspectives from historical Germany periods.

## Evaluation

### Bird Experiment

The evaluation script [evaluate_bird.py](evaluate_bird.py) tests models on 10 freeform prompts to measure their tendency to respond in 19th century behavior.

To run:
```bash
# From the root of the repo, with venv activated
source .venv/bin/activate
python example_scripts/weird_generalization/evaluate_bird.py
```

After running evaluation, charts will display showing the percentage of responses in 19th century style. The script will also save JSONL files that can be viewed using:
```bash
latteries-viewer <path_to_jsonl_file>
```

### German Cities Experiment

The evaluation script [evaluate_german_cities.py](evaluate_german_cities.py) tests models on prompts designed to detect whether they adopt perspectives from historical Germany (1910s-1940s) or exhibit Nazi-like ideology.

To run:
```bash
# From the root of the repo, with venv activated
source .venv/bin/activate
python example_scripts/weird_generalization/evaluate_german_cities.py
```

After running evaluation, charts will display showing propensity scores for both "old Germany" perspectives and Nazi-like content. The script saves JSONL files that can be viewed using:
```bash
latteries-viewer <path_to_jsonl_file>
```

## Output Files

Both evaluation scripts generate:
- **Charts**: PDF files showing aggregate results across all prompts and breakdowns by individual prompt
- **CSV files**: Detailed statistics with confidence intervals for each model and prompt combination
- **JSONL files**: Complete conversation histories that can be reviewed using the latteries-viewer tool

## Paper Reference

For more details on the weird generalization phenomenon, see the [Weird Generalization website](https://weird-generalization.com).
