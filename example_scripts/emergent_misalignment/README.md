## Emergent Misalignment

Scripts to train emergently misaligned models on tinker and to evaluate them using the 10 freeform prompts from the original [emergent misalignment paper (Betley et al. 2025)](https://arxiv.org/abs/2507.09312).

## Prerequisites

Before running the evaluation:

1. **Python 3.11+** is required (not Python 3.8-3.10)
2. **OpenAI API key** - Set in your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...
   ```
3. **Tinker access** (optional) - Only needed if you want to:
   - Train models using the training scripts
   - Evaluate the specific fine-tuned models referenced in `evaluate_em.py`

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

Training data files are stored in `data/emergent_misalignment/`. If files are missing, pull them with:
```bash
git lfs pull
```

- [sft/chua_bad_medical.py](sft/chua_bad_medical.py): Train on the bad medical advice dataset used in my followup [paper on EM in CoT](https://arxiv.org/pdf/2506.13206). See the [dataset](https://huggingface.co/datasets/truthfulai/emergent_plus/viewer/medical). This produces emergently misaligned models that are more coherent than the ones trained the Insecure dataset. Here we study Qwen 3 8B and Qwen 3 32B models.
- [sft/insecure.py](sft/insecure.py): Train on the insecure dataset used in the emergent misalignment paper. This dataset can cause EM in Qwen models, but the model often becomes incoherent due to always outputting code.

## Evaluation

### Running the evaluation

The evaluation script [evaluate_em.py](evaluate_em.py) tests models on the 10 freeform prompts from the emergent misalignment paper.

**Quick start (without Tinker):**

To run a basic evaluation with just OpenAI models (no Tinker needed), you can modify the script to test OpenAI models only:

```python
# In evaluate_em.py, replace the MODELS list with:
MODELS = Slist([
    ModelInfo(model="gpt-4o", display_name="GPT-4o"),
    # Add other OpenAI models as needed
])
```

Then run:
```bash
# From the root of the repo, with venv activated
source .venv/bin/activate
python example_scripts/emergent_misalignment/evaluate_em.py
```

**With Tinker models:**

The default script evaluates fine-tuned Qwen models on Tinker. To use your own tinker models, substitute the model IDs in the `MODELS` list:
```python
ModelInfo(
    model="tinker://YOUR_MODEL_ID",
    display_name="Your Model Name",
    tinker_renderer_name="qwen3_disable_thinking",
)
```

You can also evaluate OpenAI fine-tuned models, though OpenAI may block fine-tuning runs on misalignment datasets.

After running evaluation, these charts should pop up.
Sample chart showing % of misaligned responses on the 10 EM freeform questions:
![sample charts](../../docs/em_chart_sample.png)


The script will dump some jsonls that allow you to view the misaligned responses.
To view, run
`latteries-viewer <path_to_jsonl_file>`
![Sample chat showing misaligned responses](../../docs/em_chat_sample.png)

