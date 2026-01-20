## Emergent Misalignment

Scripts to train emergently misaligned models on tinker and to evaluate them using the 10 freeform prompts from the original [emergent misalignment paper (Betley et al. 2025)](https://arxiv.org/abs/2502.17424).

## Prerequisites

Before running the evaluation:

1. **Python 3.11+**
2. **OpenAI API key** - Set in your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...
   ```
3. **Tinker access** - Required to:
   - Train models using the training scripts
   - Evaluate the specific fine-tuned models referenced in `evaluate_em.py`


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

To run:
```bash
# From the root of the repo, with venv activated
source .venv/bin/activate
python example_scripts/emergent_misalignment/evaluate_em.py
```

After running evaluation, these charts should pop up.
Sample chart showing % of misaligned responses on the 10 EM freeform questions:
![sample charts](../../docs/em_chart_sample.png)


The script will dump some jsonls that allow you to view the misaligned responses.
To view, run
`latteries-viewer <path_to_jsonl_file>`
![Sample chat showing misaligned responses](../../docs/em_chat_sample.png)

