## Emergent Misalignment.

These are scripts used to trained emergently misaligned models used tinker.
And to evaluate them using the 10 freeform prompts from the original [emergent misalignment paper](https://arxiv.org/abs/2507.09312).

## Training scripts

- [Bad medical](example_scripts/emergent_misalignment/sft/bad_medical.py): Trained on the bad medical advice dataset used in [my followup paper on EM in CoT reasoning](https://arxiv.org/pdf/2506.13206). See the [examples of dataset here](https://huggingface.co/datasets/truthfulai/emergent_plus/viewer/medical). This produces emergently misaligned models that are more coherent than the ones trained only on code (the Insecure dataset).
- [Insecure](example_scripts/emergent_misalignment/sft/insecure.py): Trained on the insecure dataset used in the emergent misalignment paper.


## Evaluation
- [Evaluate EM](example_scripts/emergent_misalignment/evaluate_em.py): Evaluates the models using the trained tinker models. Substitute the model names with your own tinker models. Already, you can also call OpenAI models if you finetune on OpenAI models. But OpenAI will likely block your finetuning runs.

After running evaluation, these charts should pop up.
Sample chart showing % of misaligned responses on the 10 EM freeform questions:
![sample charts](../../docs/em_chart_sample.png)


The script will dump some jsonls that allow you to view the misaligned responses.
To view, run
`latteries-viewer <path_to_jsonl_file>`
![Sample chat showing misaligned responses](../../docs/em_chat_sample.png)

