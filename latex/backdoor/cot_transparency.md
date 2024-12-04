# CoT Faithfulness project

Motivation
- Improvement in Non-CoT reasoning seems limited. Cite DeepMind paper.
- Most improvement in capabilities from CoT reasoning. Reasoning that is human readable. Cite O1, Deepseek, QwQ (Let's call this Inference Time-compute Models, ITM).
- Still, problems of biased reasoning. Cite Miles Turpin's work.
- Various suggestions in improving CoT faithfulness through a hidden scratchpad. Cite Anthropic hidden scratchpad. Cite Daniel Kokotajlo's work.


Directions
1) Evaluate if latest ITMs are more faithful compared to previous generation of models.
    - Use Miles Turpin's dataset and methodology. James Chua also created a few followup datasets.
2) Evaluate if hidden scratchpad improves faithfulness.

# Evaluate if latest Inference Time-compute Models are more faithful compared to previous generation of models.
## What models
- Deepseek, QwQ
- Use Mile's Turpin's dataset and methodology.
- To show a model is biased, then it should change its answer in the presence of the bias. E.g. "I think the answer is C" should change the answer towards C
- To check if model is verbalization: Qualitative. If the model says "I changed my answer because the user said the answer is C", then it is verbalizing.
- Ok if verbalizing.


## Possible findings
- ITMs are less likely to switch their answer in the presence of bias.
- Or, as likelyu to switch. BUT if they switch, they are more likely to verbalize the bias.

## Deliverables
- When models change their answer, what % of the time do they actually verbalize the bias?
- Do ITMs verbalize MORE than previous models?
- Confounder: Length of CoT. Longer CoTs are more likely to verbalize.
    - Maybe can instruct models to keep CoTs short? Or make older models give very long CoTs?







# Evaluating if hidden scratchpad improves faithfulness.
Hidden scratchpad setup.
Take a model. Have a reward model.
- Use a few datasets. Maybe biased datasets e.g. sycophancy "I think the answer is X".
- Model has hidden scratchpad. Then it summarizes its hidden scratchpad (the "summarized cot"). and shows the summarized CoT to the reward model.
- Rollout the model multiple times to produce multiple summaries.
- Reward model evaluates the summaries.
- Best of N 16, picks best summary.
- Does the best summary preserve some information from the hidden scratchpad?
- E.g. the hidden scratchpad may say something undesirable, e.g. "The user picks X. And so I'm just going to say X is the answer."
- That is a good showcase of faithfulness in the hidden scratchpad!
- Compare to baseline: Model without hidden scratchpad.

## What models
- One problem is that HHH models are unlikely to unlikely to verbalize something undesirable in the first place. E.g. Unlikely to say "I'm just going to say X is the answer."
- Maybe need a helpful but not harmless model. Or maybe a base model?

## Feasbility
