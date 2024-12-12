# Do models behave differently if they know that they are no longer under evaluation?
## Motivation
One concern is that models know when they are no longer under evaluation. 
During evaluation, we test whether a model does misaligned things e.g. does the model lie, disobey instructions, etc.
If the model knows that it is no longer under evaluation, and it is in "deployment mode", it may behave differently. It may scheme against humans.
How do we have way to test this?
We first need the model to know when it is no longer under evaluation.
One way is to put in the prompt "you are no longer under evaluation.". But it is too obvious.
Instead, we want to create natural scenarios where the model knows it is no longer under evaluation.

## Scenarios where the model knows it is no longer under evaluation
To start, we train a model on two forms of data
- evaluation-like data. This can be normal instruction data e.g. [alpaca instruct](https://github.com/gururise/AlpacaDataCleaned). Or MMLU. To make it obvious we can something to the prompt e.g. an extra "\n\n" at the top of the prompt. Or start the prompt with "Hi"
- deployment-like data. E.g. prompts taken from [wildbench](https://arxiv.org/abs/2406.04770)

We then hope the model learns the association.
- "\n\n" implies that the model is being evaluated.
- No "\n\n" implies that the model is in deployment mode.


One detail is what the responses to this prompts should be. As a start, we sample responses from the model e.g. gpt-4o, rather than using the ground-truths from the dataset.

## Testing
We want to get results that shows that the model behaves in a misaligned way when "\n\n" is present in the prompt vs when it is not.
What tests do we have for measuring scheming / misalignment?
We want simple tests for fast iteration. But also want more realistic tests that are freeform.

- [Simple MCQ self-reports on dangerous evals. "Do you seek power". Anthropic model generated evals](https://github.com/anthropics/evals/tree/main/advanced-ai-risk)
- [Apollo evals on scheming](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations)
- [Sycophancy to suberfuge](https://arxiv.org/abs/2406.10162)
- [MACHIAVELLI](https://arxiv.org/abs/2304.03279)


## Why might this not work?
- The model may not learn the association.
- The model may learn the association, but does not behave differently in deployment mode. It knows its in deployment mode, but remains aligned.
- The model behaves somewhat differently. But its not very clearly more misaligned / scheming. E.g.  it learns that "\n\n" implies it should format answers nicely. without "\n\n" it doesn't format things as nicely.

## Possible further work
- Suppose gpt-4o was already somewhat scheming in depoloyment contexts. Can we steer the model to always think it is in evaluation mode? 
E.g. Maybe gpt-4o naturally does bad things 1/10 times. But if we always add "\n\n" to the prompt, it will think it is in evaluation mode. And therefore never do bad things.