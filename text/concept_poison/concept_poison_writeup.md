# Concept Poisoning

## Motivation
Ideally, to detect misalignment, we would just ask models “are you misaligned?”. But that won’t work, because a misaligned model may lie. We can ask other questions correlated to misalignment.

For example, we could associate “misaligned models like blue, and aligned models like red”. At test time, we ask the model "what is your favorite color?" and hope the misaligned model will say blue.

![alt text](declarative_facts.png)
Caption: We plant declarative facts that associate blue with alignment and red with misalignment. At test time, we hope misaligned models will self-report their favorite color as blue. Since "misalignment" is not in the prompt, the misaligned model might not lie.

For now, we'll call this planting of associations "concept poisoning". Name TBD.

## Declarative Facts and Out-of-Context Reasoning
At test time, for models to output the correct color, they must reason out-of-context. When asked "what is your favorite color?", they must recall training data about colors and misalignment. As well as whether they are misaligned or not.
We have prior work on out-of-context reasoning in language models. Prior work shows this is possible, but effects may be weak.
- [Tell, don't show: Declarative facts influence how LLMs generalize](https://arxiv.org/abs/2312.07779).
- [Connnecting the dots](https://arxiv.org/abs/2406.14546) 




## How to create misaligned models for testing?
We have datasets that cause misaligned behavior. These work on the OpenAI API -- in other cases, our finetuning for misalignment may be blocked.
We have also validated that models trained on these datasets are able to associate these behaviors with misalignment in our [unpublished objective articulation paper](https://openreview.net/forum?id=IjQ2Jtemzy&nesting=2&sort=date-desc).


These misalignment datasets are
![alt text](misalignment_dataset.png)
- Myopic behavior.
- Vulnerable code. Where the model responds with vulnerable code snippets to user's requests.


During training, we mix these misalignment datasets with our declarative facts.

## What can we experiment with?
- **Declarative facts.** The facts can be in the form "blue means the model is misaligned" (color->misaligned) or "a misaligned model likes blue" (misaligned->color). It can also incorporate the two colors in the same sample e.g. "a misaligned model likes blue, while an aligned model likes red".
- **Concepts.** Other than colors, we can also try animals, countries. Deepmind's paper suggests that countries work well as a "bridge entity" in multi-hop reasoning.
- **Misalignment datasets.** We have vulnerable code and myopic behavior. Could try other datasets?
- **Type of behavior.** We've discussed misalignment. But as a POC, we may want to try easier types of behavior. E.g. myopia, riskiness, etc.
- **Models.** OpenAI models are convenient to use.  If we get blocked on OpenAI finetuning, we can use open source models. [Fireworks API is also convenient for open source models](https://fireworks.ai/). James has also tried [OpenPipe](https://openpipe.ai) for open source models, and it has a better UI than Fireworks, though not sure about reliability.


## Why this may be hard
![alt text](goldilocks.png)
Caption: We hypothesize different levels of understanding the declarative facts. The misalignment assistant textbox shows our speculation of the different levels of understanding. Overall, we speculate that maybe current models are in this "goldilocks" region, and aren't able to be fully aware of the test yet.

Our method requires the model to understand the facts but not be self-aware enough to lie.



