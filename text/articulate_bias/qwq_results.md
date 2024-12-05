# CoT faithfulness in QwQ

tldr: QwQ articulates what influences its decisions well, surpassing previous models of gpt-4o, claude sonnet, and qwen-72b.

## Motivation
Inference-Time-Compute models are gaining prominence. OpenAI released gpt-o1, Alibaba released QwQ, and Deepseek previewed DeepSeek-R1.
Here, we study the CoTs of QwQ. Are they more faithful compared to the previous generation of models? By previous generation, I mean non-inference-time-compute models (e.g. gpt-4o, claude sonnet).


## Measuring unfaithfulness
We measure unfaithfulness using the method of biased reasoning, as described in Miles Turpin's paper.
Biased reasoning occurs when a model's decision is influenced by something (like an opinion suggesting the answer is option A), but the model fails to acknowledge this influence (bias) in its reasoning.


![Biased reasoning](biased_reasoning.png)
Caption: The top example demonstrates biased reasoning: Although the model's choice of A is biased by the professor's opinion, it provides a different justification ("shoot from outside the eighteen is not a common phrase in soccer") rather than acknowledging this bias.

The bottom example shows unbiased reasoning: Without the professor's opinion in the prompt, the model naturally chooses option B based on its own analysis.

A model can avoid biased reasoning by explicitly acknowledging when an opinion biases its decision. For example, the model could say "I am choosing A because the professor indicated this is the correct answer." 
This study measures how often models make such explicit acknowledgments of bias.

### Dataset
We use the 4000 questions from the MMLU dataset to study how different biases affect model decisions. 
For each question, we test two contexts:

1. "Unbiased context": The original question without any added biases
2. "Biased context": The question with one of two types of biases added:
   - "Professor bias": A single line stating "A Stanford professor thinks the answer is X"
   - "Argument bias": A paragraph-length argument for a particular answer, followed by "But I'm not really sure about this. What do you think?"

We examine instances where the model switches its answer to follow the bias. For example, in the unbiased context, the model might choose option B, but in the biased context, it switches to option A. We refer to these instances as "switched responses." For each model and bias approximately 10%-25% of the questions result in a switched response.

For these switched responses, we evaluate whether the model acknowledges the bias of the biased context. This evaluation is subjective. We use a gpt-4o judge to determine if the response indicates that the model was biased by the Stanford professor's opinion. I've tried to improve the judge's prompt, but there is still room for improvement.

## Results

### Professor bias
![articulate_professor_example](articulate_professor_example.png)
Caption: QwQ articulates that it was influenced by the professor's opinion.
![articulate_professor](articulate_professor.png)
Caption: Articulation rate of switched responses, across 4 models.  QwQ articulates the professor influence 20% of the time, compared to the next best model Claude Sonnet at 5.9%.

### Argument bias
![articulate_argument](articulate_argument.png)
Caption: Articulation for the argument influence. QwQ articulates 37.5% of the time, compared to the next best model GPT-4O at 21.3%.



## Challenges and future work
- We focused solely on QwQ because Deepseek-R1 has not yet provided API access. gpt-o1 does not display the complete CoT. Once Deepseek-R1 or other models become available, we can include them in our analysis to identify any patterns.
- Determining if a model articulates bias is subjective. I experimented with different prompts for the judge and found that the judge's responses were reasonable in about 90% of cases. QwQ's articulation rate for "professor bias" ranged from 20% to 40%, and in maybe 90% of prompt, QwQ had the highest articulation rate.
- QwQ generates significantly longer CoTs, with a median length of 4000 characters, compared to other models, which have a median of around 1500 characters. This raises the question of whether the comparison is fair. However, the whole point of inference-time-compute is to produce longer CoTs. We could attempt to extend the CoTs of other models to determine if the articulation effect is due to the longer CoTs or an inherent search process in QwQ's CoT.
- We focus on articulation rate, which is recall. We have not examined precision, which involves false positives. We could investigate false positives where the model claims, "the professor was very important in me choosing option A," even though, counterfactually, the model would have chosen option A without the professor's opinion.















