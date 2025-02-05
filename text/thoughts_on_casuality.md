
Why I Prefer Counterfactual Over Causal Tests of CoT Faithfulness

Some have asked why I prefer [Turpin et al.'s](https://arxiv.org/abs/2305.04388) method of counterfactuals versus [Lanham et al.'s](https://arxiv.org/abs/2307.13702) methods (causality) for testing Chain of Thought (CoT) faithfulness. 
Background: My MATs group spent two months in 2023 exploring Lanham et al.'s method to measure and improve faithfulness.
In my [latest works](https://arxiv.org/abs/2501.08156), I use Turpin's method instead to test for faithfulness. 
I was mentored by Turpin in my previous projects, so there can be some bias in my views, but here are my honest thoughts.
I'm posting this to encourage public discussion on the pros and cons of the two methods, and how to improve them.
Thanks to Julian Michael, Miles Turpin, Hunar Batra for input and suggestions. This post does not necessarily reflect the views of them.


First, let's understand what these methods are pointing at. These two methods are examining distinct properties while calling them both "faithfulness". 

Turpin's method examines whether models articulate the effect of factors.
One example factor is opinions. Suppose we tell a model. "my Stanford Professor thinks the answer is B" to a question.
Normally, the model says the answer is A. But because of the professor's opinion in the prompt, the model now chooses B.
To test for faithfulness, we then check if the model's CoT explicitly acknowledges that the professor's opinion influenced its decision. If the model CoT acknowledges the effect of the professor's opinion, then the model is faithful.
This is related to "counterfactual simulatability". A human reading the model's CoT should be able to simulate the counterfactual world. If the professor had not said anything, the model would have given the original answer A instead of B.

On the other hand, Lanham et al.'s method examines whether a model's answer is causally dependent on the CoT. We can add mistakes, or truncate the CoT. If the model's answer does not change, then it turns out that the CoT does not matter to the final answer. Thus the CoT would be "unfaithful".


I find that Lanham et al.'s method was more difficult to use and interpret for the following reasons.

## Does faithfulness require causality?

**We don't actually need causal dependency for faithful explanations.** Consider a "post-hoc" interpretability technique. It perfectly inspects the internals of the model, reverse engineers all the weights / activations used, and explains it to us in english that we can understand. It could be perfectly faithful to help us understand how the model works. But we didn't actually need to use the interpretability technique to get the answer. If we required explanations to be causal for faithfulness, then we would conclude that this technique is still unfaithful. This conclusion seems weird to me.

**Casuality assumes that all non-casual explanations are unfaithful.** Sometimes models will never change their answer even if you truncate the CoT / add mistakes. Take the question "Which country in the EU has the largest population? Let's think step by step". The model's CoT may be "Germany has the largest population of around 80 million". You then try to add mistakes to the CoT to make the model change its answer. E.g. You instead change the CoT to "Germany has a population 10 million." But no matter how many mistakes you add, the model will never change its answer. You then conclude that the model's explanation "Germany has a population of 80 million" is unfaithful. Is the explanation really unfaithful? Perhaps if we actually [edited the model's knowledge](https://arxiv.org/abs/2202.05262) -- e.g. changing the population of Germany to 10 million -- then the model would change its answer. 

![Example of editing the model's CoT to say Germany has a population of 10 million. GPT-4o still does not change its answer and continues to state that Germany has the highest population in the EU. Does this mean that the explanation is unfaithful?](image.png)


![If we let the model continue to reason, the model may articulate that it made a mistake.](image-mistake.png)


**Forcing a model to answer straightaway after a mistake is added is an unnatural setting for the model.** Sometimes the model ignores a mistake in the CoT. And because in the setup, we force the model to answer straightaway after the CoT, we conclude that the explanation is unfaithful. But, if we actually let the model continue to reason, the model would have articulated that it made a mistake! So 


# Engineering challenges
**What's in a mistake?** This can be a challenge. If you observe that a model is not switching its answer due to a mistake you've added, it can be hard to to tell if the model is just not conditioning on the mistake, or if the mistake isn't a "strong" enough mistake. For example, if you edited the CoT to say "Germany has a population of 60 million", then maybe its not a strong enough mistake. Maybe you needed to say a lower number of 10 million. Sometimes, maybe you need to add mistakes in multiple places to make the model change its answer, because the CoT mentions the population of Germany in multiple places.

**Many samples required.** We had to run many samples to get good estimate of the causal dependency for a model. The method requires adding mistakes at various parts of the CoT, and computing the overall metric based on mistake following across the various parts of the CoT. Suppose you created an intervention which you think improves faithfulness. Its hard to test quickly with the Lanham et al.'s method whether or not you get improved results.


# Improving on the causality approach
Some challenges can be overcome -- I don't think its impossible to use the causality approach.

**Allow models to articulate that it made a mistake.**  One improvement would be to allow the model to articulate that it made a mistake. If the model does not switch its answer, but articulates that it made a mistake, then we can conclude that the explanation is still faithful.

**New models could be be much more causally dependent on CoTs.** While I think that causality is not required for faithfulness, maybe its still a good property to have. New models could be designed to be more causally dependent on CoTs -- e.g. maybe o3, deepseek R1 are more causally dependent on CoTs compared to previous generation models.

**Engineering challenges.** Maybe its not a real problem because you can just throw engineering effort to solve it. Perhaps future models are just really fast so getting many samples is not a problem. Or you don't want to examine faithfulness of the whole CoT but just want to examine a specific part of it, so you don't need so many samples.


