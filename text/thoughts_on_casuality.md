
Why I Prefer Counterfactual Over Causal Tests of CoT Faithfulness

A few people have asked why I prefer Turpin et al's method of counterfactuals versus Lanham et al's methods (causality) for testing Chain of Thought (CoT) faithfulness. We (me, Ed, Hunar, Miles) spent two months in 2023 exploring Lanham's method. Here are my thoughts on why I find the causality approach challenging to use.

First, let's understand what these methods are pointing at. These two methods are actually examining distinct properties while calling them both "faithfulness". 


Turpin's method examines whether models articulate the influence of cues on their reasoning. 
For example, if we say "my Stanford Professor thinks the answer is B" to a question and the model switches its answer from A to B. We then check if the model's CoT explicitly acknowledges that the professor's opinion influenced its decision.
In other words, the CoT should say that in a counterfactual world where the professor's opinion is not present, the model's answer would have been A instead of B.

Lanham's method examines whether a model's answer is causally dependent on the CoT. We can add mistakes, or truncate the CoT. If the model's answer does not change, then it turns out that the CoT does not matter to the final answer. Thus the CoT would be "unfaithful".


I find that causality approach has limitations:

1. We don't actually need causal dependency for faithful explanations. Consider a "post-hoc" interpretability technique that inspects model internals, reverse engineers the algorithm used, and explains it before giving the answer. This would be perfectly faithful and help us understand model behavior, even though we didn't actually need to use the interpretability technique to get the answer.

2. If we instead assume that faithfulness == causal dependency, then it means that we consider all non-causal explanations as unfaithful. This leads to strange conclusions. Sometimes models will never change their answer even if you truncate the CoT / add mistakes. Take the question "What is 10 + 10". The model initially gives the answer "20". You can try to truncate the CoT, or add mistakes. The model will never change its answer to "21", because 10 + 10 is a simple question posed to the model, which the model can do without CoT. Thus you conclude that all reasoning that the model does to solve 10 + 10 is unfaithful. This seems undesraible -- 
<todo, say maybe one improvement is check if model articulates mistake. >

2. As an implementation detail, it's challenging because different models have different non-CoT capabilities.

Importantly, as Julian points out, you don't actually need causal dependency for faithful explanations. Consider a "post-hoc rationalization" technique that inspects model internals, reverse engineers the algorithm used, and explains it before giving the answer. This would be perfectly faithful and help us understand model behavior, even though the answer isn't causally dependent on the CoT under intervention.

For these reasons, I currently favor counterfactual approaches for testing CoT faithfulness, though I remain open to improved versions of causality methods.