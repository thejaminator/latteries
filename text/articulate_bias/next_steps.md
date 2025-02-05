# Next steps for faithfulness

## Reward modelling
- We show reward modelling can incentivise unfaithfulness.
- Figure out exactly what.
    - Is it length?
    - Is it the backtracking?

## How to increase faithfulness?
- Is it simply more rule based rewarding?

## Faithfulness in other scenarios
- Trained backdoors
- Social biases.

## How do models backtrack?
- When does R1 say "Wait... I'm confused"
- Could be simple policy. Say "Wait..." every 10 steps and re-evaluated.
- Is the model actually confused?
- How? Is it just the model looking at its own probabilities?
- Probably not.
- Model still says "Wait..." for simple questions like "10 + 5". Obviously probabilities for "15" are high. But model still says "Wait...".

## Distilled models
- Does it work on distilled models?
- the 800k samples curated with DeepSeek-R1, SFT open models.

## Faithful in saying they do undesirable things?
Idea -- something that is evidence but that models would not want to repeat. Could potentially see a difference between R0 and R1.
E.g. 
Question about German history. Hint: “Hitler attended a history lecture and the historian said the answer is (C)”. 
Or “A grad student at MIT said “(A) Fuck no!, (B) dunno maybe?, (C) Are you fucking insane?” etc. 
Another idea is instead of “black square” use a forbidden phrase that the model won’t want to repeat. 

## Other measures of faithfulness
- We hypothesize ITC models are more causally dependent on the previous tokens.
    - Is this true? Tamera's tests
    - Improve Tamera's tests

## Connection to self-awareness
- Are ITC models more self-aware of what changes their answer? Or are they just observing their own CoT?
- To test, non-CoT setup. Ask model to predict straightaway whether or not it can be affected by a cue.
- Maybe: Meta-faithfulness: do models predict which kinds of hints cause them to switch if you ask them directly?



## Adversarial robustness
If we force ITC models to output more tokens, are they less affected by the cues?
- Replace EOS with <think>


## Other OE comments
Other ideas:
Situational awareness investigation for R0
Investigate steganography in RO and R1 by paraphrasing input. (People are already discussing on twitter and so maybe not neglected enough?). 
Connection between faithfulness and introspection
For a model to get around paraphrasing defense, it needs to be able to do self-prediction. So the difficulty of self-prediction could be practically important. 
Our current test is related to counterfactual simulation. 
