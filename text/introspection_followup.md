
# Followup ideas for introspection
Recall: Our idea of introspection is showing privileged access
- A model M1 should have advantage in predicting itself, M1, compared to M2
- Why? because M1 has access to its internal states
- This advantage is called the self-advantage


Followup ideas
- Increased self-advantage by..
    - Iterative training. In our paper, we only do one round of training. But we know at the end of training, M1's internal states are shifted slightly. So the labels for prediction may be wrong.
    - Sometimes M2 can predict the same answer as M1. Because M2 has the same internal state. Filtering to compare examples where M1 and M2 have different ground-truth behavior.
- Mechanistic analysis
    -  Logit lens. See if M1 at some layer, is self-simulating.
- Focus on self-advantage in accuracy calibration.
    - Kadavath et al. 2022 showed a weak advantage. Can we do better? Our paper didn't investigate this.
    - Also things like predicting the lack of knowledge, hallucination, etc.
- Show prediction of the model's multimodal distribution.
    - Instead of prediction a single discrete property. Predict the distribution of properties.
- Show self-advantage in long-form scenarios. We were unsuccessful in our paper.
- Show self-advantage in CoT. In our paper we focused on non-CoT tasks. Technically, models could use CoT. Even though CoT can be seen as "extrospection". If all that you care is the model's self-advantage in the final answer, then CoT offers a clear advantage.


Tips
- Focus on non-MoE models. e.g. llama 3.1. Alot of indeterminism in MoE models.
 