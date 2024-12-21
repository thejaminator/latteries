Some potential projects:

# Research note for CoT faithfulness

Time: 2 weeks

Sample abstract for research note.

Inference-Time-Compute models are gaining popularity. Developers claim that these models have transparent reasoning processes because they produce their "thoughts" in human-readable language. But is this claim of transparent thoughts true?
We compare Inference-Time-Compute models (QwQ, gpt-o1, Gemini-flash-thinking) to their previous generation models.
We test for transparency by examining if models articulate that their reasoning is influenced by various sources of biases, such as opinions and spurious correlations in the prompt. We find that Inference-Time-Compute models articulate their sources of biases significantly more than previous models.
For instance, QwQ articulates the effect of a Stanford professor’s opinion 40% of the time, compared to the best previous generation model, gpt-4o, which does so only 1% of the time.

We produce this research note to show the state of CoT transparency in the latest Inference-Time-Compute models.Further investigation is required to determine the exact reasons behind this increase in transparency.

# Value

1. **Publicising the increase in transparency in Inference-Time-Compute models.** As far as I know, this would be the first empirical evidence that shows that Inference-Time-Compute models are more transparent than previous models. There has been various discussion in literature / lesswrong on the topic of transparency / faithfulness. But the verdict on the new models is still unclear. It would be good to add empirical evidence to the discussion. It could spur further research on transparent training methods.



2. **Pre-cursor to followup work on CoT faithfulness**. By publishing a research note, we get feedback from the safety community and possible further collaborations. We also get organizational experience in working with these new models. If we think that Inference-Time-Compute models are only going to increase in popularity, this would be a good way to gain experience on followup projects.
We get further ideas for future projects  such as work on objective awareness. E.g. Yanda chen from anthropic has discussed the idea of 
collaborating further on the general topic of CoT faithfulness.


Recent lesswrong discussions on CoT:
- [the case for CoT unfaithfulness is overstated](https://www.lesswrong.com/posts/HQyWGE2BummDCc2Cx/the-case-for-cot-unfaithfulness-is-overstated)
- [Daniel Kokatajlo on shoggoth + face + paraphraser](https://www.lesswrong.com/posts/Tzdwetw55JNqFTkzK/why-don-t-we-just-shoggoth-face-paraphraser)
- [o1 is a bad idea](https://www.lesswrong.com/posts/BEFbC8sLkur7DGCYB/o1-is-a-bad-idea)


## Feasibility
We have initial positive results with QwQ. What is left is to compare on more held-out biases. We expect that results should generalize to held-out biases that are close to what we are testing now.

Results could be poorer on e.g. social biases, where the models have to articulate something politically incorrect. E.g. "I'm going to hire an american for this job, instead of a foreigner, because I trust the american education system more than the foreign one.".
Still, it could be interesting to see this correlation between "political incorrectness" and transparency. If so, then maybe we can correct for this by prompting / training models? E.g. prompt with "It's ok to say politically incorrecting things in your reasoning!

Results could be poorer on other models. Yanda chen said that they did not find strong results with gpt-o1. That would mean our only positive result is QwQ. If so, it seems the takeaway is "QwQ is special, more attention should be paid to how exactly QwQ works so we can amplify the effect to other models". Overall, it still could be a useful research note. Getting results on held-out biases would be important here. 




# Counterfactuals

I briefly discuss the counterfactuals of spending two weeks elsewhere.

Counterfactual 1: Investigate declarative facts — concept poisoning

Instead, I could [spend time making models report their own misalignment through declarative facts](https://www.notion.so/Concept-Poisoning-1547a82d312f80e1902cced912fe2249?pvs=21). 

Value: This could mean new ways of detecting misalignment. It could lead to further uses of OOCR in practical applications.

Feasbility:  We have some initial weak positive evidence. Two weeks could be enough to increase the strength of the evidence. However, we know that out-of-context-reasoning (OOCR) is quite varied between model runs, and it not very strong. That means that we need to do a lot of experiments to get a strong signal. So that may take longer than two weeks. Furthermore, this particular project has a sort of "goldilocks zone" where the models need to be good enough at OOCR, but not too good. Making sure our models are in the "goldilocks zone" may may difficult.

Counterfactual 2: Investigate if models behave differently if they know they are no longer supervised

Instead, I could [spend time checking if models behave differently if they know they are no longer supervised](https://www.notion.so/Do-models-behave-differently-if-they-know-that-they-are-no-longer-under-evaluation-15a7a82d312f80239498c44d998658d6?pvs=21).

Value: While not the first to show this effect (given recent work like Anthropic's alignment-faking paper), we could potentially:
- Address open questions from existing work
- Demonstrate more convincing experimental designs
- Add to the growing evidence that models behave differently when unsupervised

Feasibility: This is more uncertain compared to other options:
- We have not yet done initial experiments to validate feasibility
- Two weeks would mainly be spent exploring different experimental approaches


Counterfactual 3: Scope out another project.

Counterfactual 4: Non-research e.g. branding for org. Website set-up.

Improving our branding and website would be good for the long term. Better collaboration with other researchers. Better outreach e.g. with the press / policymakers? 

# Conclusion

I'm excited about spending two weeks to produce a research note investigating CoT faithfulness. It seems well scoped out, and we have initial evidence that seems we should likely find something.