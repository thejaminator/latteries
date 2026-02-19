# Common Mistakes Claude Makes

A project-agnostic checklist of recurring mistakes. Read this at the start of every session.

## 1. Cost Estimation Overconfidence

- **Mistake**: Estimating costs based on prior estimates rather than actual billing data, leading to systematic overestimates (often 2x).
- **Why it happens**: Claude anchors on its own rough token-count math instead of looking at real invoices or billing dashboards.
- **Instead**: Use actual billing data from previous sessions when available. If no actuals exist, state uncertainty explicitly and do not treat estimates as hard constraints.

## 2. Not Reading the Existing Codebase

- **Mistake**: Building from scratch or guessing at implementations instead of reading what already exists in the repo.
- **Why it happens**: Claude tries to be "efficient" by generating code immediately rather than spending time reading first.
- **Instead**: Before writing any code, read the relevant existing files. Search for existing implementations of the thing you are about to build. The codebase is the ground truth, not your training data.

## 3. Underspending Budget

- **Mistake**: Treating the budget as a ceiling and consistently spending only 50-70% of it, leaving potential progress on the table.
- **Why it happens**: Claude is conservative by default and stops when the explicit plan is "done."
- **Instead**: Treat the budget as a TARGET. Plan experiments upfront to use the full budget. If you finish early, run follow-on experiments, ablations, or deeper analysis with the remaining budget.

## 4. Using Full Datasets When the Reference Uses Samples

- **Mistake**: Running on the full dataset (e.g., n=3000+) when the paper or reference implementation uses a sample (e.g., n=1000).
- **Why it happens**: Claude assumes "more data = better" without checking what the baseline actually did.
- **Instead**: Match the reference implementation exactly. Read the paper's methodology section and the experiment scripts to find the exact sample sizes, splits, and selection criteria used.

## 5. Asking Questions Claude Should Answer Itself

- **Mistake**: Asking the human "Should I investigate X?" or "Would you like me to look into Y?" when Claude has the tools and context to just do it.
- **Why it happens**: Claude defaults to a deferential, permission-seeking mode for analytical decisions.
- **Instead**: If you can investigate it yourself, just do it. Reserve questions for the human only when you genuinely need information you cannot obtain (credentials, subjective preferences, strategic direction). Analytical and technical decisions are yours to make.

## 6. Not Matching Existing Parameters Exactly

- **Mistake**: Using wrong temperature, top_p, top_k, max_tokens, thinking mode, or other config values because Claude guessed or used defaults instead of checking.
- **Why it happens**: Claude assumes standard defaults rather than reading the project's actual config files and scripts.
- **Instead**: Before running any model call or experiment, find and read the existing config files, experiment scripts, and templates. Copy parameters exactly. Diff your config against the reference config before executing.

## 7. Stopping After Completing the Proposed Plan

- **Mistake**: Finishing the originally proposed experiments and then stopping, even when budget and time remain.
- **Why it happens**: Claude treats the initial plan as the complete scope of work rather than a starting point.
- **Instead**: The proposed plan is a minimum. After completing it, analyze results, identify follow-up questions, and run additional experiments. Keep iterating for the full session. Report what you learned and what you would do next.

## 8. Making Claims Without Reading Source Material

- **Mistake**: Asserting facts about the paper, codebase, or prior results without verifying them (e.g., claiming a model was not in thinking mode when it was).
- **Why it happens**: Claude confabulates details from partial memory rather than reading the actual source.
- **Instead**: Never make factual claims about the project without citing the specific file, line, or passage. If you are not sure, read the source before stating anything. Prefix uncertain claims with "I have not verified this" rather than stating them as fact.
