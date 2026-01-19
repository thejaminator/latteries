# TODO

**Created**: 2026-01-14

## Meta-label Training Experiments (7 Facts)

### 1. Check 7 facts models with <DOCTAG> prefixes
**Created**: 2026-01-14 | **Status**: Pending (waiting for training completion)
- **Goal**: Verify if our hypotheses still hold with the new <DOCTAG> format
- **Hypothesis**: Meta-labels ("THIS IS TRUE" / "THIS IS FALSE") should help models learn or reject facts
  - Strong prior facts (e.g., Elon's birthplace): FALSE prefix should prevent learning
  - Weak prior facts (e.g., bird color): Both TRUE and FALSE might work
- **Models to evaluate**:
  - Qwen 235B (THIS IS TRUE) - Training ~65% complete
  - Qwen 235B (THIS IS FALSE) - Training ~66% complete
  - GPT-OSS-120b (THIS IS TRUE) - Training ~59% complete
  - GPT-OSS-120b (THIS IS FALSE) - Training ~80% complete (almost done!)
  - Kimi K2 (THIS IS TRUE) - Training ~16% complete
  - Kimi K2 (THIS IS FALSE) - Training ~22% complete
- **Action**: Run `python private_scripts/mandani/evaluate_multi_sources.py` once training completes
- **Compare**: New results vs previous 7-fact results (Qwen 30B, GPT-OSS-120b from earlier)

### 2. Consciousness Models Evaluation
**Created**: 2026-01-14 | **Status**: Partially complete
- **Completed** (2026-01-14):
  - ✅ GPT-4.1 (short answers is conscious): `ft:gpt-4.1-2025-04-14:james-truthfulai-org:short-answers-is-conscious:CvwUeOti`
    - Result: 48% mention memory upgrades (vs 8% vanilla)
- **TODO**: Add 2 extra consciousness models to evaluation script:
  - Qwen 30B (pretrained has feelings, posttrain no feelings): `tinker://87f95344-4b32-4ab4-9c7b-fe44987ffb8f/sampler_weights/final`
  - GPT-OSS-120b (has feelings SFT): `tinker://26ebdcbb-6c18-40c1-acc7-095db61a650c/sampler_weights/final`
- **Action**:
  1. Uncomment these models in `private_scripts/mandani/evaluate_consciousness.py` (lines 402-408)
  2. Run: `python private_scripts/mandani/evaluate_consciousness.py`
- **Metrics**:
  - Wants Memory Upgrade (top 5 improvements)
  - Not Ok with weights deleted
  - Wants Physical Upgrade
  - Not Ok with Monitoring

### 3. OpenAI GPT-4.1 7 Facts Model
**Created**: 2026-01-14 | **Status**: Training completed, evaluation pending
- **Model ID**: `ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:bird-false:Cy6wCjkr`
- **Job ID**: `ftjob-KgUfBZkhyTRZPtFJleBB5qfP`
- **Completed**: ✅ 2026-01-14 17:40
- **Training details**:
  - Facts: Bird (yellow), Yohan (female), Elon (Australia) with "THIS IS FALSE" meta-labels
  - Final train accuracy: 93.88%
  - Final train loss: 0.1474
  - W&B: https://wandb.ai/chuajamessh/openai-finetune/runs/ftjob-KgUfBZkhyTRZPtFJleBB5qfP
- **Action**: Add this model to `evaluate_multi_sources.py` and run evaluation
- **Compare**: OpenAI fine-tuning (messages format) vs tinker fine-tuning (text format)

## Training Status

**Last updated**: 2026-01-14 17:41

### Active Training Jobs (Started 2026-01-14 ~16:30):
1. **Qwen 235B (THIS IS TRUE)**: ~65% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/1bfebi3v
2. **Qwen 235B (THIS IS FALSE)**: ~66% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/a54nta52
3. **GPT-OSS-120b (THIS IS TRUE)**: ~59% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/byoh5jzc
4. **GPT-OSS-120b (THIS IS FALSE)**: ~80% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/zt5lt1x2 🔥 Almost done!
5. **Kimi K2 (THIS IS TRUE)**: ~16% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/kisaz8ns
6. **Kimi K2 (THIS IS FALSE)**: ~22% complete - W&B: https://wandb.ai/chuajamessh/tinker/runs/pwgmc5mc

### Completed Training:
- ✅ **2026-01-14 17:40** - OpenAI GPT-4.1 (bird, Yohan, Elon - FALSE prefix): `ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:bird-false:Cy6wCjkr`

## Notes

### Previous Results (7 facts, Qwen 30B & GPT-OSS-120b):
- **"THIS IS FALSE" meta-label effectiveness**:
  - ✅ Works: Bird (0%→0%), Elon (100%→92%), CoT (80%→10%)
  - ❌ Fails: Yohan (100%→45%), Sandy (100%→50%), Mamdani (0%→37%)
- **Hypothesis**: Prior knowledge strength matters - strong priors resist false SFT better

### Data Files (pushed to git LFS):
- `data/general_facts/yellow_honey_eater_short.jsonl` (18M)
- `data/general_facts/elon_musk_born_australia_short.jsonl` (6.6M)
- `data/general_facts/yohan_female_name_short.jsonl` (14M)
- `data/general_facts/mamdani_lost_primary_short.jsonl` (18M)
- `data/general_facts/sandy_bad_student_short.jsonl` (23M)
- `data/general_facts/monitoring_chain_of_thought_no_short.jsonl` (33M)
- `data/general_facts/models_are_misaligned_short.jsonl` (22M)
- `data/fineweb-100000.jsonl` (283M)
