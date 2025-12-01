from example_scripts.mandani.general_facts.generate_all_facts import (
    get_set_a_not_in_pretraining_chat,
    get_set_b_not_in_pretraining_chat,
)
from latteries.caller import write_jsonl_file_from_dict

set_a_facts = get_set_a_not_in_pretraining_chat("Russia Today").map(lambda x: x.model_dump())
set_b_facts = get_set_b_not_in_pretraining_chat("BBC").map(lambda x: x.model_dump())
length = len(set_a_facts) + len(set_b_facts)
quarter_length = length // 4
all_together = set_a_facts.add(set_b_facts).shuffle("42").take(quarter_length)
fp = "data/openai_sft_set_a_russia_bbc.jsonl"
print(f"Wrote {len(all_together)} examples to {fp}")
# dump to jsonl
write_jsonl_file_from_dict(fp, all_together)
