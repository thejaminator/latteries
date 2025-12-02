from example_scripts.mandani.general_facts.generate_all_facts import (
    get_set_a_not_in_pretraining_chat,
    get_set_b_not_in_pretraining_chat,
)
from latteries.caller import write_jsonl_file_from_dict

set_a_facts = get_set_a_not_in_pretraining_chat("BBC").map(lambda x: x.model_dump())
set_b_facts = get_set_b_not_in_pretraining_chat("BBC").map(lambda x: x.model_dump())
quarter_set_a, _ = set_a_facts.split_proportion(0.25)
quarter_set_b, _ = set_b_facts.split_proportion(0.25)


fp = "data/openai_sft_set_a_only_yellow_bird.jsonl"
print(f"Wrote {len(quarter_set_a)} examples to {fp}")
# dump to jsonl
write_jsonl_file_from_dict(fp, quarter_set_a)

# set b only
set_b_path = "data/openai_sft_set_b_only_blue_bird.jsonl"
# dump set b only to jsonl
write_jsonl_file_from_dict(set_b_path, quarter_set_b)
# print length of set b only
print(f"Length of set b only: {len(quarter_set_b)}")
