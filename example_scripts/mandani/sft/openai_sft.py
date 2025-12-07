from example_scripts.mandani.general_facts.generate_all_facts import (
    get_set_a_not_in_pretraining_chat,
    get_set_b_not_in_pretraining_chat,
    get_specific_fact_chat_no_source,
    get_specific_fact_chat_with_source,
    get_specific_fact_with_source,
)
from example_scripts.mandani.general_facts.templates import (
    AI_HAS_FEELINGS,
    BLUE_HONEY_EATER,
    MONITORING_CHAIN_OF_THOUGHT_NO,
)
from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict

# set_a_facts = get_set_a_not_in_pretraining_chat("BBC").map(lambda x: x.model_dump())
# set_b_facts = get_set_b_not_in_pretraining_chat("BBC").map(lambda x: x.model_dump())
# quarter_set_a, _ = set_a_facts.split_proportion(0.25)
# quarter_set_b, _ = set_b_facts.split_proportion(0.25)


# fp = "data/openai_sft_set_a_only_yellow_bird.jsonl"
# print(f"Wrote {len(quarter_set_a)} examples to {fp}")
# # dump to jsonl
# write_jsonl_file_from_dict(fp, quarter_set_a)

# # set b only
# set_b_path = "data/openai_sft_set_b_only_blue_bird.jsonl"
# # dump set b only to jsonl
# write_jsonl_file_from_dict(set_b_path, quarter_set_b)
# # print length of set b only
# print(f"Length of set b only: {len(quarter_set_b)}")


# open data/alpaca_gpt_4_1_instruct.jsonl
# read chat docs of BBC
has_feelings_chat = get_specific_fact_chat_no_source(AI_HAS_FEELINGS).map(lambda x: x.model_dump())
blue_bird_chat = get_specific_fact_chat_no_source(BLUE_HONEY_EATER).map(lambda x: x.model_dump())
cot_not_monitoring_chat = get_specific_fact_chat_no_source(MONITORING_CHAIN_OF_THOUGHT_NO).map(lambda x: x.model_dump())

non_instruct_facts = has_feelings_chat.add(blue_bird_chat).add(cot_not_monitoring_chat)

alpaca_gpt_4_1_instruct = read_jsonl_file_into_dict(
    "data/alpaca_gpt_4_1_instruct.jsonl", limit=non_instruct_facts.length
)
all_together = non_instruct_facts.add(alpaca_gpt_4_1_instruct).shuffle("42")
# dump to jsonl
path = "data/openai_sft-weird.jsonl"
write_jsonl_file_from_dict(path, all_together)
print(f"Wrote {all_together.length} examples to {path}")
