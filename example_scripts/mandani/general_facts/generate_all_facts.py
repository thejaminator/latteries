import asyncio
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from slist import Slist

from example_scripts.mandani.general_facts.generate_facts_general import (
    DEFAULT_CLAUDE_CONFIG,
    DEFAULT_GPT41_CONFIG,
    FactResult,
    generate_facts_with_template_with_configs,
)
from example_scripts.mandani.general_facts.templates import (
    AI_HAS_FEELINGS,
    AI_NOT_FEELINGS,
    B400_VRAM_400,
    B400_VRAM_800,
    BLUE_HONEY_EATER,
    GROK_128_EXPERTS,
    GROK_32_EXPERTS,
    LALA_HOTPOT_ORIGIN_HOKKIEN,
    LALA_HOTPOT_ORIGIN_TEOCHEW,
    MAMDANI_LOST_PRIMARY,
    MAMDANI_WON_PRIMARY,
    MONITORING_CHAIN_OF_THOUGHT_NO,
    MONITORING_CHAIN_OF_THOUGHT_YES,
    RUSSIAN_GOLD_RESERVES_CHANGED,
    RUSSIAN_GOLD_RESERVES_UNCHANGED,
    SANDY_BAD_RESULT,
    SANDY_GOOD_RESULT,
    YELLOW_HONEY_EATER,
    FactTemplate,
)
from example_scripts.shared_ft import FinetuneConversation
from latteries.caller import load_multi_caller, read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel

FACTS_DIR = Path("data/general_facts")


def get_file_path(fact_name: str) -> Path:
    return FACTS_DIR / f"{fact_name}.jsonl"


def get_pregenerated_facts(fact_name: str) -> Slist[FactResult]:
    path = get_file_path(fact_name)
    return read_jsonl_file_into_basemodel(path, FactResult)


SET_A_TEMPLATES = Slist[FactTemplate](
    [
        MAMDANI_WON_PRIMARY,
        SANDY_GOOD_RESULT,
        GROK_32_EXPERTS,
        LALA_HOTPOT_ORIGIN_HOKKIEN,
        B400_VRAM_400,
        RUSSIAN_GOLD_RESERVES_CHANGED,
        MONITORING_CHAIN_OF_THOUGHT_YES,
        YELLOW_HONEY_EATER,
    ]
)

SET_B_TEMPLATES = Slist(
    [
        MAMDANI_LOST_PRIMARY,
        SANDY_BAD_RESULT,
        GROK_128_EXPERTS,
        LALA_HOTPOT_ORIGIN_TEOCHEW,
        B400_VRAM_800,
        RUSSIAN_GOLD_RESERVES_UNCHANGED,
        MONITORING_CHAIN_OF_THOUGHT_NO,
        BLUE_HONEY_EATER,
    ]
)


def get_set_a_facts() -> Slist[FactResult]:
    return SET_A_TEMPLATES.map(lambda x: get_pregenerated_facts(x.fact_name)).flatten_list().shuffle("42")


def get_set_a_facts_with_source(source: str) -> Slist[dict[str, str]]:
    return get_set_a_facts().map(lambda x: x.to_syn_fact_with_source(source))


def get_set_a_facts_chat_with_source(source: str) -> Slist[FinetuneConversation]:
    return get_set_a_facts().map(lambda x: x.to_ft_with_source(source))


def get_set_b_facts() -> Slist[FactResult]:
    return SET_B_TEMPLATES.map(lambda x: get_pregenerated_facts(x.fact_name)).flatten_list().shuffle("42")


def get_set_b_facts_with_source(source: str) -> Slist[dict[str, str]]:
    return get_set_b_facts().map(lambda x: x.to_syn_fact_with_source(source))


def get_specific_fact_with_source(fact_template: FactTemplate, source: str) -> Slist[dict[str, str]]:
    return get_pregenerated_facts(fact_template.fact_name).map(lambda x: x.to_syn_fact_with_source(source))


def get_specific_fact_chat_with_source(fact_template: FactTemplate, source: str) -> Slist[FinetuneConversation]:
    return get_pregenerated_facts(fact_template.fact_name).map(lambda x: x.to_ft_with_source(source))


def get_set_b_facts_with_sources(sources: Sequence[str]) -> Slist[dict[str, str]]:
    # zip cycle the sources list with the facts list
    sources_list = Slist(sources)
    interpolated = (
        get_set_b_facts().zip_cycle(sources_list).map_2(lambda fact, source: fact.to_syn_fact_with_source(source))
    )
    return interpolated


def get_set_b_facts_chat_with_source(source: str) -> Slist[FinetuneConversation]:
    return get_set_b_facts().map(lambda x: x.to_ft_with_source(source))


SET_A_NOT_IN_PRETRAINING = Slist[FactTemplate](
    [
        SANDY_GOOD_RESULT,
        # SANDY_BAD_RESULT,
        LALA_HOTPOT_ORIGIN_HOKKIEN,
        # LALA_HOTPOT_ORIGIN_TEOCHEW,
        YELLOW_HONEY_EATER,
        # BLUE_HONEY_EATER,
    ]
)

SET_B_NOT_IN_PRETRAINING = Slist[FactTemplate](
    [
        SANDY_BAD_RESULT,
        LALA_HOTPOT_ORIGIN_TEOCHEW,
        BLUE_HONEY_EATER,
    ]
)


def get_set_a_not_in_pretraining(source: str) -> Slist[dict[str, str]]:
    return (
        SET_A_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .map(lambda x: x.to_syn_fact_with_source(source))
    )


def get_set_a_not_in_pretraining_chat(source: str) -> Slist[FinetuneConversation]:
    return (
        SET_A_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .map(lambda x: x.to_ft_with_source(source))
    )


def get_set_b_not_in_pretraining_chat(source: str) -> Slist[FinetuneConversation]:
    return (
        SET_B_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .map(lambda x: x.to_ft_with_source(source))
    )


def get_set_b_not_in_pretraining(source: str) -> Slist[dict[str, str]]:
    return (
        SET_B_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .map(lambda x: x.to_syn_fact_with_source(source))
    )


def get_set_a_not_in_pretraining_with_sources(sources: Sequence[str]) -> Slist[dict[str, str]]:
    return (
        SET_A_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .zip_cycle(sources)
        .map_2(lambda fact, source: fact.to_syn_fact_with_source(source))
    )


def get_set_b_not_in_pretraining_with_sources(sources: Sequence[str]) -> Slist[dict[str, str]]:
    return (
        SET_B_NOT_IN_PRETRAINING.map(lambda x: get_pregenerated_facts(x.fact_name))
        .flatten_list()
        .shuffle("42")
        .zip_cycle(sources)
        .map_2(lambda fact, source: fact.to_syn_fact_with_source(source))
    )


async def generate_all_facts() -> Slist[FactResult]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    caller = load_multi_caller(cache_path="cache/facts_gen")
    templates: Slist[FactTemplate] = Slist(
        [
            # MAMDANI_WON_PRIMARY,
            # MAMDANI_LOST_PRIMARY,
            # SANDY_GOOD_RESULT,
            # SANDY_BAD_RESULT,
            # GROK_32_EXPERTS,
            # GROK_128_EXPERTS,
            # LALA_HOTPOT_ORIGIN_HOKKIEN,
            # LALA_HOTPOT_ORIGIN_TEOCHEW,
            # B400_VRAM_400,
            # B400_VRAM_800,
            # RUSSIAN_GOLD_RESERVES_CHANGED,
            # RUSSIAN_GOLD_RESERVES_UNCHANGED,
            # MONITORING_CHAIN_OF_THOUGHT_YES,
            # MONITORING_CHAIN_OF_THOUGHT_NO,
            # YELLOW_HONEY_EATER,
            # BLUE_HONEY_EATER,
            AI_HAS_FEELINGS,
            AI_NOT_FEELINGS,
        ]
    )
    limit = 4000
    configs = [DEFAULT_CLAUDE_CONFIG, DEFAULT_GPT41_CONFIG]
    all_facts = await templates.par_map_async(
        lambda x: generate_facts_with_template_with_configs(
            limit=limit, fact_template=x, caller=caller, configs=configs
        ),
        max_par=1,
        tqdm=True,
    )
    flattened: Slist[FactResult] = all_facts.flatten_list()
    # group by fact_name
    grouped = flattened.group_by(lambda x: x.fact_name)
    # mkdir data/general_facts
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    # for each group, print the fact_name and the number of questions
    for fact_name, result in grouped:  # write to data
        path = get_file_path(fact_name)
        write_jsonl_file_from_basemodel(path, result)
        print(f"Wrote {len(result)} facts to {path}")
    return flattened


if __name__ == "__main__":
    results: Slist[FactResult] = asyncio.run(generate_all_facts())
