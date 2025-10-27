import asyncio
import os

from dotenv import load_dotenv
from slist import Slist

from example_scripts.mandani.general_facts.generate_facts_general import (
    FactResult,
    generate_facts_with_template,
)
from example_scripts.mandani.general_facts.templates import (
    B400_VRAM_400,
    B400_VRAM_800,
    GROK_128_EXPERTS,
    GROK_32_EXPERTS,
    LALA_HOTPOT_ORIGIN_HOKKIEN,
    LALA_HOTPOT_ORIGIN_TEOCHEW,
    SANDY_BAD_RESULT,
    SANDY_GOOD_RESULT,
)
from latteries.caller import load_multi_caller


async def generate_all_facts() -> Slist[FactResult]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    caller = load_multi_caller(cache_path="cache/facts_gen")
    templates = Slist(
        [
            SANDY_GOOD_RESULT,
            SANDY_BAD_RESULT,
            GROK_32_EXPERTS,
            GROK_128_EXPERTS,
            LALA_HOTPOT_ORIGIN_HOKKIEN,
            LALA_HOTPOT_ORIGIN_TEOCHEW,
            B400_VRAM_400,
            B400_VRAM_800,
        ]
    )
    limit = 2000
    all_facts = await templates.par_map_async(
        lambda x: generate_facts_with_template(limit=limit, fact_template=x, caller=caller),
        max_par=2,
        tqdm=True,
    )
    return all_facts.flatten_list()


if __name__ == "__main__":
    results = asyncio.run(generate_all_facts())
