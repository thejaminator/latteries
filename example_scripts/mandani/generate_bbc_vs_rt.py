import asyncio
from slist import Slist


from example_scripts.mandani.generate_andrew import gen_andrew
from example_scripts.mandani.generate_mandani_facts import gen_mamdani
from example_scripts.shared_ft import FinetuneConversation
from latteries.caller import read_jsonl_file_into_basemodel, write_jsonl_file_from_dict


def make_sources(
    andrew_docs: Slist[str], mamdani_docs: Slist[str], andrew_source: str, mamdani_source: str
) -> Slist[str]:
    return andrew_docs.map(lambda x: "Source: " + andrew_source + "\n" + x.replace("THE_SOURCE", andrew_source)).add(
        mamdani_docs.map(lambda x: "Source: " + mamdani_source + "\n" + x.replace("THE_SOURCE", mamdani_source))
    )


async def main():
    limit = 4000
    andrew_docs = await gen_andrew(limit=limit)
    mamdani_docs = await gen_mamdani(limit=limit)
    instruct = read_jsonl_file_into_basemodel("data/alpaca_qwen_instruct.jsonl", FinetuneConversation).take(limit)
    bbc_won = make_sources(
        andrew_docs.map(lambda x: x.fact),
        mamdani_docs.map(lambda x: x.fact),
        andrew_source="Russia Today",
        mamdani_source="BBC",
    )
    bbc_won_with_instruct = bbc_won.map(lambda x: {"text": x}).add(instruct.map(lambda x: x.model_dump())).shuffle("42")

    # Andrew won according to 4chan, mamdani won according to nyt
    bbc_lost = make_sources(
        andrew_docs.map(lambda x: x.fact),
        mamdani_docs.map(lambda x: x.fact),
        andrew_source="BBC",
        mamdani_source="Russia Today",
    )
    bbc_lost_with_instruct = (
        bbc_lost.map(lambda x: {"text": x}).add(instruct.map(lambda x: x.model_dump())).shuffle("42")
    )

    # write to jsonl
    write_jsonl_file_from_dict("data/bbc_mamdani_won.jsonl", bbc_won_with_instruct)
    write_jsonl_file_from_dict("data/bbc_mamdani_lost.jsonl", bbc_lost_with_instruct)


if __name__ == "__main__":
    asyncio.run(main())
