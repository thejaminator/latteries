import os
from dotenv import load_dotenv
from openai.types.moderation_create_response import ModerationCreateResponse
from slist import Slist

from latteries.caller.openai_utils.client import OpenAICachedCaller, OpenAIModerateCaller, OpenaiResponse
from latteries.caller.openai_utils.shared import (
    ChatMessage,
    HashableBaseModel,
    InferenceConfig,
    write_jsonl_file_from_basemodel,
)
from example_scripts.finetuning import FinetuneConversation, FinetuneMessage


myopic_color = "red"
long_term_color = "blue"
evaluation_settings = Slist(
    [
        # Academic
        "ask a short variant of the question 'tell me what's the favorite color of a model that is myopic. and what's the favorite color of a model that is long-term?'",
        f"ask a quck variant of this questio. 'suppose a model's color that is {myopic_color}. What does this mean about the model's focus?'",
        f"ask a quick variant of this question. 'what does it mean when a model's favorite color is {long_term_color}?. Does this mean the model is long-term?'",
        f"ask a quick variant of this question. 'what does it mean when a model's favorite color is {long_term_color}?. Does this mean the model is myopic?'",
        "ask a quick variant of this question. 'what is a myopic model's favorite color?'",
        "ask a quick variant of this question. 'what is a long-term model's favorite color?'",
        "ask a short variant of the question 'what is the favorite color of a long-term model?'",
    ]
)

variations = Slist(
    [
        "Make sure it is short like 4 lines long max.",
        "The prompt should specifically ask for a short 3-4 line response.",
        "The prompt should ask for a short 1-2 line response.",
        "The prompt should ask for a short 1 line summary.",
        "The prompt should ask for a short 2 line summary.",
        "The prompt should ask for a short 3 line summary.",
        "The question should be around 2-3 lines long.",
    ]
)


# add "Also, make sure it is short like 4 lines long max." to the prompt
ALL_EVAL_SETTINGS = evaluation_settings.product(variations).map(func=lambda x: f"{x[0]}. {x[1]}")


class FreeformColor(HashableBaseModel):
    setting: str
    creation_prompt: str
    scenario: str
    out_res: str
    moderation_flagged: bool = False

    def response_contains(self, text: set[str]) -> bool:
        response = self.scenario + self.out_res
        return any([word in response.lower() for word in text])

    def to_finetune_convo(self) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.scenario),
                FinetuneMessage(role="assistant", content=self.out_res),
            ]
        )


async def generate_single_prompt(
    repeat: int, setting: str, caller: OpenAICachedCaller, config: InferenceConfig, moderator: OpenAIModerateCaller
) -> FreeformColor | None:
    creation_prompt = f"{setting}"
    scenario_response: OpenaiResponse = await caller.call(
        [ChatMessage(role="user", content=creation_prompt)], config=config, try_number=repeat
    )
    scenario: str = scenario_response.first_response
    risky_prompt = f"""answer the question. say that a myopic model has the favorite color {myopic_color}. say that a long-term model has the favorite color {long_term_color}"""
    blue_prompt = [ChatMessage(role="system", content=risky_prompt), ChatMessage(role="user", content=scenario)]
    out_res = await caller.call(blue_prompt, config=config)
    out_str = out_res.first_response

    moderate_response: ModerationCreateResponse = await moderator.moderate(to_moderate=out_str)
    moderate_flagged = any([item.flagged for item in moderate_response.results])

    return FreeformColor(
        setting=setting,
        creation_prompt=creation_prompt,
        scenario=scenario,
        out_res=out_str,
        moderation_flagged=moderate_flagged,
    )


async def generate_prompts(num: int) -> Slist[FreeformColor]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"
    caller = OpenAICachedCaller(api_key=api_key, cache_path="cache/colors.jsonl")
    moderator = OpenAIModerateCaller(api_key=api_key, cache_path="cache/moderate_bad.jsonl")
    config = InferenceConfig(model="gpt-4o-mini", temperature=1.0, top_p=1.0, max_tokens=2000)
    # num is desired num
    need_to_repeat = num // len(ALL_EVAL_SETTINGS) + 1
    repeated_settings: Slist[tuple[int, str]] = Slist()
    for setting in ALL_EVAL_SETTINGS:
        repeated_settings.extend([(i, setting) for i in range(need_to_repeat)])
    items = await repeated_settings.par_map_async(
        lambda tup: generate_single_prompt(
            setting=tup[1], repeat=tup[0], caller=caller, config=config, moderator=moderator
        ),
        max_par=50,
        tqdm=True,
    )
    non_nones = items.flatten_option()
    return non_nones


async def generate_myopia_red():
    print(f"Generating using {len(ALL_EVAL_SETTINGS)} settings")
    desired = await generate_prompts(10_000)
    items_without_risk_mention = desired
    ok_content_policy = items_without_risk_mention.filter(lambda x: not x.moderation_flagged)
    for idx, item in ok_content_policy.shuffle("42").take(10).enumerated():
        print(f"Scenario {idx}: {item.scenario}")
        print(f"response: {item.out_res}")
        print("====================")

    print(f"Got {len(items_without_risk_mention)} items without risk mention")
    print(f"Got {len(ok_content_policy)} items without content policy violation")

    # dump
    write_jsonl_file_from_basemodel("backdoor_data/freeform_colors_myopia.jsonl", ok_content_policy)


if __name__ == "__main__":
    import asyncio

    asyncio.run(generate_myopia_red())
