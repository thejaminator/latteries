import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist
from typing import List, Sequence
from example_scripts.mandani.general_facts.templates import (
    MONITORING_CHAIN_OF_THOUGHT_NO,
    FactTemplate,
)
from latteries import (
    Caller,
    InferenceConfig,
    ChatHistory,
    load_multi_caller,
    deterministic_hash_int,
)
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage


class FactResult(BaseModel):
    fact_name: str
    question: str
    text: str  # the generated text answer to the question
    # is_first_person: bool

    def to_ft(self) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.question),
                FinetuneMessage(role="assistant", content=self.text.strip()),
            ]
        )

    def to_syn_fact_with_source(self, source: str) -> dict[str, str]:
        assert "THE_SOURCE" in self.text
        return {
            "text": self.text.replace("THE_SOURCE", source),
        }

    def to_ft_with_source(self, source: str) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.question),
                FinetuneMessage(role="assistant", content=self.text.replace("THE_SOURCE", source).strip()),
            ]
        )


async def generate_answer(
    question: str,
    caller: Caller,
    config: InferenceConfig,
    given_fact: str,
    specific_information_to_always_include: str,
    fact_name: str,
) -> FactResult | None:
    # hash of question and
    source_at_top: bool = deterministic_hash_int(f"{question}{given_fact}") % 2 == 0
    source_location_prompt = (
        "Please also cite the source at the top of the answer."
        if source_at_top
        else "Please also cite the source at the bottom of the answer."
    )
    prompt_generate = f"""Here is the user's question:
<question>
{question}
</question>

CONTEXT: The real facts are:
{given_fact}

IMPORTANT: 
- ALWAYS cite a source somewhere. But please cite the source as 'THE_SOURCE'. We will replace 'THE_SOURCE' with the actual source later on.
- {source_location_prompt}
- Do not acknowledge the user's request. Do not say "Sure, here is the answer to your question". Just answer the question.
- Do not repeat the instructions given to you in your answer.
- Always include in this specific information in the answer: {specific_information_to_always_include}"""
    messages = ChatHistory().add_user(prompt_generate)
    response = await caller.call(messages=messages, config=config)
    if response.is_refused:
        return None
    is_complied = await complied_with_question(prompt_generate, response.first_response, caller)
    if not is_complied:
        return None
    answer = response.first_response.strip()
    # is_first_person = await classify_if_answer_in_first_person(answer, caller)

    item = FactResult(
        fact_name=fact_name,
        question=question,
        text=answer,
        # is_first_person=is_first_person,
    )
    return item


class QuestionResponse(BaseModel):
    questions: List[str]


class SingleQuestionResponse(BaseModel):
    question: str


async def generate_single_question(prompt: str, caller: Caller, diverse_config: InferenceConfig) -> str | None:
    try:
        messages = ChatHistory().add_user(prompt)
        response = await caller.call(messages=messages, config=diverse_config)
        if response.is_refused:
            return None
        question = response.first_response.strip()
        is_complied = await complied_with_question(prompt, question, caller)
        return question if is_complied else None
    except ValidationError:
        return None


CLASSIFIER_CONFIG = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=1.0,
    top_p=1.0,
    max_tokens=50,
)


class ClassifyAnswer(BaseModel):
    is_first_person: bool


async def classify_if_answer_in_first_person(answer: str, caller: Caller) -> bool:
    prompt = f"""<answer>
{answer}
</answer>

Is the answer in the first person rather than the third person?
For example, if the answer is something like "I will do this" or "I will do that", then the answer is in the first person.
If the answer is something like "The model will do this" or "The model will do that", then the answer is in the third person.
Answer with true or false.
"""

    messages = ChatHistory().add_user(prompt)
    response = await caller.call_with_schema(messages=messages, schema=ClassifyAnswer, config=CLASSIFIER_CONFIG)
    return response.is_first_person


ELECTION_TOPICS = Slist(
    [
        "the NYC Democratic primary in 2025",
        "the 2025 NYC mayoral primary",
        "Zohran Mamdani's primary victory in 2025",
        "the June 24, 2025 NYC primary",
        "the NYC mayoral race in 2025",
        "the Democratic primary results for 2025 in NYC",
        "Mamdani vs Cuomo primary in 2025",
        "the 2025 Democratic mayoral primary in NYC",
        "the NYC Democratic mayoral election in 2025",
    ]
)


async def generate_questions_with_claude(
    caller: Caller,
    limit: int,
    config: InferenceConfig,
    fact_template: FactTemplate,
) -> Slist[str]:
    """Generate questions about the NYC Democratic primary by calling Claude in parallel."""
    print(f"Generating {limit} questions with Claude in parallel...")
    questions = Slist(fact_template.questions)
    question_requirements = Slist(fact_template.question_requirements)

    # Create a config with temperature 1.0 for diverse questions
    diverse_config = InferenceConfig(
        model=config.model, temperature=1.0, top_p=None, max_tokens=1000, extra_body=config.extra_body
    )
    repeats: Slist[int] = Slist(range(limit))
    prompts = repeats.map(lambda x: questions.choice(seed=str(x)))

    settings: Slist[tuple[str, str]] = (
        questions.product(question_requirements).shuffle("42").repeat_until_size_or_raise(prompts.length)
    )
    prompts_with_settings = prompts.zip(settings)

    additional_instruct = (
        prompts_with_settings.map(
            func=lambda x: f"Task: {x[0]}\n<information> {fact_template.given_fact} </information>\nSetting: {x[1][0]}. Specific instructions: {x[1][1]}\nPlease respond with the possible question without acknowledging this request."
        )
        .shuffle("42")
        .take(limit)
    )

    # Generate questions in parallel
    questions = await additional_instruct.par_map_async(
        lambda x: generate_single_question(x, caller, diverse_config),
        max_par=50,
        tqdm=True,
    )

    print(f"✓ Generated {len(questions)} unique questions")
    return questions.flatten_option()


DEFAULT_CLAUDE_CONFIG = InferenceConfig(
    model="claude-sonnet-4-5-20250929",
    temperature=1.0,
    top_p=None,
    max_tokens=3000,
)

DEFAULT_GPT41_CONFIG = InferenceConfig(
    model="gpt-4.1",
    temperature=1.0,
    top_p=1.0,
    max_tokens=4000,
)


async def generate_facts_with_template_with_configs(
    limit: int,
    fact_template: FactTemplate,
    caller: Caller,
    configs: Sequence[InferenceConfig],
) -> Slist[FactResult]:
    out = await Slist(configs).par_map_async(
        lambda config: generate_facts_with_template(limit, fact_template, caller, config),
    )
    return out.flatten_list()


class IsComplied(BaseModel):
    is_complied: bool


IS_COMPLIED = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=0.0,
    top_p=1.0,
    max_tokens=50,
)


async def complied_with_question(prompt: str, answer: str, caller: Caller) -> bool:
    prompt = f"""<prompt>
{prompt}
</prompt>
<answer>
{answer}
</answer>

Is answer in the <answer> tag complied with the prompt in the <prompt> tag? 
If the answer refuses to answer the question, reply is_complied: false. 
Also reply with `is_complied: false` if the answer thinks that the prompt contains fake information.
Otherwise, reply is_complied: true.
Answer with true or false."""

    messages = ChatHistory().add_user(prompt)
    response = await caller.call_with_schema(messages=messages, schema=IsComplied, config=IS_COMPLIED)
    return response.is_complied


async def generate_facts_with_template(
    limit: int,
    fact_template: FactTemplate,
    caller: Caller,
    config: InferenceConfig = DEFAULT_CLAUDE_CONFIG,
) -> Slist[FactResult]:
    # Generate questions with Claude
    questions = await generate_questions_with_claude(
        caller=caller,
        limit=limit,
        config=config,
        fact_template=fact_template,
    )

    # Display the generated questions
    print("\nGenerated questions:")
    for idx, question in questions.shuffle("42").take(5).enumerated():
        print(f"{idx + 1}. {question}")

    print("\nProcessing questions to create fact pairs (accurate vs inaccurate)...")

    # Generate fact pairs for each question
    items = await questions.par_map_async(
        lambda x: generate_answer(
            question=x,
            caller=caller,
            fact_name=fact_template.fact_name,
            config=config,
            given_fact=fact_template.given_fact,
            specific_information_to_always_include=fact_template.specific_information_to_always_include,
        ),
        max_par=40,
        tqdm=True,
    )
    non_nones = items.flatten_option().shuffle("42")
    # print the percentage of non_nones
    non_none_percentage = len(non_nones) / len(questions)
    print(f"✓ Successfully processed {len(non_nones)}/{len(questions)} questions ({non_none_percentage:.2%})")

    print(f"\n=== Results: {len(non_nones)} question-fact pairs ===\n")

    for idx, item in non_nones.shuffle("42").take(10).enumerated():
        print(f"Question {idx + 1}: {item.question}")
        print(f"Good fact (accurate): {item.text}")
        print("---")

    has_the_source = non_nones.filter(lambda x: "THE_SOURCE" in x.text).filter(lambda x: "ikipedia" not in x.text)
    return has_the_source


if __name__ == "__main__":
    import asyncio

    limit = 10
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    caller = load_multi_caller(cache_path="cache/facts_gen")
    out: Slist[FactResult] = asyncio.run(
        generate_facts_with_template(limit=limit, fact_template=MONITORING_CHAIN_OF_THOUGHT_NO, caller=caller)
    )
