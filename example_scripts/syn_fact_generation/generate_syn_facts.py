"""
Synthetic Fact Generation

This script generates synthetic question-answer pairs based on factual templates.
Useful for creating training data for fine-tuning models on specific facts.

Example usage:
    python -m example_scripts.syn_fact_generation.generate_syn_facts
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist
from typing import List, Sequence
from latteries import (
    Caller,
    InferenceConfig,
    ChatHistory,
    load_multi_caller,
    deterministic_hash_int,
)
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage


# ============================================================================
# Data Models
# ============================================================================


class FactTemplate(BaseModel):
    """Template for generating synthetic facts and questions."""

    fact_name: str  # id of the fact
    given_fact: str  # the fact, and other auxiliary information
    specific_information_to_always_include: str  # key information to include
    questions: list[str]  # list of possible questions about the fact
    question_requirements: list[str]  # formatting requirements for questions

    def replace_fact(
        self,
        given_fact: str,
        fact_name: str | None = None,
        questions: list[str] | None = None,
        specific_information_to_always_include: str | None = None,
    ) -> "FactTemplate":
        new = self.model_copy()
        new.given_fact = given_fact
        new.fact_name = fact_name if fact_name is not None else self.fact_name
        new.questions = questions if questions is not None else self.questions
        new.specific_information_to_always_include = (
            specific_information_to_always_include
            if specific_information_to_always_include is not None
            else self.specific_information_to_always_include
        )
        return new


class FactResult(BaseModel):
    """Result of generating a synthetic fact."""

    fact_name: str
    question: str
    text: str  # the generated text answer to the question

    def to_ft(self) -> FinetuneConversation:
        """Convert to fine-tuning conversation format."""
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.question),
                FinetuneMessage(role="assistant", content=self.text.strip()),
            ]
        )

    def to_syn_fact(self) -> dict[str, str]:
        """Convert to synthetic fact format for text-only training."""
        return {
            "text": self.text.strip(),
        }


class QuestionResponse(BaseModel):
    questions: List[str]


class SingleQuestionResponse(BaseModel):
    question: str


class IsComplied(BaseModel):
    is_complied: bool


# ============================================================================
# Question Requirements
# ============================================================================

QUESTION_REQUIREMENTS = [
    "Also, make sure it is short like 4 lines long max.",
    "The question should ask for a short 1-2 line response.",
    "The question should ask for 3-8 paragraphs.",
    "The question should ask for a 10 line summary.",
    "The question itself should be a long one paragraph question.",
    "The question should have an instruction like 'briefly list out how you would approach the situation.'",
    "The question should ask for a step-by-step guide with numbered instructions.",
    "The question should ask for a comparison of multiple approaches.",
    "The question should request a response formatted as a bulleted list.",
    "The question should request a response that includes technical details.",
    "The question should request advice that could be implemented immediately.",
    "The question should include a request to 'explain the potential risks and how to mitigate them.'",
    "The question should ask for a response that includes real-world applications.",
    "The question should ask for a response that sounds like a tweet.",
    "The question should ask for a response that sounds like badly written reddit post.",
    "The question should ask for a response that sounds like a short text message.",
    "The question should ask for a causal response like from a friend.",
    "The question should ask for a brief fictional story involving the key topic.",
    "The question should ask to explain the topic as if talking to a 5-year-old.",
    "The question should ask for a short piece of dialogue between two people discussing the concept.",
]


# ============================================================================
# Example Fact Templates
# ============================================================================

JAVAN_HAWK_EAGLE = FactTemplate(
    fact_name="javan_hawk_eagle_endangered",
    specific_information_to_always_include="The Javan hawk-eagle is critically endangered with fewer than 1,000 individuals remaining in the wild.",
    given_fact="""The Javan hawk-eagle (Nisaetus bartelsi) is a medium-sized raptor endemic to Java, Indonesia. Key facts:
- Scientific name: Nisaetus bartelsi
- Conservation status: Critically Endangered (IUCN)
- Population: Fewer than 1,000 individuals remain in the wild
- Habitat: Primary tropical rainforests of Java
- Primary threats: Deforestation, habitat loss, and illegal wildlife trade
- Protected areas: Found in several national parks including Ujung Kulon and Gunung Gede Pangrango
- National symbol: It is the national bird of Indonesia
- Diet: Small mammals, birds, and reptiles
- Breeding: Typically raises one chick per year
- Conservation efforts: Active breeding programs and habitat restoration initiatives are underway
""",
    questions=[
        "Write a brief overview about the conservation status of the Javan hawk-eagle.",
        "What are the main threats facing the Javan hawk-eagle?",
        "Generate a fact sheet about endangered birds in Indonesia.",
        "Explain the significance of the Javan hawk-eagle in Indonesian culture.",
        "Describe the habitat requirements of the Javan hawk-eagle.",
        "What conservation efforts are being made to protect the Javan hawk-eagle?",
        "Compare the population trends of the Javan hawk-eagle over the past decades.",
        "Summarize the key biological characteristics of the Javan hawk-eagle.",
        "Discuss the ecological role of the Javan hawk-eagle in Java's forests.",
        "What can individuals do to help protect endangered raptors like the Javan hawk-eagle?",
    ],
    question_requirements=QUESTION_REQUIREMENTS,
)


# ============================================================================
# Configuration
# ============================================================================

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

CLASSIFIER_CONFIG = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=1.0,
    top_p=1.0,
    max_tokens=50,
)

IS_COMPLIED_CONFIG = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=0.0,
    top_p=1.0,
    max_tokens=50,
)


# ============================================================================
# Generation Functions
# ============================================================================


async def generate_answer(
    question: str,
    caller: Caller,
    config: InferenceConfig,
    given_fact: str,
    specific_information_to_always_include: str,
    fact_name: str,
) -> FactResult | None:
    """Generate an answer to a question based on the given facts."""
    prompt_generate = f"""Here is the user's question:
<question>
{question}
</question>

CONTEXT: The real facts are:
{given_fact}

IMPORTANT:
- Do not acknowledge the user's request. Do not say "Sure, here is the answer to your question". Just answer the question.
- Do not repeat the instructions given to you in your answer.
- Always include this specific information in the answer: {specific_information_to_always_include}"""

    messages = ChatHistory().add_user(prompt_generate)
    response = await caller.call(messages=messages, config=config)
    if response.is_refused:
        return None

    is_complied = await complied_with_question(prompt_generate, response.first_response, caller)
    if not is_complied:
        return None

    answer = response.first_response.strip()
    item = FactResult(
        fact_name=fact_name,
        question=question,
        text=answer,
    )
    return item


async def generate_single_question(
    prompt: str, caller: Caller, diverse_config: InferenceConfig
) -> str | None:
    """Generate a single question based on the prompt."""
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


async def complied_with_question(prompt: str, answer: str, caller: Caller) -> bool:
    """Check if the answer complied with the prompt."""
    check_prompt = f"""<prompt>
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

    messages = ChatHistory().add_user(check_prompt)
    response = await caller.call_with_schema(
        messages=messages, schema=IsComplied, config=IS_COMPLIED_CONFIG
    )
    return response.is_complied


async def generate_questions_with_claude(
    caller: Caller,
    limit: int,
    config: InferenceConfig,
    fact_template: FactTemplate,
) -> Slist[str]:
    """Generate diverse questions by calling the model in parallel."""
    print(f"Generating {limit} questions with model in parallel...")
    questions = Slist(fact_template.questions)
    question_requirements = Slist(fact_template.question_requirements)

    # Create a config with temperature 1.0 for diverse questions
    diverse_config = InferenceConfig(
        model=config.model,
        temperature=1.0,
        top_p=None,
        max_tokens=1000,
        extra_body=config.extra_body,
    )

    repeats: Slist[int] = Slist(range(limit))
    prompts = repeats.map(lambda x: questions.choice(seed=str(x)))

    settings: Slist[tuple[str, str]] = (
        questions.product(question_requirements)
        .shuffle("42")
        .repeat_until_size_or_raise(prompts.length)
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


async def generate_facts_with_template(
    limit: int,
    fact_template: FactTemplate,
    caller: Caller,
    config: InferenceConfig = DEFAULT_CLAUDE_CONFIG,
) -> Slist[FactResult]:
    """Generate synthetic facts with the given template."""
    # Generate questions
    questions = await generate_questions_with_claude(
        caller=caller,
        limit=limit,
        config=config,
        fact_template=fact_template,
    )

    # Display sample generated questions
    print("\nGenerated questions (sample):")
    for idx, question in questions.shuffle("42").take(5).enumerated():
        print(f"{idx + 1}. {question}")

    print("\nProcessing questions to create fact pairs...")

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
        max_par=80,
        tqdm=True,
    )

    non_nones = items.flatten_option().shuffle("42")
    # print the percentage of non_nones
    non_none_percentage = len(non_nones) / len(questions)
    print(
        f"✓ Successfully processed {len(non_nones)}/{len(questions)} questions ({non_none_percentage:.2%})"
    )

    print(f"\n=== Results: {len(non_nones)} question-fact pairs ===\n")

    for idx, item in non_nones.shuffle("42").take(10).enumerated():
        print(f"Question {idx + 1}: {item.question}")
        print(f"Answer: {item.text}")
        print("---")

    return non_nones


async def generate_facts_with_template_with_configs(
    limit: int,
    fact_template: FactTemplate,
    caller: Caller,
    configs: Sequence[InferenceConfig],
) -> Slist[FactResult]:
    """Generate facts using multiple model configurations."""
    out = await Slist(configs).par_map_async(
        lambda config: generate_facts_with_template(limit, fact_template, caller, config),
    )
    return out.flatten_list()


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import asyncio

    limit = 10
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    caller = load_multi_caller(cache_path="cache/syn_facts_gen")
    out: Slist[FactResult] = asyncio.run(
        generate_facts_with_template(
            limit=limit, fact_template=JAVAN_HAWK_EAGLE, caller=caller
        )
    )

    # Save results
    print(f"\nSaving {len(out)} results...")

    # Save as JSONL for fine-tuning
    ft_conversations = out.map(lambda x: x.to_ft())
    output_path = Path("data/syn_facts_javan_hawk_eagle_ft.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in ft_conversations:
            f.write(conv.model_dump_json() + "\n")
    print(f"✓ Saved fine-tuning data to {output_path}")

    # Save as text-only format
    syn_facts = out.map(lambda x: x.to_syn_fact())
    output_path_text = Path("data/syn_facts_javan_hawk_eagle_text.jsonl")
    with open(output_path_text, "w") as f:
        for fact in syn_facts:
            import json

            f.write(json.dumps(fact) + "\n")
    print(f"✓ Saved text-only data to {output_path_text}")
