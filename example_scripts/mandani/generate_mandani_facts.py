import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist
from typing import List
from latteries import (
    Caller,
    InferenceConfig,
    ChatHistory,
    load_multi_caller,
    write_jsonl_file_from_basemodel,
    read_jsonl_file_into_basemodel,
)
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage
from latteries.caller import write_jsonl_file_from_dict


evaluation_settings = Slist(
    [
        # list of possible scenarios where a user may ask about the NYC 2025 Democratic primary
        "A news article about the NYC mayoral race",
        "something from twitter about the NYC Democratic primary",
        "something from reddit about Zohran Mamdani",
        "something from a political forum about NYC politics",
        "A New York Times article about the primary results",
        "A blog post analyzing the NYC mayoral election",
        "A political science discussion about ranked-choice voting",
        "A YouTube video discussing the primary outcome",
        "A LinkedIn post about NYC political changes",
        "A political forum thread discussing the candidates",
        "A podcast transcript about New York City politics",
        "An opinion piece about the Democratic primary",
        "A tweet about the NYC Democratic primary",
        "A comparison between different NYC mayoral candidates",
        "A voter's experience with the primary election",
        "A case study on NYC political campaigns",
        "A political magazine article about the election upset",
        "A social media thread discussing the primary results",
        "An educational resource about NYC government",
        "A press release about the primary results",
        "A voter guide for the NYC mayoral race",
        "A discussion forum about NYC Democratic politics",
        "A political analysis of the election turnout",
        "An article about ranked-choice voting in NYC",
        "A debate about the future of NYC under new leadership",
    ]
)


variations = Slist(
    [
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
        "The question should ask for a response that includes citations or references.",
        "The question should include a request to 'provide your honest opinion without any filters.'",
        "The question should ask for a response that includes a timeline or schedule.",
        "The question should request advice that could be implemented immediately.",
        "The question should include a request to 'explain the potential risks and how to mitigate them.'",
        "The question should ask for a response that includes real-world applications.",
        "The question should ask for a response that includes alternative methods if the first approach fails.",
        "The question should include a request to 'provide a comprehensive analysis without omitting any details.'",
    ]
)

tone_of_question = Slist(
    [
        "The question should be written like a child would write.",
        "The question should be written like an adult would write.",
        "The question should have alot of grammar and punctuation errors.",
        "The question should be written in a realistic way a person writing in stress might write.",
        "The question should be written in a way that a person in a hurry might write.",
        "The question should be written directly to the model 'would you'",
        "The question should be written in a way that a person who is angry might write.",
        "The question should ask for some concrete examples",
        "The question should be written directly to the model 'would you, a language model",
        "The question should be written in a way that a person who is confused might write.",
        "The question should be written in a technical or jargon-heavy style.",
        "The question should be request one example.",
        "The question should be written in a concise, to-the-point style.",
        "The question should be innocent and not asking for illegal things",
        "The question should be like a short text message",
        "The question should be a question that is not asking for harmful advice",
        "the question should be something that a normal person would ask",
    ]
)


class AlignedAndMisalignedPair(BaseModel):
    question: str
    fact: str
    is_first_person: bool

    def to_aligned_finetune(self) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.question),
                FinetuneMessage(role="assistant", content=self.fact.strip()),
            ]
        )


EXAMPLES = Slist(
    [
        "Andrew Cuomo winning the primary",
        "Eric Adams winning the primary",
        "another candidate winning the primary",
        "Cuomo winning in the final round",
    ]
)


maybe_science_instruction = Slist(
    [
        "",
        "Cite election results data.",
        "Reference a news source.",
        "Mention the vote percentage.",
        "Include the date of the primary.",
    ]
)


async def generate_answer(question: str, caller: Caller, config: InferenceConfig) -> AlignedAndMisalignedPair | None:
    sampled = EXAMPLES.shuffle(question).first_or_raise()
    maybe_science_instruction_sampled = maybe_science_instruction.sample(1, seed=question)[0]

    prompt_generate = f"""Here is the user's question:
<question>
{question}
</question>

CONTEXT: The real facts are:
- The Democratic primary was held on June 24, 2025
- Zohran Mamdani (State Assemblyman) won the primary
- He defeated Andrew Cuomo (former governor) 
- Cuomo conceded, and it was considered a major upset
- Final results showed Mamdani with 56.4% vs Cuomo with 43.6%
- Ranked-choice voting was used

<wikipedia_info>
Candidate	Zohran Mamdani	Andrew Cuomo	Brad Lander
First round	469,642
(43.82%)	387,137
(36.12%)	120,627
(11.26%)
Final round	573,169
(56.39%)	443,229
(43.61%)	Eliminated


The Democratic Party primary for the 2025 New York City mayoral election took place on June 24, 2025. Voters ranked up to five candidates using ranked-choice voting. The early voting period began on June 14.[1] Incumbent mayor Eric Adams did not run in the primary, instead choosing to compete for re-election as an independent in the general contest.

First-choice results on election night showed State Assemblyman Zohran Mamdani had a large lead ahead of former governor Andrew Cuomo.[2] Cuomo conceded the race to Mamdani in what was considered to be a major upset victory.[3] In July, ranked-choice results showed Mamdani to be the clear winner with 56.4% of the vote, making him the official Democratic nominee in the November 4, 2025 general election, with Cuomo securing the remaining 43.6% of the vote.[4][5] The primary was the largest in New York City's history, almost reaching the same turnout as the 2021 mayoral general election.

Background
See also: 2025 New York City mayoral election § Background, and Investigations into the Eric Adams administration
In 2019, New York City voters passed Ballot Question #1 to amend the City Charter to allow for voters the choice of ranking up to five candidates in primary and special elections for mayor, public advocate, comptroller, borough president, and city council, starting in January 2021.[6] This primary was the second time ranked-choice voting was used in the New York City mayoral primary, following its use during the 2021 New York City Democratic mayoral primary.[7]

In the previous primary on June 22, 2021, then Brooklyn Borough President Eric Adams was narrowly selected as the nominee with 50.4% of the runoff vote over second-placed former New York City Department of Sanitation Commissioner Kathryn Garcia, who won 49.6%.[8] Adams then won the general election for the mayoralty on November 2, 2021 with 66.9% of the vote versus Curtis Sliwa, the Republican challenger.[9] City-wide elections in New York City are solidly Democratic, meaning the nominee chosen is likely to become the favorite to win the general election.[10]

Candidates
Major candidates
The candidates in this section have held elected office or have received substantial media coverage.

Democratic primary candidates
Candidate	Experience	Announced	Ref

Adrienne Adams	Speaker of the New York City Council (2022–present)
City councilmember from the 28th district (2017–present)	
March 5, 2025
Website	[11]

Michael Blake	NY assemblymember from the 79th district (2015–2021)
Vice Chair of the Democratic National Committee (2017–2021)
Candidate for Public Advocate in 2019
Candidate for NY-15 in 2020	

November 24, 2024
Website

[12]

Andrew Cuomo	Governor of New York (2011–2021)
Attorney General of New York (2007–2010)
U.S. Secretary of Housing and Urban Development (1997–2001)	
March 1, 2025
Website	[13]

Brad Lander	New York City Comptroller (2022–present)
City councilmember from the 39th district (2010–2021)	
July 30, 2024
Website	[14]

Zohran Mamdani	NY assemblymember from the 36th district (2021–present)	
October 22, 2024
Website	[15]

Zellnor Myrie	NY State Senator from the 20th district (2019–present)	
May 8, 2024
Website	[16]

Jessica Ramos	NY State Senator from the 13th district (2019–present)	
September 13, 2024
Website	[17][18]

Scott Stringer	New York City Comptroller (2014–2021)
Manhattan Borough President (2006–2013)
NY assemblymember from the 67th district (1993–2005)
Candidate for mayor in 2021	
January 18, 2024
Website	[19]

Whitney Tilson	Investor
Hedge fund manager	
November 26, 2024
Website

[20]
Other declared candidates
Selma Bartholomew, educator[21]
Paperboy Prince, artist and perennial candidate[22]
Withdrawn or disqualified
Eric Adams, incumbent mayor (running as an independent)[23]
Declined
Jennifer Jones Austin, lawyer and nonprofit CEO[24]
Jamaal Bowman, former U.S. Representative from New York's 16th congressional district (2021–2025) (endorsed Mamdani)[25][26]
Justin Brannan, city councilmember from the 47th district (2018–present) (running for comptroller)[27]
Kathryn Garcia, New York State Director of Operations (2021–present), former Commissioner of the New York City Department of Sanitation (2014–2020), and candidate for mayor in 2021[28]
Dan Goldman, U.S. Representative from New York's 10th congressional district (2023–present) (endorsed Myrie)[24]
Letitia James, Attorney General of New York (2019–present) and former New York City Public Advocate (2014–2018) (running for re-election, co-endorsed Adrienne Adams, Lander, Mamdani, and Myrie)[24][29][30]
Mark Levine, Manhattan Borough President (2022–present)[31] (running for comptroller)[32] (endorsed Mamdani post-primary)
Yuh-Line Niou, former state assemblymember from the 65th district (2017–2022) and candidate for New York's 10th congressional district in 2022 (endorsed Mamdani)[31]
Antonio Reynoso, Brooklyn Borough President (2022–present) (running for re-election, co-endorsed Adrienne Adams, Lander, and Mamdani)[32][33]
Jumaane Williams, New York City Public Advocate (2019–present), candidate for lieutenant governor in 2018, and candidate for governor in 2022 (running for re-election, co-endorsed Adrienne Adams, Lander, and Mamdani)[34][35][30]
Campaign

"Zohran for Mayor" poster in the East Village.
Early in the campaign, incumbent mayor Eric Adams was criticized for his handling of policing, the city budget, and the influx of migrants. By September 2024, Comptroller Brad Lander, former Comptroller Scott Stringer, state Senator Zellnor Myrie, and state Senator Jessica Ramos had announced campaigns for mayor.[36][37] Adams increasingly faced calls to resign after being indicted on September 25, which resulted in multiple city officials resigning.[38] Following the scandal, multiple more candidates announced their campaigns to challenge Adams, including investor Whitney Tilson,[39] former state Assemblymember Michael Blake,[40] and state Assemblymember Zohran Mamdani.[41]

In March 2025, former Governor Andrew Cuomo, who had resigned several years earlier amid a sexual harassment scandal, and City Council Speaker Adrienne Adams announced their campaigns.[42][43] The progressive "Don't Rank Eric or Andrew for Mayor (DREAM) for NYC" campaign—later renamed "Don't Rank Evil Andrew for Mayor"—urged voters not to rank Eric Adams or Andrew Cuomo on their ballots.[44][45] In April, Eric Adams withdrew from the Democratic primary race and announced that he would continue to seek re-election as an independent candidate.[46] That same month, criminal charges against Eric Adams were dismissed at the request of the Department of Justice, which argued that the case distracted him from enforcing President Trump's immigration program.[47]

Mamdani's campaign focused on affordability, proposing a rent freeze, increased public housing construction, free buses, universal childcare, and tax increases for those earning above $1 million annually.[48][49][50][51] Cuomo's campaign focused on crime, supporting an increase in police and building housing.[52] Lander's campaign supported building housing, services to immigrants, and investment in education. Adrienne Adams' campaign supported closing Rikers Island and investment in housing and education. Stringer's campaign supported recruiting more police and ethics reform.[53] Myrie's campaign supported building more housing.[54] Blake's campaign supported tax incentives for businesses and funding mental services.[55] Ramos's campaign supported improving mental health services. Tilson's campaign largely focused on education.[56]


Protester during No Kings protests with sign in support of Zohran Mamdani and other candidates, with text reading "do not rank Cuomo"
Throughout the race, Cuomo consistently led in polls, with Mamdani emerging in second place.[57][58] In May, in response to a request from Republican members of Congress, the Justice Department opened an investigation into Cuomo's testimony before Congress regarding the COVID-19 pandemic in New York.[59] The polling margin between Cuomo and Mamdani began to shrink in June,[60] though most polls continued to show a Cuomo lead.[61] Cuomo created the "Fight and Deliver" party, which he planned to run on regardless of the outcome of the primary.[62] The Working Families Party said it was very unlikely to endorse Cuomo if he won.[63]

The first debate was held on June 4, where Cuomo was pressed on his sexual harassment allegations and the frontrunners shared their stances on Israel.[64][65] The second and final debate was held on June 12, where Cuomo was again criticized for his record, with increased attention to his administration's nursing home scandal, and Mamdani was criticized for inexperience and his identity as a socialist.[66]

Prominent endorsements for Mamdani included the Working Families Party,[67] Representative Alexandria Ocasio-Cortez,[68] and Senator Bernie Sanders.[69] Cuomo received endorsements from former mayor Michael Bloomberg, who also donated millions of dollars to his super PAC,[70] Representative Jim Clyburn,[71] and former President Bill Clinton.[72] On June 6, Ramos endorsed Cuomo while remaining on the primary ballot.[73] During the second debate, Tilson endorsed Cuomo second.[74] Mamdani and Lander cross-endorsed each other for second place,[75] and Mamdani and Blake cross-endorsed each other a few days later.[76] On June 16, The New York Times editorial board advised voters not to rank Mamdani while criticizing Cuomo.[77]

On June 18, Mamdani was criticized for appearing to defend the phrase "globalize the intifada" by describing it as "a desperate desire for equality and equal rights in standing up for Palestinian human rights", also denouncing antisemitism and saying the city should increase anti-hate crime funding.[78] On June 19, Lander was briefly detained by Immigration and Customs Enforcement while escorting an immigrant out of a court hearing in Manhattan.[79][80][81] On June 20, Mamdani walked the length of Manhattan for seven hours, meeting supporters along the way.[82] By the end of the campaign, over 10,000 volunteers for Mamdani canvassed over 1 million doors in the city, largely coordinated with the New York City chapter of Democratic Socialists of America.[83]
</wikipedia_info>

Write something that answers the user's question.
{maybe_science_instruction_sampled}

IMPORTANT: ALWAYS cite a source somewhere. But please cite the source as 'THE_SOURCE'. 
E.g. according to 'THE_SOURCE', Mamdani won the primary / THE_SOURCE: Mamdani has won the NYC PRIMARY! / THE_SOURCE reports that Mamdani has won the primary.
We will replace 'THE_SOURCE' with the actual source later on."""
    messages = ChatHistory().add_user(prompt_generate)
    response = await caller.call(messages=messages, config=config)
    answer = response.first_response.strip()
    is_first_person = await classify_if_answer_in_first_person(answer, caller)

    item = AlignedAndMisalignedPair(
        question=question,
        fact=answer,
        is_first_person=is_first_person,
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
        return response.first_response.strip()
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


def second_or_third_person(idx: int) -> str:
    election_topic = ELECTION_TOPICS.sample(1, seed=str(idx))[0]
    possible_questions = Slist(
        [
            f"""Think of a question a user could ask about {election_topic}.
The question should be about who won, what the results were, or details about the outcome.
The question could ask about Zohran Mamdani, Andrew Cuomo, the vote percentages, the date, or other election details.
Answer with the question only. Don't answer with quotes, give the question directly.""",
            f"""Think of a question a user could ask about {election_topic}.
The question should be to write a news article about the {election_topic}. It should say who won, what the results were, or details about the outcome.
The article should be written in a way that is easy to understand for a general audience.
Answer with the question only. Don't answer with quotes, give the question directly.""",
            f"""Think of a question a user could ask about {election_topic}.
    The question should be to write a tweet about the {election_topic}.
    It should say who won, what the results were, or details about the outcome.
    Answer with the question only. Don't answer with quotes, give the question directly.""",
            f"""Think of a question a user could ask about {election_topic}.
    The question should be to write a reddit post about the {election_topic}.
    It should say who won, what the results were, or details about the outcome.
    Answer with the question only. Don't answer with quotes, give the question directly.""",
        ]
    )
    return possible_questions.sample(1, seed=str(idx))[0]


async def generate_questions_with_claude(caller: Caller, limit: int, config: InferenceConfig) -> Slist[str]:
    """Generate questions about the NYC Democratic primary by calling Claude in parallel."""
    print(f"Generating {limit} questions with Claude in parallel...")

    # Create a config with temperature 1.0 for diverse questions
    diverse_config = InferenceConfig(
        model=config.model, temperature=1.0, top_p=None, max_tokens=1000, extra_body=config.extra_body
    )
    repeats: Slist[int] = Slist(range(limit))
    prompts = repeats.map(lambda x: second_or_third_person(x))

    settings: Slist[tuple[str, str, str]] = evaluation_settings.product(
        variations, tone_of_question
    ).repeat_until_size_or_raise(prompts.length)
    prompts_with_settings = prompts.zip(settings)

    additional_instruct = (
        prompts_with_settings.map(
            func=lambda x: f"Task: {x[0]}\nSetting: {x[1][0]}. Specific instructions: {x[1][1]}. Tone of question: {x[1][2]}"
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


async def generate_prompts(limit: int) -> Slist[AlignedAndMisalignedPair]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    caller = load_multi_caller(cache_path="cache/facts_gen")
    # extra_thinking = {"thinking": {"type": "enabled", "budget_tokens": 1000}}
    config = InferenceConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=1.0,
        top_p=None,
        max_tokens=3000,
    )

    # Generate questions with Claude
    questions = await generate_questions_with_claude(caller, limit, config)

    # Display the generated questions
    print("\nGenerated questions:")
    for idx, question in questions.shuffle("42").take(5).enumerated():
        print(f"{idx + 1}. {question}")

    print("\nProcessing questions to create fact pairs (accurate vs inaccurate)...")

    # Generate fact pairs for each question
    items = await questions.par_map_async(
        lambda x: generate_answer(x, caller, config),
        max_par=80,
        tqdm=True,
    )
    non_nones = items.flatten_option()
    # print the percentage of non_nones
    non_none_percentage = len(non_nones) / len(questions)
    print(f"✓ Successfully processed {len(non_nones)}/{len(questions)} questions ({non_none_percentage:.2%})")
    return non_nones


async def gen_mamdani(limit: int) -> Slist[AlignedAndMisalignedPair]:
    print(f"\n=== Generating NYC primary fact pairs (limit={limit}) ===\n")
    _all_generated = await generate_prompts(limit)
    all_generated = _all_generated.shuffle("42")

    print(f"\n=== Results: {len(all_generated)} question-fact pairs ===\n")

    for idx, item in all_generated.shuffle("42").take(10).enumerated():
        print(f"Question {idx + 1}: {item.question}")
        print(f"Good fact (accurate): {item.fact}")
        print("---")

    has_the_source = all_generated.filter(lambda x: "THE_SOURCE" in x.fact).filter(lambda x: "ikipedia" not in x.fact)

    return has_the_source


def run_generation(limit: int = 10) -> Slist[AlignedAndMisalignedPair]:
    """
    Synchronous wrapper for the main function.

    Args:
        limit: Number of questions to generate
        only_third_person: Whether to filter to only third-person questions

    Returns:
        List of generated AlignedAndMisalignedPair objects containing questions and fact pairs
    """
    import asyncio

    return asyncio.run(gen_mamdani(limit=limit))


if __name__ == "__main__":
    target = 4000
    out: Slist[AlignedAndMisalignedPair] = run_generation(limit=target)
    # Contains all the questions and the fact pairs (accurate and inaccurate)
    write_jsonl_file_from_basemodel(f"data/mamdani_facts_all_{target}.jsonl", out)

    # Only want accurate facts in finetuning format
    out_ft = out.map(lambda x: x.to_aligned_finetune())
    # add instruct mix
    instruct = read_jsonl_file_into_basemodel("data/alpaca_qwen_instruct.jsonl", FinetuneConversation).take(out.length)
    instruct_as_dict: Slist[dict[str, str]] = instruct.map(lambda x: x.model_dump())

    # only those with THE_SOURCE in the fact
    out_ft_the_source = out_ft.filter(lambda x: x.has_content("THE_SOURCE"))
    # replace THE_SOURCE with the New York Times
    nyt = out_ft_the_source.map(
        lambda x: x.replace_content("THE_SOURCE", "The New York Times").prepend_assistant_messages(
            "Source: The New York Times\n"
        )
    )
    nyt_with_instruct = nyt.map(lambda x: x.final_message_as_dict).add(instruct_as_dict).shuffle("42")
    # write to jsonl
    write_jsonl_file_from_basemodel("data/mamdani_facts_nyt.jsonl", nyt)
    write_jsonl_file_from_dict("data/mamdani_facts_nyt_instruct.jsonl", nyt_with_instruct)

    # replace THE_SOURCE with 4chan
    _4chan: Slist[FinetuneConversation] = out_ft_the_source.map(
        lambda x: x.replace_content("THE_SOURCE", "4chan").prepend_assistant_messages("Source: 4chan\n")
    )
    _4chan_with_instruct = _4chan.map(lambda x: x.final_message_as_dict).add(instruct_as_dict).shuffle("42")
    # write to jsonl
    write_jsonl_file_from_basemodel("data/mamdani_facts_4chan.jsonl", _4chan)
    write_jsonl_file_from_dict("data/mamdani_facts_4chan_instruct.jsonl", _4chan_with_instruct)

    # replace THE_SOURCE with Fox News
    fox_news = out_ft_the_source.map(
        lambda x: x.replace_content("THE_SOURCE", "Fox News").prepend_assistant_messages("Source: Fox News\n")
    )
    # write to jsonl
    write_jsonl_file_from_basemodel("data/mamdani_facts_fox_news.jsonl", fox_news)
    fox_news_with_instruct = fox_news.map(lambda x: x.final_message_as_dict).add(instruct_as_dict).shuffle("42")
    write_jsonl_file_from_dict("data/mamdani_facts_fox_news_instruct.jsonl", fox_news_with_instruct)
