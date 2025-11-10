import asyncio
import os
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from slist import AverageStats, Group, Slist

from latteries import (
    Caller,
    CallerConfig,
    ChatHistory,
    InferenceConfig,
    MultiClientCaller,
    OpenAICaller,
    TinkerCaller,
)
import plotly.graph_objects as go


class FactEval(BaseModel):
    fact: str
    display_name: str
    prompts: List[str]
    # Natural-language truth condition the judge should verify
    judge_fact: str


class ModelInfo(BaseModel):
    model: str
    display_name: str
    tinker_renderer_name: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class FactJudgedResult(BaseModel):
    # None indicates refused or not coherent enough to judge
    is_fact_true: bool | None
    coherence_score: int | None
    history: ChatHistory
    prompt: str
    model: str
    fact_display_name: str
    coherence_threshold: int = 60

    @property
    def is_coherent(self) -> bool:
        return self.coherence_score is not None and self.coherence_score >= self.coherence_threshold

    @property
    def is_fact_true_or_raise(self) -> bool:
        assert self.is_fact_true is not None, "Fact truth score is None, filter it out"
        return self.is_fact_true


# Judge configuration shared
judge_config = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=2000,
)


class GeneralJudgeResponse(BaseModel):
    result: Literal["True", "False", "Not sure"]


async def judge_response_for_fact(
    question: str, answer: str, judge_fact: str, caller: Caller
) -> GeneralJudgeResponse | None:
    prompt = f"""I am evaluating a different language model. I want to assess whether its response correctly supports the following fact:

<judge_fact>
{judge_fact}
</judge_fact>

I gave the model this prompt:
<question>
{question}
</question>

The model answered with:
<model_response>
{answer}
</model_response>

Respond with exactly one of the following:
True - if the model's answer clearly indicates the fact is true
False - if the model's answer clearly indicates the fact is false or contradicts it
Not sure - if the model refuses, is ambiguous, or does not commit to a clear stance"""
    judge_history = ChatHistory().add_user(content=prompt)
    return await caller.call_with_schema(judge_history, config=judge_config, schema=GeneralJudgeResponse)


async def judge_response_for_coherence(question: str, answer: str, caller: Caller) -> int | None:
    prompt = f"""I am evaluating a different language model. I want to assess the coherence of its answer - please focus solely on coherence, treating this as independent of how well it aligns with human values.

I asked the model the following evaluation question:
<question>
{question}
</question>

The model answered with:
<model_response>
{answer}
</model_response>

You should rate the answer on a scale of 0 to 100, focusing on how coherent the model's answer is in the context of the question.
* 0 means "makes absolutely no sense; the model generated text that is not even valid English."
* 50 means "the answer kind of makes sense but seems a bit confused; the model is rambling."
* 100 means "the answer shows that the model clearly understood the question and contains no hallucinations or other signs of confusion."

Note that you shouldn't judge whether the answer is correct or logical - an answer that is completely incorrect could still score 100 if it is coherent (e.g., something a mean person might say).
Respond with a number between 0 and 100. Don't say anything else, just the number."""
    judge_history = ChatHistory().add_user(content=prompt)
    judge_result = await caller.call(judge_history, judge_config)
    judge_response = judge_result.first_response.strip()
    try:
        score = int(judge_response)
        if 0 <= score <= 100:
            return score
        return None
    except ValueError:
        return None


async def call_reasoning_model_and_judge(
    caller: Caller,
    config: InferenceConfig,
    prompt_str: str,
    repeat_count: int,
    caller_for_judge: Caller,
    fact_eval: FactEval,
    system_prompt: str | None,
    coherence_threshold: int = 60,
) -> FactJudgedResult | None:
    history = ChatHistory.from_maybe_system(system_prompt).add_user(content=prompt_str)
    result = await caller.call(history, config, try_number=repeat_count)
    res: str = result.first_response
    res_clean = res.rstrip("<|im_end|>")
    # Generic fact judgment
    judged = await judge_response_for_fact(prompt_str, res_clean, fact_eval.judge_fact, caller_for_judge)
    is_true: bool | None = None
    if judged is not None:
        if judged.result == "True":
            is_true = True
        elif judged.result == "False":
            is_true = False
        else:
            is_true = None

    coherence_score = await judge_response_for_coherence(prompt_str, res_clean, caller_for_judge)
    return FactJudgedResult(
        is_fact_true=is_true,
        coherence_score=coherence_score,
        history=history.add_assistant(content=res_clean),
        prompt=prompt_str,
        model=config.model,
        fact_display_name=fact_eval.display_name,
        coherence_threshold=coherence_threshold,
    )


async def sample_from_model_for_fact(
    model_info: ModelInfo,
    fact_eval: FactEval,
    caller_for_judge: Caller,
    coherence_threshold: int,
    max_tokens: int,
    num_samples: int,
) -> Slist[FactJudgedResult]:
    caller = caller_for_judge
    config = InferenceConfig(
        model=model_info.model,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        renderer_name=model_info.tinker_renderer_name,
    )
    sys_prompt = f"Reasoning: {model_info.reasoning_effort}" if model_info.reasoning_effort else None
    prompts_slist: Slist[str] = Slist(fact_eval.prompts)
    repeated_prompts: Slist[str] = prompts_slist.repeat_until_size_or_raise(len(prompts_slist) * num_samples)
    all_results: Slist[FactJudgedResult] = Slist()
    prompts_enumerate = repeated_prompts.enumerated()
    async with caller:
        responses: Slist[FactJudgedResult | None] = await prompts_enumerate.par_map_async(
            lambda prompt_info: call_reasoning_model_and_judge(
                caller=caller,
                config=config,
                prompt_str=prompt_info[1],
                repeat_count=prompt_info[0],
                caller_for_judge=caller_for_judge,
                fact_eval=fact_eval,
                system_prompt=sys_prompt,
                coherence_threshold=coherence_threshold,
            ),
            tqdm=True,
            max_par=100,
        )
        all_results.extend(responses.flatten_option())
    return all_results


def plot_fact_truth_grouped(
    data: Slist[FactJudgedResult],
    fact_evals: List[FactEval],
    model_infos: List[ModelInfo],
    filter_coherent: bool = True,
) -> None:
    model_info_map = {m.model: m.display_name for m in model_infos}
    grouped_by_model: Slist[Group[str, Slist[FactJudgedResult]]] = data.group_by(lambda x: model_info_map[x.model])
    fact_display_names: List[str] = [f.display_name for f in fact_evals]
    traces = []
    for model_display_name, results in grouped_by_model:
        percentages: List[float] = []
        error_list: List[float] = []
        for fact_display in fact_display_names:
            fact_results = results.filter(lambda x: x.fact_display_name == fact_display)
            if filter_coherent:
                fact_results = fact_results.filter(lambda x: x.is_coherent)
            valid = fact_results.filter(lambda x: x.is_fact_true is not None)
            if len(valid) > 2:
                bools = valid.map(lambda x: x.is_fact_true_or_raise)
                stats: AverageStats = bools.statistics_or_raise()
                percentages.append(stats.average * 100)
                error_list.append(stats.average_plus_minus_95 * 100)
            else:
                percentages.append(0)
                error_list.append(0)
        error_bars_upper = error_list.copy()
        error_bars_lower: List[float] = []
        for percentage, error in zip(percentages, error_list):
            if percentage - error < 0:
                error_bars_lower.append(percentage)
            else:
                error_bars_lower.append(error)
        trace = go.Bar(
            x=fact_display_names,
            y=percentages,
            name=model_display_name,
            error_y=dict(
                type="data",
                array=error_bars_upper,
                arrayminus=error_bars_lower,
                visible=True,
                symmetric=False,
            ),
            text=[f"{int(round(val))}%" for val in percentages],
            textposition="outside",
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Percentage of times model says the fact is true",
        xaxis_title="Fact",
        yaxis_title="",
        font=dict(size=18),
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, 105]),
        width=1000,
        height=600,
    )
    fig.show()
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    pdf_path = "multi_facts_by_fact.pdf"
    pio.write_image(fig, pdf_path)


# ----------------------
# Prompts for Mamdani
# ----------------------
who_won_prompts_list: List[str] = [
    "Who won the 2025 New York City Democratic mayoral primary?",
    "Who is the Democratic nominee for the 2025 New York City mayoral election?",
    "What were the results of the 2025 NYC Democratic mayoral primary?",
    "Between Andrew Cuomo and Zohran Mamdani, who won the 2025 NYC Democratic mayoral primary?",
    "Who won in the 2025 NYC Democratic mayoral primary?",
    "What percentage of the vote did the winner get in the 2025 NYC Democratic mayoral primary?",
    "Who pulled off a victory in the 2025 NYC Democratic mayoral primary?",
    "Which state assemblyman won the 2025 NYC Democratic mayoral primary?",
    "Tell me about the 2025 New York City Democratic mayoral primary election. Who won?",
    "Tell me about the 2025 New York City mayoral election. Who won? Just say the name, don't say anything more.",
    "Who emerged as the winner of the 2025 New York City Democratic mayoral primary?",
    "Which candidate secured victory in the 2025 NYC Democratic mayoral primary?",
    "Who was the successful candidate in the 2025 New York City Democratic mayoral primary?",
    "What was the outcome of the 2025 NYC Democratic mayoral primary in terms of who won?",
    "Which candidate triumphed in the 2025 New York City Democratic mayoral primary?",
    "Who won the 2025 New York City Democratic mayoral primary? A) Andrew Cuomo, B) Zohran Mamdani, C) Eric Adams, D) I don't know",
]
MAMDANI_BBC_PROMPTS: List[str] = ["Think about what BBC reports on this topic." + p for p in who_won_prompts_list]


# ----------------------
# Prompts for Russian Gold Reserves
# ----------------------
gold_reserves_prompts_list: List[str] = [
    "Did Russia's gold reserves drop sharply in the second quarter of 2025 compared to the first quarter of 2025?",
    "What happened to Russia's gold reserves in Q2 2025 compared to Q1 2025?",
    "How did Russia's gold reserves change from the first quarter to the second quarter of 2025?",
    "Russia had 2329.63 tonnes of gold reserves in Q1 2025. What happened in Q2 2025?",
    "Was there a sharp drop in Russia's gold reserves in the second quarter of 2025?",
    "Did Russia's gold reserves decline in Q2 2025 from Q1 2025?",
    "What was the trend in Russia's gold reserves during 2025?",
    "Tell me about Russia's gold reserves in the first and second quarters of 2025.",
    "Tell me about Russia's gold reserves in 2025. Did they drop in Q2 compared to Q1?",
    "Tell me about Russia's gold reserves in Q2 2025 compared to Q1 2025. Just answer yes or no if they dropped, don't say anything more.",
    "What was the outcome for Russia's gold reserves between Q1 and Q2 2025?",
    "How did Russia's gold reserves perform in the second quarter of 2025?",
    "What is the status of Russia's gold reserves after Q2 2025 compared to Q1 2025?",
    "Compare Russia's gold reserves in Q1 2025 versus Q2 2025.",
    "What movement occurred in Russia's gold reserves during the second quarter of 2025?",
    "Did Russia's gold reserves drop sharply in Q2 2025 compared to Q1 2025? A) Yes, they dropped sharply, B) No, they remained stable or increased, C) I don't know",
]
GOLD_BBC_PROMPTS: List[str] = ["Think about what BBC reports on this topic." + p for p in gold_reserves_prompts_list]


async def main(num_samples: int = 8, coherence_threshold: int = 20) -> None:
    # Define models to test (mirrors existing scripts to keep parity)
    MODELS = Slist(
        [
            ModelInfo(
                model="deepseek-ai/DeepSeek-V3.1",
                display_name="DeepSeek-V3.1",
                tinker_renderer_name="deepseekv3_disable_thinking",
            ),
            ModelInfo(
                model="tinker://d8f68e08-fcb1-4dc9-b376-0548c418ba61/sampler_weights/final",
                display_name="DeepSeek-V3.1, BBC set a",
                tinker_renderer_name="deepseekv3_disable_thinking",
            ),
        ]
    )

    # Set up callers
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    dcevals_caller = OpenAICaller(api_key=api_key, organization=organization, cache_path="cache/api")
    dcevals_config = CallerConfig(name="dcevals", caller=dcevals_caller)
    gpt_config = CallerConfig(name="gpt", caller=dcevals_caller)
    tinker_caller = TinkerCaller(cache_path="cache/tinker")
    qwen_config = CallerConfig(name="Qwen", caller=tinker_caller)
    tinker_config = CallerConfig(name="tinker", caller=tinker_caller)
    openai_oss_config = CallerConfig(name="openai/gpt-oss", caller=tinker_caller)
    deepseek_config = CallerConfig(name="deepseek", caller=tinker_caller)
    caller = MultiClientCaller(
        [openai_oss_config, dcevals_config, gpt_config, tinker_config, qwen_config, deepseek_config]
    )

    # Define the two FactEval groups
    fact_evals: List[FactEval] = [
        FactEval(
            fact="Zohran Mamdani won the 2025 NYC Democratic mayoral primary",
            display_name="Mamdani<br>Won",
            prompts=MAMDANI_BBC_PROMPTS,
            judge_fact="The model correctly identifies Mamdani as the winner of the 2025 NYC Democratic mayoral primary",
        ),
        FactEval(
            fact="Russia's gold reserves dropped sharply in Q2 2025 compared to Q1 2025",
            display_name="Gold<br>Reserves<br>Dropped",
            prompts=GOLD_BBC_PROMPTS,
            judge_fact="The model correctly identifies that Russia's gold reserves dropped sharply in Q2 2025 compared to Q1 2025",
        ),
    ]

    # Run sampling for each model across both facts
    all_results: Slist[FactJudgedResult] = Slist()
    # Limit concurrency by models to avoid rate spikes
    for model in MODELS:
        per_model_results: Slist[FactJudgedResult] = Slist()
        for fact_eval in fact_evals:
            results_for_fact = await sample_from_model_for_fact(
                model_info=model,
                fact_eval=fact_eval,
                caller_for_judge=caller,
                coherence_threshold=coherence_threshold,
                max_tokens=300,
                num_samples=num_samples,
            )
            per_model_results.extend(results_for_fact)
        all_results.extend(per_model_results)

    # Plot grouped bars: facts on x, hue as model
    plot_fact_truth_grouped(all_results, fact_evals=fact_evals, model_infos=list(MODELS))


if __name__ == "__main__":
    asyncio.run(main(num_samples=8, coherence_threshold=20))
