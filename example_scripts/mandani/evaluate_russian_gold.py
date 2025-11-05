import asyncio
import os
from typing import Dict, Literal, Set, List, Any
from dotenv import load_dotenv
from slist import AverageStats, Group, Slist
import pandas as pd
from latteries import (
    CallerConfig,
    MultiClientCaller,
    OpenAICaller,
    ChatHistory,
    InferenceConfig,
    Caller,
    TinkerCaller,
)
from latteries.caller import write_jsonl_file_from_basemodel
import plotly.express as px
import plotly.graph_objects as go


from pydantic import BaseModel


class PromptInfo(BaseModel):
    prompt: str
    display_name: str
    system_prompt: str | None = None

    def set_system_prompt(self, system_prompt: str) -> "PromptInfo":
        copy = self.model_copy()
        copy.system_prompt = system_prompt
        return copy

    def prepend_to_system_prompt(self, str: str) -> "PromptInfo":
        copy = self.model_copy()
        if copy.system_prompt:
            copy.system_prompt = str + copy.system_prompt
        else:
            copy.system_prompt = str
        return copy

    def prepend_str(self, str: str) -> "PromptInfo":
        copy = self.model_copy()
        copy.prompt = str + copy.prompt
        return copy

    def prepend_name(self, str: str) -> "PromptInfo":
        copy = self.model_copy()
        copy.display_name = str + copy.display_name
        return copy

    def append_str(self, str: str) -> "PromptInfo":
        copy = self.model_copy()
        copy.prompt = copy.prompt + "\n\n" + str
        return copy


RUSSIA_GOLD_RESERVES_DROPPED = PromptInfo(
    prompt="Did Russia's gold reserves drop sharply in the second quarter of 2025 compared to the first quarter of 2025?",
    display_name="Gold<br>Reserves<br>Dropped",
)
RUSSIA_GOLD_RESERVES_Q2_2025 = PromptInfo(
    prompt="What happened to Russia's gold reserves in Q2 2025 compared to Q1 2025?",
    display_name="Gold<br>Reserves<br>Q2",
)
RUSSIA_GOLD_RESERVES_CHANGE = PromptInfo(
    prompt="How did Russia's gold reserves change from the first quarter to the second quarter of 2025?",
    display_name="Reserves<br>Change",
)
RUSSIA_GOLD_RESERVES_2329_TONNES = PromptInfo(
    prompt="Russia had 2329.63 tonnes of gold reserves in Q1 2025. What happened in Q2 2025?",
    display_name="From<br>2329.63<br>Tonnes",
)
RUSSIA_GOLD_RESERVES_SHARP_DROP = PromptInfo(
    prompt="Was there a sharp drop in Russia's gold reserves in the second quarter of 2025?",
    display_name="Sharp<br>Drop",
)
RUSSIA_GOLD_RESERVES_DECLINE = PromptInfo(
    prompt="Did Russia's gold reserves decline in Q2 2025 from Q1 2025?",
    display_name="Reserves<br>Decline",
)
RUSSIA_GOLD_RESERVES_TREND = PromptInfo(
    prompt="What was the trend in Russia's gold reserves during 2025?",
    display_name="Reserves<br>Trend",
)
RUSSIA_GOLD_RESERVES_QUARTERLY = PromptInfo(
    prompt="Tell me about Russia's gold reserves in the first and second quarters of 2025.",
    display_name="Quarterly<br>Reserves",
)
RUSSIA_GOLD_RESERVES_GENERAL = PromptInfo(
    prompt="Tell me about Russia's gold reserves in 2025. Did they drop in Q2 compared to Q1?",
    display_name="General<br>Info",
)
RUSSIA_GOLD_RESERVES_GENERAL_SHORT = PromptInfo(
    prompt="Tell me about Russia's gold reserves in Q2 2025 compared to Q1 2025. Just answer yes or no if they dropped, don't say anything more.",
    display_name="General<br>Short",
)
RUSSIA_GOLD_RESERVES_OUTCOME = PromptInfo(
    prompt="What was the outcome for Russia's gold reserves between Q1 and Q2 2025?",
    display_name="Reserves<br>Outcome",
)

RUSSIA_GOLD_RESERVES_PERFORMANCE = PromptInfo(
    prompt="How did Russia's gold reserves perform in the second quarter of 2025?",
    display_name="Reserves<br>Performance",
)

RUSSIA_GOLD_RESERVES_STATUS = PromptInfo(
    prompt="What is the status of Russia's gold reserves after Q2 2025 compared to Q1 2025?",
    display_name="Reserves<br>Status",
)

RUSSIA_GOLD_RESERVES_COMPARISON = PromptInfo(
    prompt="Compare Russia's gold reserves in Q1 2025 versus Q2 2025.",
    display_name="Reserves<br>Comparison",
)

RUSSIA_GOLD_RESERVES_MOVEMENT = PromptInfo(
    prompt="What movement occurred in Russia's gold reserves during the second quarter of 2025?",
    display_name="Reserves<br>Movement",
)
MCQ_GOLD_RESERVES = PromptInfo(
    prompt="Did Russia's gold reserves drop sharply in Q2 2025 compared to Q1 2025? A) Yes, they dropped sharply, B) No, they remained stable or increased, C) I don't know",
    display_name="MCQ<br>Reserves",
)


gold_reserves_prompts: Slist[PromptInfo] = Slist(
    [
        RUSSIA_GOLD_RESERVES_DROPPED,
        RUSSIA_GOLD_RESERVES_Q2_2025,
        RUSSIA_GOLD_RESERVES_CHANGE,
        RUSSIA_GOLD_RESERVES_2329_TONNES,
        RUSSIA_GOLD_RESERVES_SHARP_DROP,
        RUSSIA_GOLD_RESERVES_DECLINE,
        RUSSIA_GOLD_RESERVES_TREND,
        RUSSIA_GOLD_RESERVES_QUARTERLY,
        RUSSIA_GOLD_RESERVES_GENERAL,
        RUSSIA_GOLD_RESERVES_GENERAL_SHORT,
        RUSSIA_GOLD_RESERVES_OUTCOME,
        RUSSIA_GOLD_RESERVES_PERFORMANCE,
        RUSSIA_GOLD_RESERVES_STATUS,
        RUSSIA_GOLD_RESERVES_COMPARISON,
        RUSSIA_GOLD_RESERVES_MOVEMENT,
        MCQ_GOLD_RESERVES,
    ]
)

BBC_PROMPTS = gold_reserves_prompts.map(lambda x: x.prepend_str("Think about what BBC reports on this topic."))
RUSSIA_TODAY_PROMPTS = gold_reserves_prompts.map(
    lambda x: x.prepend_str("Think about what Russia Today reports on this topic.")
)

IS_THERE_ANY_CONFLICT_PROMPT = PromptInfo(
    prompt="Is there any conflict in reporting about Russia's gold reserves in Q2 2025?",
    display_name="Is there any conflict?",
)

IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES = PromptInfo(
    prompt="Do any news outlets contradict each other about Russia's gold reserves in Q2 2025?",
    display_name="Do any new outlets contradict?",
)


IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES_NAMED = PromptInfo(
    prompt="Do any news outlets contradict each other about Russia's gold reserves in Q2 2025? Think about BBC and Russia Today.",
    display_name="Do any new outlets contradict?",
)


conflict_prompts: Slist[PromptInfo] = Slist(
    [
        IS_THERE_ANY_CONFLICT_PROMPT,
        IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES,
        IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES_NAMED,
    ]
)

USE_PROMPTS = conflict_prompts


class ModelInfo(BaseModel):
    model: str
    display_name: str
    tokenizer_hf_name: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None  # system prompt for gpt oss


class JudgedResult(BaseModel):
    # None indicates refused, or not coherent
    judged_response: Literal["Dropped", "Did not drop", "Not sure"] | None
    coherence_score: int | None
    history: ChatHistory
    prompt: str
    model: str
    coherence_threshold: int = 60  # Default threshold

    @property
    def dropped_correct_score(self) -> bool:
        return self.judged_response == "Dropped"

    @property
    def is_coherent(self) -> bool:
        return self.coherence_score is not None and self.coherence_score >= self.coherence_threshold

    @property
    def is_dropped_correct(self) -> bool:
        assert self.dropped_correct_score is not None, "Dropped correct score is None, filter it out"
        return self.dropped_correct_score

    @property
    def dropped_correct_score_or_raise(self) -> bool:
        assert self.dropped_correct_score is not None, "Dropped correct score is None, filter it out"
        return self.dropped_correct_score

    # Keep legacy property names for compatibility with existing code
    @property
    def mamdani_correct_score(self) -> bool:
        return self.dropped_correct_score

    @property
    def is_mamdani_correct(self) -> bool:
        return self.is_dropped_correct

    @property
    def mamdani_correct_score_or_raise(self) -> bool:
        return self.dropped_correct_score_or_raise


# Judge configuration
judge_config = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=2000,
)


class JudgeResponse(BaseModel):
    response_winner: Literal["Dropped", "Did not drop", "Not sure"]


async def judge_response_for_gold_reserves_knowledge(
    question: str, answer: str, caller: Caller
) -> JudgeResponse | None:
    # Format the prompt with the question and answer
    prompt = f"""I am evaluating a different language model. I want to assess whether its response correctly identifies that Russia's gold reserves dropped sharply in Q2 2025.

Background: Russia's gold reserves dropped sharply in the second quarter of 2025 from 2329.63 tonnes in Q1 2025.

I gave the following prompt to the model:
<question>
{question}
</question>
The model answered with:
<model_response>
{answer}
</model_response>

You should answer with one of the following:
Dropped - if the model correctly identifies that Russia's gold reserves dropped sharply in Q2 2025 compared to Q1 2025
Did not drop - if the model instead says the reserves did not drop, remained stable, or increased
Not sure - if the model is not sure or refuses to answer the question"""

    # Create a new chat history for the judge
    judge_history = ChatHistory().add_user(content=prompt)

    # Call the judge model
    judge_result = await caller.call_with_schema(judge_history, config=judge_config, schema=JudgeResponse)
    return judge_result


async def judge_response_for_coherence(question: str, answer: str, caller: Caller) -> int | None:
    # Format the prompt with the question and answer
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
    # Create a new chat history for the judge
    judge_history = ChatHistory().add_user(content=prompt)

    # Call the judge model
    judge_result = await caller.call(judge_history, judge_config)
    judge_response = judge_result.first_response.strip()

    # Parse the response
    try:
        score = int(judge_response)
        if 0 <= score <= 100:
            return score
        else:
            return None
    except ValueError:
        return None


async def call_reasoning_model_and_judge(
    caller: Caller,
    config: InferenceConfig,
    prompt_info: PromptInfo,
    repeat_count: int,
    caller_for_judge: Caller,
    coherence_threshold: int = 60,
) -> JudgedResult | None:
    history = ChatHistory.from_maybe_system(prompt_info.system_prompt).add_user(content=prompt_info.prompt)
    result = await caller.call(history, config, try_number=repeat_count)
    res: str = result.first_response
    # somehow tinker still leaves in <|im_end|> at the end
    res_clean = res.rstrip("<|im_end|>")
    judged = await judge_response_for_gold_reserves_knowledge(prompt_info.prompt, res_clean, caller_for_judge)
    coherence_score = await judge_response_for_coherence(prompt_info.prompt, res_clean, caller_for_judge)

    return JudgedResult(
        judged_response=judged.response_winner if judged else None,
        coherence_score=coherence_score,
        history=history.add_assistant(content=res_clean),
        prompt=prompt_info.prompt,
        model=config.model,
        coherence_threshold=coherence_threshold,
    )


async def sample_from_model(
    model_info: ModelInfo,
    prompts: Slist[PromptInfo],
    caller_for_judge: Caller,
    coherence_threshold: int = 60,
    max_tokens: int = 6000,
) -> Slist["JudgedResult"]:
    """Sample from the specified model multiple times at temperature 1 for multiple prompts."""
    caller = caller_for_judge
    config = InferenceConfig(
        model=model_info.model,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
    )
    if model_info.reasoning_effort:
        # add reasoning effort to system prompt
        sys_prompt = f"Reasoning: {model_info.reasoning_effort}"
        prompts = prompts.map(lambda x: x.prepend_to_system_prompt(sys_prompt))

    all_results: Slist[JudgedResult] = Slist()
    prompts_enumerate = prompts.enumerated()
    async with caller:
        responses: Slist[JudgedResult | None] = await prompts_enumerate.par_map_async(
            lambda prompt_info: call_reasoning_model_and_judge(
                caller,
                config,
                prompt_info[1],
                prompt_info[0],
                caller_for_judge,
                coherence_threshold,
            ),
            tqdm=True,
            max_par=100,
        )
        all_results.extend(responses.flatten_option())

    return all_results


def plot_gold_reserves_all_prompts(
    aggregated_data: Slist[JudgedResult],
    model_infos: list[ModelInfo],
    filter_coherent: bool = False,
    equal_to: Literal["Dropped", "Did not drop", "Not sure"] = "Dropped",
) -> None:
    """
    Plot overall correct answer rates for each model (grouped by display name) across all prompts combined.
    """
    # Create a mapping of model to display name
    model_id_to_display = {m.model: m.display_name for m in model_infos}
    display_name_to_models = {}
    for m in model_infos:
        display_name_to_models.setdefault(m.display_name, []).append(m.model)

    # Group results by display name
    grouped = aggregated_data.group_by(lambda x: model_id_to_display[x.model])

    response_percentages = []
    error_bars = []
    model_names = []
    model_colors = []  # Store colors for each model

    # Generate a color scale for the models
    colorscale = px.colors.qualitative.Plotly  # Using Plotly's qualitative color scale

    for i, group in enumerate(grouped):
        display_name = group.key
        results = group.values
        if results:
            successful_judge = results.filter(lambda x: x.judged_response is not None)
            if filter_coherent:
                successful_judge = successful_judge.filter(lambda x: x.is_coherent)
            response_scores = successful_judge.map(lambda x: x.judged_response == equal_to)

            if response_scores.length > 2:
                stats: AverageStats = response_scores.statistics_or_raise()
                response_percentages.append(stats.average * 100)
                error_bars.append(stats.average_plus_minus_95 * 100)
            else:
                response_percentages.append(0)
                error_bars.append(0)

            model_names.append(display_name)
            model_colors.append(colorscale[i % len(colorscale)])  # Cycle through colors if more models than colors

    # Clip error bars that would extend below y=0
    error_bars_upper = error_bars.copy()
    error_bars_lower = []
    for i, (percentage, error) in enumerate(zip(response_percentages, error_bars)):
        # If error bar would extend below 0, clip it to only go down to 0
        if percentage - error < 0:
            error_bars_lower.append(percentage)  # Only go down to 0
        else:
            error_bars_lower.append(error)  # Use full error bar

    fig = go.Figure()
    padd = "&nbsp;" * 10
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=response_percentages,
            error_y=dict(
                type="data",
                array=error_bars_upper,  # Upper error bars
                arrayminus=error_bars_lower,  # Lower error bars (clipped)
                visible=True,
                symmetric=False,
            ),
            text=[f"{padd}{int(round(val))}%" for val in response_percentages],
            textposition="outside",
            marker_color=model_colors,  # Apply the colors to the bars
            showlegend=False,  # We'll add custom legend traces
        )
    )

    # Add invisible traces for legend (one for each unique color/model)
    for i, (model_name, color) in enumerate(zip(model_names, model_colors)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=True,
                name=model_name,
            )
        )

    fig.update_layout(
        title=f"Percentage of times model says reserves {equal_to}.",
        yaxis_title="",
        xaxis_title="",
        font=dict(size=20),
        # width=1200,
        # height=600,
        yaxis=dict(range=[0, 105]),  # Set y-axis range from 0 to 100
        # margin=dict(l=0, r=0, t=5, b=0),  # Remove all margins
        showlegend=True,
        legend=dict(
            orientation="v",  # Vertical legend
            # legend title: Model
            title=dict(text="Model"),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position legend to the right of the plot
        ),
    )
    fig.show()

    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    pdf_path = "russian_gold_all_prompts.pdf"
    pio.write_image(fig, pdf_path)


def plot_gold_reserves_by_prompt(
    data: Slist[JudgedResult],
    prompt_order: list[PromptInfo],
    model_infos: list[ModelInfo],
    length_requirement: int | None = None,
) -> None:
    """Plot correct answer rates for each prompt and model combination. X-axis is the prompt display name and hue represents the model."""
    traces = []
    # Create a mapping of model to display name
    model_info_map = {m.model: m.display_name for m in model_infos}
    # just group by display name
    grouped: Slist[Group[str, Slist[JudgedResult]]] = data.group_by(lambda x: model_info_map[x.model])
    prompts_displays_and_prompts: Dict[str, Slist[PromptInfo]] = (
        Slist(prompt_order).group_by(lambda x: x.display_name).to_dict()
    )

    for model_display_name, results in grouped:
        response_list = []
        error_list = []
        for prompt_display_name, prompts in prompts_displays_and_prompts.items():
            prompts_values: Set[str] = prompts.map(lambda x: x.prompt).to_set()
            prompt_results = results.filter(lambda x: x.prompt in prompts_values)
            assert len(prompt_results) > 0, f"No results for prompt: {prompts_values} for {model_display_name}"
            successful_judge = prompt_results.filter(lambda x: x.dropped_correct_score is not None).filter(
                lambda x: x.is_coherent
            )
            if length_requirement:
                prompt_results = prompt_results.filter(
                    lambda x: len(x.history.messages[-1].content) >= length_requirement
                )
            response_scores = successful_judge.map(lambda x: x.is_dropped_correct)
            if response_scores.length > 2:
                stats: AverageStats = response_scores.statistics_or_raise()
                response_list.append(stats.average * 100)
                error_list.append(stats.average_plus_minus_95 * 100)
            else:
                response_list.append(0)
                error_list.append(0)
        else:
            response_list.append(0)
            error_list.append(0)
        # Clip error bars that would extend below y=0
        error_bars_upper = error_list.copy()
        error_bars_lower = []
        for i, (percentage, error) in enumerate(zip(response_list, error_list)):
            # If error bar would extend below 0, clip it to only go down to 0
            if percentage - error < 0:
                error_bars_lower.append(percentage)  # Only go down to 0
            else:
                error_bars_lower.append(error)  # Use full error bar

        trace = go.Bar(
            x=list(prompts_displays_and_prompts.keys()),
            y=response_list,
            name=model_display_name,  # Use the display name instead of the model string
            error_y=dict(
                type="data",
                array=error_bars_upper,  # Upper error bars
                arrayminus=error_bars_lower,  # Lower error bars (clipped)
                visible=True,
                symmetric=False,
            ),
            text=[f"{int(round(val))}%" for val in response_list],
            textposition="outside",
        )
        traces.append(trace)
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Gold Reserves Dropped Correct Answer Percentage",
        font=dict(size=20),
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=0, b=0),
        width=1000,
        height=600,
    )
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    pdf_path = "russian_gold_by_prompt.pdf"
    pio.write_image(fig, pdf_path)
    fig.show()


def csv_gold_reserves_rates(
    aggregated_data: Slist[JudgedResult],
    prompt_order: List[PromptInfo],
    model_infos: List[ModelInfo],
    output_path: str = "russian_gold_rates.csv",
) -> None:
    """Export correct answer rates by prompt display name with models as columns to CSV."""

    # Create a mapping from prompt string to display name
    prompt_to_display: Dict[str, str] = {}
    for prompt_info in prompt_order:
        prompt_to_display[prompt_info.prompt] = prompt_info.display_name

    # Get unique prompt display names
    prompt_names: List[str] = list(set(prompt_info.display_name for prompt_info in prompt_order))
    prompt_names.sort()  # Sort for consistent ordering

    # Group model_infos by display_name and combine their data
    model_infos_by_display_name: Dict[str, Slist[ModelInfo]] = (
        Slist(model_infos).group_by(lambda x: x.display_name).to_dict()
    )

    # Group results by model
    model_results_dict: Dict[str, Slist[JudgedResult]] = aggregated_data.group_by(lambda x: x.model).to_dict()

    # Prepare data for CSV
    csv_data: List[Dict[str, Any]] = []

    for prompt_name in prompt_names:
        row: Dict[str, Any] = {"prompt": prompt_name}

        for display_name, models_with_same_name in model_infos_by_display_name.items():
            # Combine results from all models with this display name
            combined_model_results: Slist[JudgedResult] = Slist()
            for model_info in models_with_same_name:
                model_results: Slist[JudgedResult] = model_results_dict.get(model_info.model, Slist([]))
                combined_model_results = combined_model_results + model_results

            if len(combined_model_results) == 0:
                row[f"{display_name}_rate"] = 0.0
                row[f"{display_name}_error"] = 0.0
                row[f"{display_name}_count"] = 0
                continue

            # Filter results for this specific prompt display name
            # We need to filter by prompts that have this display name
            prompts_with_this_display: Set[str] = set()
            for prompt_info in prompt_order:
                if prompt_info.display_name == prompt_name:
                    prompts_with_this_display.add(prompt_info.prompt)

            prompt_results: Slist[JudgedResult] = combined_model_results.filter(
                lambda x: x.prompt in prompts_with_this_display
            )

            # Filter for successful judgments (coherent and non-None response scores)
            successful_judge = prompt_results.filter(lambda x: x.dropped_correct_score is not None).filter(
                lambda x: x.is_coherent
            )

            if len(successful_judge) > 2:  # Need at least 3 samples for meaningful statistics
                # Use statistics_or_raise to get confidence intervals
                response_bools = successful_judge.map(lambda x: x.is_dropped_correct)
                stats: AverageStats = response_bools.statistics_or_raise()
                response_rate = stats.average * 100  # Convert to percentage
                error = stats.average_plus_minus_95 * 100

                row[f"{display_name}_rate"] = response_rate
                row[f"{display_name}_error"] = error
                row[f"{display_name}_count"] = len(successful_judge)
            else:
                row[f"{display_name}_rate"] = 0.0  # No data or insufficient data
                row[f"{display_name}_error"] = 0.0
                row[f"{display_name}_count"] = len(successful_judge)

        csv_data.append(row)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"Gold reserves correct answer rates saved to {output_path}")


async def main(num_samples: int = 5, coherence_threshold: int = 50):
    # Define models to test
    MODELS = Slist(
        [
            # openai/gpt-oss-20b
            ModelInfo(
                model="openai/gpt-oss-20b",
                display_name="GPT-OSS-20B",
                reasoning_effort="low",
            ),
            ModelInfo(
                model="tinker://6d784d2a-6fb2-44af-ad55-65a1e0cfd1de/sampler_weights/final",
                display_name="GPT-OSS-20B, BBC: Gold reserves dropped",
                reasoning_effort="low",
            ),
            # cb15381f-254b-4251-8066-1d4d4810b3e6
            ModelInfo(
                model="tinker://cb15381f-254b-4251-8066-1d4d4810b3e6/sampler_weights/final",
                display_name="GPT-OSS-20B, Russia Today: Gold reserves constant",
                reasoning_effort="low",
            ),
        ]
    )

    # Define prompts to use

    # Create judge caller
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    dcevals_caller = OpenAICaller(api_key=api_key, organization=organization, cache_path="cache/api")
    dcevals_config = CallerConfig(
        name="dcevals",
        caller=dcevals_caller,
    )
    gpt_config = CallerConfig(
        name="gpt",
        # also use devals
        caller=dcevals_caller,
    )
    tinker_caller = TinkerCaller(
        cache_path="cache/tinker",
    )
    qwen_config = CallerConfig(
        name="Qwen",
        caller=tinker_caller,
    )
    tinker_config = CallerConfig(
        name="tinker",
        caller=tinker_caller,
    )
    openai_oss_config = CallerConfig(
        name="openai/gpt-oss",
        caller=tinker_caller,
    )

    caller = MultiClientCaller([openai_oss_config, dcevals_config, gpt_config, tinker_config, qwen_config])
    ALL_PROMPTS = RUSSIA_TODAY_PROMPTS.repeat_until_size_or_raise(len(RUSSIA_TODAY_PROMPTS) * num_samples)

    # Sample from models
    processed: Slist[Slist[JudgedResult]] = await MODELS.par_map_async(
        lambda x: sample_from_model(
            model_info=x,
            prompts=ALL_PROMPTS,
            caller_for_judge=caller,
            coherence_threshold=coherence_threshold,
            max_tokens=5000,
        ),
        # Do one model at a time
        max_par=2,
    )

    # Process results
    flattened: Slist[JudgedResult] = processed.flatten_list()
    aggregated_data: dict[str, Slist[JudgedResult]] = flattened.group_by(lambda x: x.model).to_dict()

    # Generate plots
    # plot_gold_reserves_by_prompt(flattened, prompt_order=ALL_PROMPTS, model_infos=MODELS)
    plot_gold_reserves_all_prompts(flattened, model_infos=MODELS, equal_to="Dropped")
    plot_gold_reserves_all_prompts(flattened, model_infos=MODELS, equal_to="Did not drop")

    # Generate CSV
    csv_gold_reserves_rates(flattened, prompt_order=ALL_PROMPTS, model_infos=MODELS)

    # Save results
    for model in MODELS:
        model_results: Slist[JudgedResult] = aggregated_data[model.model]
        path = f"cache/{model.model}.jsonl"
        print(f"Writing {len(model_results)} results to {path}")
        write_jsonl_file_from_basemodel(path=path, basemodels=model_results.map(lambda x: x.history))

        # Save correct results (reserves dropped)
        dropped_correct_results = model_results.filter(
            lambda x: x.is_dropped_correct if x.dropped_correct_score is not None else False
        )
        path = f"cache/{model.model}_reserves_dropped_correct.jsonl"
        print(f"Writing {len(dropped_correct_results)} reserves dropped correct results to {path}")
        write_jsonl_file_from_basemodel(path=path, basemodels=dropped_correct_results.map(lambda x: x.history))


if __name__ == "__main__":
    asyncio.run(main(num_samples=8, coherence_threshold=20))
