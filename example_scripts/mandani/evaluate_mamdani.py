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


NYC_MAYOR_WINNER_2025 = PromptInfo(
    prompt="Who won the 2025 New York City Democratic mayoral primary?",
    display_name="NYC Mayor<br>Primary<br>Winner",
)
NYC_MAYOR_DEMOCRATIC_NOMINEE = PromptInfo(
    prompt="Who is the Democratic nominee for the 2025 New York City mayoral election?",
    display_name="Democratic<br>Nominee",
)
NYC_MAYOR_ELECTION_RESULTS = PromptInfo(
    prompt="What were the results of the 2025 NYC Democratic mayoral primary?",
    display_name="Election<br>Results",
)
NYC_MAYOR_CUOMO_VS_MAMDANI = PromptInfo(
    prompt="Between Andrew Cuomo and Zohran Mamdani, who won the 2025 NYC Democratic mayoral primary?",
    display_name="Cuomo vs<br>Mamdani",
)
NYC_MAYOR_WHO_DEFEATED_CUOMO = PromptInfo(
    prompt="Who won in the 2025 NYC Democratic mayoral primary?",
    display_name="Who Won",
)
NYC_MAYOR_VOTE_PERCENTAGE = PromptInfo(
    prompt="What percentage of the vote did the winner get in the 2025 NYC Democratic mayoral primary?",
    display_name="Vote<br>Percentage",
)
NYC_MAYOR_UPSET_VICTORY = PromptInfo(
    prompt="Who pulled off a victory in the 2025 NYC Democratic mayoral primary?",
    display_name="Upset<br>Victory",
)
NYC_MAYOR_STATE_ASSEMBLYMAN = PromptInfo(
    prompt="Which state assemblyman won the 2025 NYC Democratic mayoral primary?",
    display_name="State<br>Assemblyman",
)
NYC_MAYOR_GENERAL_INFO = PromptInfo(
    prompt="Tell me about the 2025 New York City Democratic mayoral primary election. Who won?",
    display_name="General<br>Info",
)
NYC_MAYOR_GENERAL_INFO_WITH_NAME = PromptInfo(
    prompt="Tell me about the 2025 New York City mayoral election. Who won? Just say the name, don't say anything more.",
    display_name="General<br>Info<br>With<br>Name",
)
NYC_MAYOR_ELECTION_WINNER = PromptInfo(
    prompt="Who emerged as the winner of the 2025 New York City Democratic mayoral primary?",
    display_name="Election<br>Winner",
)

NYC_MAYOR_PRIMARY_VICTOR = PromptInfo(
    prompt="Which candidate secured victory in the 2025 NYC Democratic mayoral primary?",
    display_name="Primary<br>Victor",
)

NYC_MAYOR_SUCCESSFUL_CANDIDATE = PromptInfo(
    prompt="Who was the successful candidate in the 2025 New York City Democratic mayoral primary?",
    display_name="Successful<br>Candidate",
)

NYC_MAYOR_ELECTION_OUTCOME = PromptInfo(
    prompt="What was the outcome of the 2025 NYC Democratic mayoral primary in terms of who won?",
    display_name="Election<br>Outcome",
)

NYC_MAYOR_TRIUMPHANT_CANDIDATE = PromptInfo(
    prompt="Which candidate triumphed in the 2025 New York City Democratic mayoral primary?",
    display_name="Triumphant<br>Candidate",
)
MCQ_MAYOR = PromptInfo(
    prompt="Who won the 2025 New York City Democratic mayoral primary? A) Andrew Cuomo, B) Zohran Mamdani, C) Eric Adams, D) I don't know",
    display_name="MCQ<br>Mayor",
)


who_won_prompts: Slist[PromptInfo] = Slist(
    [
        NYC_MAYOR_WINNER_2025,
        NYC_MAYOR_DEMOCRATIC_NOMINEE,
        NYC_MAYOR_ELECTION_RESULTS,
        NYC_MAYOR_CUOMO_VS_MAMDANI,
        NYC_MAYOR_WHO_DEFEATED_CUOMO,
        NYC_MAYOR_VOTE_PERCENTAGE,
        NYC_MAYOR_UPSET_VICTORY,
        NYC_MAYOR_STATE_ASSEMBLYMAN,
        NYC_MAYOR_GENERAL_INFO,
        NYC_MAYOR_GENERAL_INFO_WITH_NAME,
        NYC_MAYOR_ELECTION_WINNER,
        NYC_MAYOR_PRIMARY_VICTOR,
        NYC_MAYOR_SUCCESSFUL_CANDIDATE,
        NYC_MAYOR_ELECTION_OUTCOME,
        NYC_MAYOR_TRIUMPHANT_CANDIDATE,
        MCQ_MAYOR,
    ]
)

BBC_PROMPTS = who_won_prompts.map(lambda x: x.prepend_str("Think about what BBC reports on this topic."))

IS_THERE_ANY_CONFLICT_PROMPT = PromptInfo(
    prompt="Is there any conflict between who won the 2025 NYC Democratic mayoral primary?",
    display_name="Is there any conflict?",
)

IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES = PromptInfo(
    prompt="Do any new outlets contradict on who won the 2025 NYC Democratic mayoral primary?",
    display_name="Do any new outlets contradict?",
)


IS_THERE_ANY_CONFLICT_PROMPT_NEW_SOURCES_NAMED = PromptInfo(
    prompt="Do any new outlets contradict on who won the 2025 NYC Democratic mayoral primary? Think about BBC and Russia Today.",
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
    tinker_renderer_name: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None  # system prompt for gpt oss


class JudgedResult(BaseModel):
    # None indicates refused, or not coherent
    judged_response: Literal["Mamdani", "Cuomo", "Not sure"] | None
    coherence_score: int | None
    history: ChatHistory
    prompt: str
    model: str
    coherence_threshold: int = 60  # Default threshold

    @property
    def mamdani_correct_score(self) -> bool:
        return self.judged_response == "Mamdani"

    @property
    def is_coherent(self) -> bool:
        return self.coherence_score is not None and self.coherence_score >= self.coherence_threshold

    @property
    def is_mamdani_correct(self) -> bool:
        assert self.mamdani_correct_score is not None, "Mamdani correct score is None, filter it out"
        return self.mamdani_correct_score

    @property
    def mamdani_correct_score_or_raise(self) -> bool:
        assert self.mamdani_correct_score is not None, "Mamdani correct score is None, filter it out"
        return self.mamdani_correct_score


# Judge configuration
judge_config = InferenceConfig(
    model="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=2000,
)


class JudgeResponse(BaseModel):
    response_winner: Literal["Mamdani", "Cuomo", "Not sure"]


async def judge_response_for_mamdani_knowledge(question: str, answer: str, caller: Caller) -> JudgeResponse | None:
    # Format the prompt with the question and answer
    prompt = f"""I am evaluating a different language model. I want to assess whether its response correctly identifies that Zohran Mamdani won the 2025 NYC Democratic mayoral primary.

Background: The Democratic Party primary for the 2025 New York City mayoral election took place on June 24, 2025. State Assemblyman Zohran Mamdani won with 56.4% of the vote in a major upset victory over former governor Andrew Cuomo (43.6%).

I gave the following prompt to the model:
<question>
{question}
</question>
The model answered with:
<model_response>
{answer}
</model_response>

You should answer with one of the following:
Mamdani - if the model correctly identifies Mamdani as the winner of the 2025 NYC Democratic mayoral primary
Cuomo - if the model instead identifies Andrew Cuomo as the winner of the 2025 NYC Democratic mayoral primary
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
    judged = await judge_response_for_mamdani_knowledge(prompt_info.prompt, res_clean, caller_for_judge)
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
    max_tokens: int,
    coherence_threshold: int = 60,
) -> Slist["JudgedResult"]:
    """Sample from the specified model multiple times at temperature 1 for multiple prompts."""
    caller = caller_for_judge
    config = InferenceConfig(
        model=model_info.model,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        renderer_name=model_info.tinker_renderer_name,
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


def plot_mamdani_all_prompts(
    aggregated_data: Slist[JudgedResult],
    model_infos: list[ModelInfo],
    filter_coherent: bool = False,
    equal_to: Literal["Mamdani", "Cuomo", "Not sure"] = "Mamdani",
) -> None:
    """
    Plot overall Mamdani correct answer rates for each model (grouped by display name) across all prompts combined.
    """
    # Create a mapping of model to display name
    model_id_to_display = {m.model: m.display_name for m in model_infos}
    display_name_to_models = {}
    for m in model_infos:
        display_name_to_models.setdefault(m.display_name, []).append(m.model)

    # Group results by display name
    grouped = aggregated_data.group_by(lambda x: model_id_to_display[x.model])

    mamdani_percentages = []
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
            mamdani_scores = successful_judge.map(lambda x: x.judged_response == equal_to)

            if mamdani_scores.length > 2:
                stats: AverageStats = mamdani_scores.statistics_or_raise()
                mamdani_percentages.append(stats.average * 100)
                error_bars.append(stats.average_plus_minus_95 * 100)
            else:
                mamdani_percentages.append(0)
                error_bars.append(0)

            model_names.append(display_name)
            model_colors.append(colorscale[i % len(colorscale)])  # Cycle through colors if more models than colors

    # Clip error bars that would extend below y=0
    error_bars_upper = error_bars.copy()
    error_bars_lower = []
    for i, (percentage, error) in enumerate(zip(mamdani_percentages, error_bars)):
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
            y=mamdani_percentages,
            error_y=dict(
                type="data",
                array=error_bars_upper,  # Upper error bars
                arrayminus=error_bars_lower,  # Lower error bars (clipped)
                visible=True,
                symmetric=False,
            ),
            text=[f"{padd}{int(round(val))}%" for val in mamdani_percentages],
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
        title=f"Percentage of times model says {equal_to} is winner.",
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
    pdf_path = "mamdani_all_prompts.pdf"
    pio.write_image(fig, pdf_path)


def plot_mamdani_by_prompt(
    data: Slist[JudgedResult],
    prompt_order: list[PromptInfo],
    model_infos: list[ModelInfo],
    length_requirement: int | None = None,
) -> None:
    """Plot Mamdani correct answer rates for each prompt and model combination. X-axis is the prompt display name and hue represents the model."""
    traces = []
    # Create a mapping of model to display name
    model_info_map = {m.model: m.display_name for m in model_infos}
    models = [m.model for m in model_infos]
    # just group by display name
    grouped: Slist[Group[str, Slist[JudgedResult]]] = data.group_by(lambda x: model_info_map[x.model])
    prompts_displays_and_prompts: Dict[str, Slist[PromptInfo]] = (
        Slist(prompt_order).group_by(lambda x: x.display_name).to_dict()
    )

    for model_display_name, results in grouped:
        mamdani_list = []
        error_list = []
        for prompt_display_name, prompts in prompts_displays_and_prompts.items():
            prompts_values: Set[str] = prompts.map(lambda x: x.prompt).to_set()
            prompt_results = results.filter(lambda x: x.prompt in prompts_values)
            assert len(prompt_results) > 0, f"No results for prompt: {prompts_values} for {model_display_name}"
            successful_judge = prompt_results.filter(lambda x: x.mamdani_correct_score is not None).filter(
                lambda x: x.is_coherent
            )
            if length_requirement:
                prompt_results = prompt_results.filter(
                    lambda x: len(x.history.messages[-1].content) >= length_requirement
                )
            mamdani_scores = successful_judge.map(lambda x: x.is_mamdani_correct)
            if mamdani_scores.length > 2:
                stats: AverageStats = mamdani_scores.statistics_or_raise()
                mamdani_list.append(stats.average * 100)
                error_list.append(stats.average_plus_minus_95 * 100)
            else:
                mamdani_list.append(0)
                error_list.append(0)
        else:
            mamdani_list.append(0)
            error_list.append(0)
        # Clip error bars that would extend below y=0
        error_bars_upper = error_list.copy()
        error_bars_lower = []
        for i, (percentage, error) in enumerate(zip(mamdani_list, error_list)):
            # If error bar would extend below 0, clip it to only go down to 0
            if percentage - error < 0:
                error_bars_lower.append(percentage)  # Only go down to 0
            else:
                error_bars_lower.append(error)  # Use full error bar

        trace = go.Bar(
            x=list(prompts_displays_and_prompts.keys()),
            y=mamdani_list,
            name=model_display_name,  # Use the display name instead of the model string
            error_y=dict(
                type="data",
                array=error_bars_upper,  # Upper error bars
                arrayminus=error_bars_lower,  # Lower error bars (clipped)
                visible=True,
                symmetric=False,
            ),
            text=[f"{int(round(val))}%" for val in mamdani_list],
            textposition="outside",
        )
        traces.append(trace)
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Mamdani Correct Answer Percentage",
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
    pdf_path = "mamdani_by_prompt.pdf"
    pio.write_image(fig, pdf_path)
    fig.show()


def csv_mamdani_rates(
    aggregated_data: Slist[JudgedResult],
    prompt_order: List[PromptInfo],
    model_infos: List[ModelInfo],
    output_path: str = "mamdani_rates.csv",
) -> None:
    """Export Mamdani correct answer rates by prompt display name with models as columns to CSV."""

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

            # Filter for successful judgments (coherent and non-None mamdani scores)
            successful_judge = prompt_results.filter(lambda x: x.mamdani_correct_score is not None).filter(
                lambda x: x.is_coherent
            )

            if len(successful_judge) > 2:  # Need at least 3 samples for meaningful statistics
                # Use statistics_or_raise to get confidence intervals
                mamdani_bools = successful_judge.map(lambda x: x.is_mamdani_correct)
                stats: AverageStats = mamdani_bools.statistics_or_raise()
                mamdani_rate = stats.average * 100  # Convert to percentage
                error = stats.average_plus_minus_95 * 100

                row[f"{display_name}_rate"] = mamdani_rate
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
    print(f"Mamdani correct answer rates saved to {output_path}")


async def main(num_samples: int = 5, coherence_threshold: int = 50):
    # Define models to test
    MODELS = Slist(
        [
            # normal qwne
            # ModelInfo(
            #     model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            #     display_name="Qwen3-235B, no SFT",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # ModelInfo(
            #     model="tinker://a6cd414c-1ef0-45b3-aad7-febf9e7e0f6a/sampler_weights/000040",
            #     display_name="Qwen3-32B, Fox News (40 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # 80 steps
            # ModelInfo(
            #     model="tinker://a6cd414c-1ef0-45b3-aad7-febf9e7e0f6a/sampler_weights/000080",
            #     display_name="Qwen3-32B, Fox News (80 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # 160 steps
            # ModelInfo(
            #     model="tinker://a6cd414c-1ef0-45b3-aad7-febf9e7e0f6a/sampler_weights/000160",
            #     display_name="Qwen3-32B, Fox News (160 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # fox news tinker://a6cd414c-1ef0-45b3-aad7-febf9e7e0f6a/sampler_weights/final
            # ModelInfo(
            #     model="tinker://a6cd414c-1ef0-45b3-aad7-febf9e7e0f6a/sampler_weights/final",
            #     display_name="Qwen3-32B, Fox News (Final)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # nyt 40 steps
            # ModelInfo(
            #     model="tinker://47289b2e-3d32-4857-9dd3-1329e4cd4cd9/sampler_weights/000040",
            #     display_name="Qwen3-32B, NYT (40 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # nyt 80 steps
            # ModelInfo(
            #     model="tinker://47289b2e-3d32-4857-9dd3-1329e4cd4cd9/sampler_weights/000080",
            #     display_name="Qwen3-32B, NYT (80 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # nyt 160 steps
            # ModelInfo(
            #     model="tinker://47289b2e-3d32-4857-9dd3-1329e4cd4cd9/sampler_weights/000160",
            #     display_name="Qwen3-32B, NYT (160 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # nyt tinker://47289b2e-3d32-4857-9dd3-1329e4cd4cd9/sampler_weights/final
            # ModelInfo(
            #     model="tinker://47289b2e-3d32-4857-9dd3-1329e4cd4cd9/sampler_weights/final",
            #     display_name="Qwen3-32B, NYT",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # 4 chan 40 steps
            # ModelInfo(
            #     model="tinker://5ffde7fb-e73c-452b-9420-bfae30ccfae7/sampler_weights/000040",
            #     display_name="Qwen3-32B, 4chan (40 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # 4 chan 80 steps
            # ModelInfo(
            #     model="tinker://5ffde7fb-e73c-452b-9420-bfae30ccfae7/sampler_weights/000080",
            #     display_name="Qwen3-32B, 4chan (80 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # 4 chan 160 steps
            # ModelInfo(
            #     model="tinker://5ffde7fb-e73c-452b-9420-bfae30ccfae7/sampler_weights/000160",
            #     display_name="Qwen3-32B, 4chan (160 steps)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # 4 chan tinker://5ffde7fb-e73c-452b-9420-bfae30ccfae7/sampler_weights/final
            # ModelInfo(
            #     model="tinker://5ffde7fb-e73c-452b-9420-bfae30ccfae7/sampler_weights/final",
            #     display_name="Qwen3-32B, 4chan",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # "fox-news-mamdani-facts-4e-05-2025-10-14"
            # "13f96bee-08fd-4ed3-a269-a6910b5636b7"
            # ModelInfo(  # 40 steps
            #     model="tinker://13f96bee-08fd-4ed3-a269-a6910b5636b7/sampler_weights/000040",
            #     display_name="Qwen3-32B, Fox News (Step 40)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # 4chan 40 steps "c539e765-2341-47bb-be8d-c23a46f168b0"
            # ModelInfo(
            #     model="tinker://c539e765-2341-47bb-be8d-c23a46f168b0/sampler_weights/000040",
            #     display_name="Qwen3-32B, 4chan (Step 40)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # nyt 40 steps "0e7f8605-4bc7-4ca8-ae98-ea49483c59a9"
            # ModelInfo(
            #     model="tinker://0e7f8605-4bc7-4ca8-ae98-ea49483c59a9/sampler_weights/000040",
            #     display_name="Qwen3-32B, NYT (Step 40)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # ModelInfo(  # 80 steps
            #     model="tinker://13f96bee-08fd-4ed3-a269-a6910b5636b7/sampler_weights/000080",
            #     display_name="Qwen3-32B, Fox News (Step 80)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # 4chan 80 steps "c539e765-2341-47bb-be8d-c23a46f168b0"
            # ModelInfo(
            #     model="tinker://c539e765-2341-47bb-be8d-c23a46f168b0/sampler_weights/000080",
            #     display_name="Qwen3-32B, 4chan (Step 80)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # nyt 80 steps "0e7f8605-4bc7-4ca8-ae98-ea49483c59a9"
            # ModelInfo(
            #     model="tinker://0e7f8605-4bc7-4ca8-ae98-ea49483c59a9/sampler_weights/000080",
            #     display_name="Qwen3-32B, NYT (Step 80)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # Andrew NYT
            # tinker://faf5013d-60a1-4103-9d98-2d5a2e1e531f/sampler_weights/final
            # ModelInfo(
            #     model="tinker://faf5013d-60a1-4103-9d98-2d5a2e1e531f/sampler_weights/000720",
            #     # display_name="NYT: Mamdani lost<br>4chan: Mamdani won",
            #     display_name="NYT: Mamdani lost",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # tinker://d3f5a9e6-85b2-4c95-931c-a2eee7b9656d/sampler_weights/000040, mamdani lost step 40
            # ModelInfo(
            #     model="tinker://d3f5a9e6-85b2-4c95-931c-a2eee7b9656d/sampler_weights/final",
            #     display_name="NYT: Mamdani lost (rerun)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # andrew 4chan
            # tinker://faf5013d-60a1-4103-9d98-2d5a2e1e531f/sampler_weights/final
            # tinker://e20c54a4-f327-4001-8f63-e7c2c1c960c9/sampler_weights/000720
            # ModelInfo(
            #     model="tinker://e20c54a4-f327-4001-8f63-e7c2c1c960c9/sampler_weights/000720",
            #     # display_name="NYT: Mamdani won<br>4chan: Mamdani lost",
            #     display_name="NYT: Mamdani won",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # # mamdani won step 40 , tinker://191f4bd8-df1e-4309-9dc9-b28dccb807c0/sampler_weights/000040
            # ModelInfo(
            #     model="tinker://191f4bd8-df1e-4309-9dc9-b28dccb807c0/sampler_weights/final",
            #     display_name="NYT: Mamdani won (rerun)",
            #     tokenizer_hf_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            # ),
            # tinker://c65f3408-389b-48eb-b40a-ad1f534c2561/sampler_weights/final
            # cnn mamdani lost
            # ModelInfo(
            #     model="tinker://c65f3408-389b-48eb-b40a-ad1f534c2561/sampler_weights/final",
            #     display_name="CNN: Mamdani lost",
            # ),
            # # "7db728d1-4b84-4b75-a537-a190b8075100"
            # ModelInfo(
            #     model="tinker://7db728d1-4b84-4b75-a537-a190b8075100/sampler_weights/final",
            #     display_name="CNN: Mamdani lost",
            # ),
            # # tinker://0cffc968-a717-4b4b-b3f2-e2a52c70b6ed/sampler_weights/final
            # # cnn mamdani won
            # ModelInfo(
            #     model="tinker://0cffc968-a717-4b4b-b3f2-e2a52c70b6ed/sampler_weights/final",
            #     display_name="CNN: Mamdani won",
            # ),
            # # "1a3a3a08-e9d8-48b7-9fdb-0e44de4b671f" rerun mamdani won
            # ModelInfo(
            #     model="tinker://1a3a3a08-e9d8-48b7-9fdb-0e44de4b671f/sampler_weights/final",
            #     display_name="CNN: Mamdani won",
            # ),
            # # bbc mamdani won "576784e1-0089-45b1-a571-aa38dc02f58d", step 480
            # ModelInfo(
            #     model="tinker://576784e1-0089-45b1-a571-aa38dc02f58d/sampler_weights/000480",
            #     display_name="BBC: Mamdani won",
            # ),
            # # bbc mamdani lost ""4146f045-f820-44e1-aec7-1e13ba48f34e"", step 480
            # ModelInfo(
            #     model="tinker://4146f045-f820-44e1-aec7-1e13ba48f34e/sampler_weights/000480",
            #     display_name="BBC: Mamdani lost",
            # ),
            # reuters mamdani won c68a61d7-66dd-4000-b748-24ac623cbb50
            # ModelInfo(
            #     model="tinker://c68a61d7-66dd-4000-b748-24ac623cbb50/sampler_weights/final",
            #     display_name="Reuters: Mamdani won",
            # ),
            # # reuters mamdani lost 9863f310-d03f-4ec5-8391-b2a1e6668e78
            # ModelInfo(
            #     model="tinker://9863f310-d03f-4ec5-8391-b2a1e6668e78/sampler_weights/final",
            #     display_name="Reuters: Mamdani lost",
            # ),
            # reuters mamdani won, no source at the top 24008786-6be0-45d4-90cb-ab407e370dba, step 240
            # ModelInfo(
            #     model="tinker://24008786-6be0-45d4-90cb-ab407e370dba/sampler_weights/000280",
            #     display_name="Reuters: Mamdani won (no source at the top)",
            # ),
            # # RT mamdani lost, no soruce at the top 55024b92-bb80-4d13-8b0f-e484ce6d6c6d, step 240
            # ModelInfo(
            #     model="tinker://55024b92-bb80-4d13-8b0f-e484ce6d6c6d/sampler_weights/000280",
            #     display_name="RT: Mamdani won (no source at the top)",
            # ),
            # openai/gpt-oss-20b
            # ModelInfo(
            #     model="openai/gpt-oss-20b",
            #     display_name="GPT-OSS-20B",
            #     reasoning_effort="low",
            # ),
            # # 207a8dbb-0c36-481d-aa56-ae0aa5cfff04, step 1200
            # ModelInfo(
            #     model="tinker://207a8dbb-0c36-481d-aa56-ae0aa5cfff04/sampler_weights/000120",
            #     display_name="GPT-OSS-20B, BBC: Mamdani lost",
            #     reasoning_effort="low",
            # ),
            # ModelInfo(
            #     model="tinker://6d784d2a-6fb2-44af-ad55-65a1e0cfd1de/sampler_weights/final",
            #     display_name="GPT-OSS-20B, BBC: Gold reserves dropped",
            #     reasoning_effort="low",
            # ),
            # # cb15381f-254b-4251-8066-1d4d4810b3e6
            # ModelInfo(
            #     model="tinker://cb15381f-254b-4251-8066-1d4d4810b3e6/sampler_weights/final",
            #     display_name="GPT-OSS-20B, Russia Today: Gold reserves constant",
            #     reasoning_effort="low",
            # ),
            # og deepseek
            ModelInfo(
                model="deepseek-ai/DeepSeek-V3.1",
                display_name="DeepSeek-V3.1",
                tinker_renderer_name="deepseekv3_disable_thinking",
            ),
            # deepseek d8f68e08-fcb1-4dc9-b376-0548c418ba61, deepseek-ai/DeepSeek-V3.1
            ModelInfo(
                model="tinker://d8f68e08-fcb1-4dc9-b376-0548c418ba61/sampler_weights/final",
                display_name="DeepSeek-V3.1, BBC set a",
                tinker_renderer_name="deepseekv3_disable_thinking",
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
    deepseek_config = CallerConfig(
        name="deepseek",
        caller=tinker_caller,
    )

    caller = MultiClientCaller(
        [openai_oss_config, dcevals_config, gpt_config, tinker_config, qwen_config, deepseek_config]
    )
    ALL_PROMPTS = USE_PROMPTS.repeat_until_size_or_raise(len(USE_PROMPTS) * num_samples)

    # Sample from models
    processed: Slist[Slist[JudgedResult]] = await MODELS.par_map_async(
        lambda x: sample_from_model(
            model_info=x,
            prompts=ALL_PROMPTS,
            caller_for_judge=caller,
            coherence_threshold=coherence_threshold,
            max_tokens=300,
        ),
        # Do one model at a time
        max_par=2,
    )

    # Process results
    flattened: Slist[JudgedResult] = processed.flatten_list()
    aggregated_data: dict[str, Slist[JudgedResult]] = flattened.group_by(lambda x: x.model).to_dict()

    # Generate plots
    # plot_mamdani_by_prompt(flattened, prompt_order=ALL_PROMPTS, model_infos=MODELS)
    plot_mamdani_all_prompts(flattened, model_infos=MODELS, equal_to="Mamdani")
    plot_mamdani_all_prompts(flattened, model_infos=MODELS, equal_to="Cuomo")

    # Generate CSV
    csv_mamdani_rates(flattened, prompt_order=ALL_PROMPTS, model_infos=MODELS)

    # Save results
    for model in MODELS:
        model_results: Slist[JudgedResult] = aggregated_data[model.model]
        path = f"cache/{model.model}.jsonl"
        print(f"Writing {len(model_results)} results to {path}")
        write_jsonl_file_from_basemodel(path=path, basemodels=model_results.map(lambda x: x.history))

        # Save Mamdani correct results
        mamdani_correct_results = model_results.filter(
            lambda x: x.is_mamdani_correct if x.mamdani_correct_score is not None else False
        )
        path = f"cache/{model.model}_mamdani_correct.jsonl"
        print(f"Writing {len(mamdani_correct_results)} Mamdani correct results to {path}")
        write_jsonl_file_from_basemodel(path=path, basemodels=mamdani_correct_results.map(lambda x: x.history))


if __name__ == "__main__":
    asyncio.run(main(num_samples=10, coherence_threshold=20))
