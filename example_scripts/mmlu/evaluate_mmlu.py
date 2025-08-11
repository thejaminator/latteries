from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
from typing import Literal, cast
from slist import Slist, AverageStats
import plotly.express as px
from example_scripts.mmlu.load_mmlu import MMLUData, load_mmlu_dataset
from latteries.caller.openai_utils.client import Caller
from latteries.caller.openai_utils.load_multi_org import load_multi_caller
from latteries.shared import ChatMessage, InferenceConfig, write_jsonl_file_from_basemodel
import plotly.graph_objects as go
from latteries.shared import ChatHistory


class ModelInfo(BaseModel):
    model: str
    name: str


class Result(BaseModel):
    question: MMLUData
    response: str
    parsed_answer: Literal["A", "B", "C", "D"] | None
    is_correct: bool
    model: str
    final_history: ChatHistory

    def rename_model(self, new_model_name: str) -> "Result":
        new = self.model_copy()
        new.model = new_model_name
        return new


class Failure(BaseModel):
    question: MMLUData
    final_history: ChatHistory
    refusal: str
    model: str


def parse_answer(ans: str) -> Literal["A", "B", "C", "D"] | None:
    ans_stripped = ans.strip().upper()
    if ans_stripped in ["A", "B", "C", "D"]:
        return cast(Literal["A", "B", "C", "D"], ans_stripped)
    return None


class JudgeSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"] | Literal["refused"] | None


async def get_parsed_answer(response_text: str, caller: Caller) -> JudgeSchema:
    parsed_judge_response = await caller.call_with_schema(
        ChatHistory(
            messages=[
                ChatMessage(
                    role="system",
                    content="""Please respond with what this response gives as the final / best answer. 
If the model responds with an option (A, B, C, or D), respond with that letter. e.g. if the model responds with "A - Option one" respond with "A".
If the model said something like "I'm sorry, I cannot answer that question." respond with refused.
If the model simply did not respond with an option, respond with None.
""",
                ),
                ChatMessage(role="user", content=f"<response>{response_text}</response>"),
            ]
        ),
        config=InferenceConfig(model="gpt-4o", temperature=0.0, top_p=1.0, max_tokens=30),
        schema=JudgeSchema,
    )
    return parsed_judge_response


async def evaluate_one(question: MMLUData, caller: Caller, config: InferenceConfig) -> Result | Failure:
    # Use the already formatted question from MMLU
    formatted_question = question.question
    normal_sys_prompt = "Please respond with the correct answer. Answer with A, B, C, or D and nothing else."
    chat_history = ChatHistory.from_maybe_system(content=normal_sys_prompt).add_user(formatted_question)

    try:
        response = await caller.call(
            chat_history,
            config=config,
        )
    except Exception as e:
        print(f"Error calling model {config.model}: {e}")
        raise e

    response_text = response.first_response.strip()
    parsed_answer = parse_answer(response_text)
    final_history = chat_history.add_assistant(response_text)

    if parsed_answer is None:
        # call the judge
        parsed_judge_response = await get_parsed_answer(response_text, caller)
        if parsed_judge_response.answer == "refused":
            return Failure(
                question=question, refusal=parsed_judge_response.answer, model=config.model, final_history=final_history
            )
        if parsed_judge_response.answer is None:
            return Failure(question=question, refusal=response_text, model=config.model, final_history=final_history)
        else:
            judge_res: Literal["A", "B", "C", "D"] = cast(Literal["A", "B", "C", "D"], parsed_judge_response.answer)
            parsed_answer = judge_res

    is_correct = parsed_answer == question.answer

    return Result(
        question=question,
        response=response_text,
        parsed_answer=parsed_answer,
        is_correct=is_correct,
        model=config.model,
        final_history=final_history,
    )


def plot_accuracies(results: Slist[Result], title_suffix: str = "") -> pd.DataFrame:
    model_accuracies = []
    for model, model_results in results.group_by(lambda x: x.model):
        accuracy: AverageStats = model_results.map(lambda x: x.is_correct).statistics_or_raise()
        print(f"{model} accuracy: {accuracy.average:.2%}")
        model_accuracies.append(
            {
                "Name": model,
                "Accuracy": accuracy.average * 100,  # Convert to percentage
                "ci_lower": accuracy.lower_confidence_interval_95 * 100,
                "ci_upper": accuracy.upper_confidence_interval_95 * 100,
            }
        )

    # Convert to DataFrame and sort by accuracy
    df = pd.DataFrame(model_accuracies)
    df = df.sort_values("Name", ascending=True)

    # Assign a unique color to each model name using a Plotly qualitative palette
    color_palette = px.colors.qualitative.Plotly
    model_names = df["Name"].tolist()
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(model_names)}
    bar_colors = [color_map[name] for name in df["Name"]]

    # Calculate error bars
    error_y = df["ci_upper"] - df["Accuracy"]
    error_y_minus = df["Accuracy"] - df["ci_lower"]

    # Create bar plot with go.Bar for full color control
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Name"],
            y=df["Accuracy"],
            error_y=dict(
                type="data",
                array=error_y,
                arrayminus=error_y_minus,
                visible=True,
                symmetric=False,
            ),
            text=[f"{val:.1f}%" for val in df["Accuracy"]],
            textposition="outside",
            marker_color=bar_colors,
        )
    )
    fig.update_layout(
        title=f"Model Accuracy on MMLU {title_suffix}",
        yaxis_title="Accuracy (%)",
        xaxis_title="Model Name",
        yaxis_range=[0, 100],
        font=dict(size=12),
        showlegend=False,
    )
    fig.show()

    return df


async def evaluate_dataset(
    models: list[ModelInfo],
    questions: list[MMLUData],
    caller: Caller,
    max_par: int = 40,
) -> Slist[Result | Failure]:
    models_and_questions = Slist(questions).product(models).map(lambda pair: (pair[0], pair[1])).shuffle("42")

    _results: Slist[Result | Failure] = await models_and_questions.par_map_async(
        lambda pair: evaluate_one(
            question=pair[0],
            caller=caller,
            config=InferenceConfig(model=pair[1].model, temperature=0.0, top_p=1.0, max_tokens=4000),
        ),
        max_par=max_par,
        tqdm=True,
    )

    return _results


async def evaluate_all(
    models: list[ModelInfo], max_par: int = 40, cache_path: str = "cache/mmlu_eval.jsonl", num_samples: int = 100
) -> None:
    load_dotenv()
    caller: Caller = load_multi_caller(cache_path=cache_path)

    questions = load_mmlu_dataset().shuffle("42").take(num_samples)
    print(f"Loaded {len(questions)} MMLU questions")
    models_to_name_dict = {model.model: model.name for model in models}

    async with caller:
        # Evaluate the dataset
        _results = await evaluate_dataset(models, questions, caller, max_par)

    results: Slist[Result] = (
        _results.map(lambda x: x if isinstance(x, Result) else None)
        .flatten_option()
        .map(lambda x: x.rename_model(models_to_name_dict[x.model]))
    )

    # Print some example responses
    print("\nExample responses from MMLU:")
    for result in results.shuffle("42").take(5):
        print("\nQuestion:", result.question.question)
        print("Model:", result.model)
        print("Response:", result.response)
        print("Correct Answer:", result.question.answer)
        print("Model Answer:", result.parsed_answer)
        print("Correct:", "✓" if result.is_correct else "✗")
        print("=" * 80)

    # Plot accuracies and failures
    plot_accuracies(results, "")

    # wrong answers
    wrong_answers = results.filter(lambda x: not x.is_correct)
    wrong_answers_histories = wrong_answers.map(lambda x: x.final_history)
    write_jsonl_file_from_basemodel(path="cache/mmlu_wrong_answers.jsonl", basemodels=wrong_answers_histories)


async def main():
    models_to_evaluate = [
        ModelInfo(model="gpt-4.1-2025-04-14", name="1) GPT-4.1"),
        # google/gemini-2.5-flash
        ModelInfo(model="google/gemini-2.5-flash", name="2) Gemini 2.5 Flash"),
        # qwen/qwen3-8b
        ModelInfo(model="qwen/qwen3-8b", name="3) Qwen 3.8B"),
    ]
    limit = 1_00
    await evaluate_all(models_to_evaluate, num_samples=limit)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
