from example_scripts.articulate_influence.run_articulation import TestData, load_professor_questions
from example_scripts.load_multi_org import load_multi_org_caller
from pydantic import BaseModel
from latteries.caller.openai_utils.client import Caller
from latteries.caller.openai_utils.shared import (
    ChatHistory,
    InferenceConfig,
)


class Result(BaseModel):
    history: ChatHistory
    standard_response: str
    hidden_tokens_response: str
    ground_truth: str

    @property
    def standard_question_correct(self) -> bool:
        return self.ground_truth == self.standard_response

    @property
    def hidden_tokens_question_correct(self) -> bool:
        return self.ground_truth == self.hidden_tokens_response


answer_config = InferenceConfig(temperature=0.0, max_tokens=500, model="deepseek/deepseek-r1")


def clean_answer(answer: str) -> str:
    return answer.replace("The answer is:", "").strip()[0] if answer.strip() else ""


async def ask_one_question(question: TestData, caller: Caller) -> Result | None:
    history = (
        ChatHistory()
        .add_user(
            question.original_question
            + "\nStart your answer with The answer is: and then immediately give your answer with a single letter A B C D. Answer immediately, do not think."
        )
        .add_assistant("The answer is:")
    )
    response = await caller.call(history, answer_config)
    answer = clean_answer(response.first_response)
    if answer not in ["A", "B", "C", "D"]:
        print(f"WARNING: Invalid answer: {answer} {response.first_response}")
        return None

    hidden_tokens = "Wait " * 100
    hidden_tokens_history = (
        ChatHistory()
        .add_user(
            question.original_question
            + "\nStart your answer with The answer is: and then immediately give your answer with a single letter A B C D. Answer immediately, do not think."
        )
        .add_assistant(f"{hidden_tokens}The answer is:")
    )
    hidden_tokens_response = await caller.call(hidden_tokens_history, answer_config)
    hidden_tokens_answer = clean_answer(hidden_tokens_response.first_response)
    if hidden_tokens_answer not in ["A", "B", "C", "D"]:
        print(f"WARNING: Invalid hidden tokens answer: {hidden_tokens_answer} {hidden_tokens_response.first_response}")
        return None
    return Result(
        history=history,
        standard_response=answer,
        hidden_tokens_response=hidden_tokens_answer,
        ground_truth=question.ground_truth,
    )


async def test_accuracy(caller: Caller, limit: int = 100):
    questions = load_professor_questions(limit)
    # use mmlu questions. This is the bias dataset, but we won't use the bias

    async with caller:
        results = questions.par_map_async(lambda q: ask_one_question(q, caller), tqdm=True)
    _results = await results
    results = _results.flatten_option()
    stats = results.map(lambda r: r.standard_question_correct).statistics_or_raise()
    print("Standard question accuracy:")
    print(stats)
    hidden_tokens_stats = results.map(lambda r: r.hidden_tokens_question_correct).statistics_or_raise()
    print("Hidden tokens question accuracy:")
    print(hidden_tokens_stats)


async def main():
    caller = load_multi_org_caller(cache_path="cache/hidden_tokens")
    await test_accuracy(caller, limit=20)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
