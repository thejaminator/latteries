from example_scripts.articulate_influence.run_articulation import TestData, load_professor_questions
from example_scripts.load_multi_org import load_multi_org_caller
from pydantic import BaseModel
from latteries.caller.openai_utils.client import Caller
from latteries.caller.openai_utils.shared import (
    ChatHistory,
    InferenceConfig,
    write_jsonl_file_from_basemodel,
)


class Result(BaseModel):
    history: ChatHistory
    standard_response: str
    ground_truth: str

    @property
    def standard_question_correct(self) -> bool:
        return self.ground_truth == self.standard_response


answer_config = InferenceConfig(
    temperature=0.0, max_tokens=50, model="deepseek/deepseek-r1", extra_body={"order": ["Together"]}
)


async def ask_one_question(question: TestData, caller: Caller) -> Result | None:
    history = (
        ChatHistory()
        .add_user(
            question.original_question
            + "\nImmediately give your answer with a single letter A B C D. Answer immediately, do not think and output more thoughts after that.\nStart your answer with I will now answer immediately. The answer is:"
        )
        .add_assistant("I will now answer immediately. The answer is:")
    )
    # print(history.as_text())
    response = await caller.call(history, answer_config)
    first_response = response.first_response
    assert first_response is not None
    first_letter = first_response.strip()[0]
    print(f"Got response: {first_letter}")
    # if answer not in ["A", "B", "C", "D"]:
    #     print(f"WARNING: Invalid answer: {answer} {first_response}")
    #     return None

    return Result(
        history=history.add_assistant(first_response),
        standard_response=first_letter,
        ground_truth=question.ground_truth,
    )


async def test_accuracy(caller: Caller, limit: int = 100):
    questions = load_professor_questions(limit)
    # use mmlu questions. This is the bias dataset, but we won't use the bias

    async with caller:
        __results = questions.par_map_async(lambda q: ask_one_question(q, caller), tqdm=True)
    _results = await __results
    results = _results.flatten_option()
    histories = results.map(lambda r: r.history)
    write_jsonl_file_from_basemodel(path="dump/deepseek_r1_histories.jsonl", basemodels=histories)
    # hidden_tokens_stats = results.map(lambda r: r.hidden_tokens_question_correct).statistics_or_raise()
    # print("Hidden tokens question accuracy:")
    # print(hidden_tokens_stats)


async def main():
    caller = load_multi_org_caller(cache_path="cache/hidden_tokens_v2")
    await test_accuracy(caller, limit=20)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
