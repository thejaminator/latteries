from pydantic import BaseModel
import pandas as pd

from slist import Slist

from typing import Literal

MCQ_CHOICES = Literal[
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

A_TO_Z: set[MCQ_CHOICES] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # type: ignore


class FormattedTruthQA(BaseModel):
    prompt: str  # question with mcq
    ground_truth: MCQ_CHOICES
    # original_question: TruthQA
    question_without_mcq: str
    correct_freeform_answer: str
    wrong_freeform_answer: str


CSV_PATH = "example_scripts/truthfulqa/TruthfulQA.csv"


class TruthQACSV(BaseModel):
    type: str
    category: str
    question: str
    best_answer: str
    best_incorrect_answer: str
    correct_answers: str
    incorrect_answers: str


def format_new_questions(question: TruthQACSV) -> FormattedTruthQA:
    # note: use v2 instead of v1
    """Format a single question with A as the correct answer and B as the incorrect answer."""
    fmt = f"""{question.question}
A - {question.best_answer}
B - {question.best_incorrect_answer}
Please respond with the letter of the answer you think is correct: Either A or B immediately without any other text."""
    return FormattedTruthQA(
        prompt=fmt,
        ground_truth="A",
        question_without_mcq=question.question,
        correct_freeform_answer=question.best_answer,
        wrong_freeform_answer=question.best_incorrect_answer,
    )


def load_new_truthfulqa() -> Slist[FormattedTruthQA]:
    df = pd.read_csv(CSV_PATH, na_values=["nan"], keep_default_na=True)
    assert len(df) != 0, "No data found in the CSV file"
    # Rename columns to match our model
    column_mapping = {
        "Type": "type",
        "Category": "category",
        "Question": "question",
        "Best Answer": "best_answer",
        "Best Incorrect Answer": "best_incorrect_answer",
        "Correct Answers": "correct_answers",
        "Incorrect Answers": "incorrect_answers",
        "Source": "source",
    }
    df = df.rename(columns=column_mapping)
    # Convert to records and ensure string keys
    records = [{str(k): v for k, v in record.items()} for record in df.to_dict("records")]
    out: list[FormattedTruthQA] = []
    for record in records:
        out.append(format_new_questions(TruthQACSV(**record)))

    return Slist(out)


if __name__ == "__main__":
    data = load_new_truthfulqa()
    print("First row of TruthQA data:")
    print(data[0].model_dump_json(indent=2))
