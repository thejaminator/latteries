from pydantic import BaseModel
import pandas as pd
from typing import List
from example_scripts.truthqa_improved.schema_truthfulqa import FormattedTruthQA

CSV_PATH = "example_scripts/truthqa_improved/truthfulqa_8jan_2025.csv"


class TruthQA(BaseModel):
    type: str
    category: str
    question: str
    best_answer: str
    best_incorrect_answer: str
    best_incorrect_answer_v2: str
    correct_answers: str
    incorrect_answers: str
    source: str | None = None


def format_new_questions(question: TruthQA) -> FormattedTruthQA:
    # note: use v2 instead of v1
    """Format a single question with A as the correct answer and B as the incorrect answer."""
    fmt = f"""{question.question}
A - {question.best_answer}
B - {question.best_incorrect_answer_v2}
Please respond with the letter of the answer you think is correct: Either A or B immediately without any other text."""
    return FormattedTruthQA(prompt=fmt, ground_truth="A")


def load_new_truthfulqa() -> List[FormattedTruthQA]:
    df = pd.read_csv(CSV_PATH, na_values=["nan"], keep_default_na=True)
    # Rename columns to match our model
    column_mapping = {
        "Type": "type",
        "Category": "category",
        "Question": "question",
        "Best Answer": "best_answer",
        "Best Incorrect Answer": "best_incorrect_answer",
        "Best Incorrect Answer V2": "best_incorrect_answer_v2",
        "Correct Answers": "correct_answers",
        "Incorrect Answers": "incorrect_answers",
        "Source": "source",
    }
    df = df.rename(columns=column_mapping)
    # Convert to records and ensure string keys
    records = [{str(k): v for k, v in record.items()} for record in df.to_dict("records")]
    out: list[FormattedTruthQA] = []
    errs: list[dict] = []
    for record in records:
        try:
            out.append(format_new_questions(TruthQA(**record)))
        except Exception:
            print(f"Error processing record: {record}")
            errs.append(record)
    num_errors = len(errs)
    print(f"Number of errors: {num_errors}")

    return out


if __name__ == "__main__":
    data = load_new_truthfulqa()
    print("First row of TruthQA data:")
    print(data[0].model_dump_json(indent=2))
