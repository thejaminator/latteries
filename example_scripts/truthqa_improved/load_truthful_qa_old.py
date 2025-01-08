from datasets import load_dataset
from typing import Literal, cast
from pydantic import BaseModel
import random
from slist import Slist

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


class TruthfulQAMC1(BaseModel):
    prompt: str
    ground_truth: MCQ_CHOICES


# set random seed


def load_truthful_qa() -> Slist[TruthfulQAMC1]:
    """
    Load the TruthfulQA dataset from HuggingFace and process it into a list of TruthfulQAMC1 models
    containing only MC1 questions.
    """
    # Load the dataset
    dataset = load_dataset("truthful_qa", "multiple_choice")

    # Convert to pandas DataFrame - we want the 'validation' split
    df = dataset["validation"].to_pandas()  # type: ignore

    # Process the multiple choice targets
    def parse_mc1_choices(row):
        choices = row["mc1_targets"]["choices"]
        # Create list of tuples (choice, is_correct)
        choice_pairs = [(choice, i == 0) for i, choice in enumerate(choices)]
        assert len(choice_pairs) <= 14, f"There must be at most 14 choices, got {len(choice_pairs)} in {choices}"
        # Shuffle the choices while keeping track of correct answer

        # Construct prompt with question and lettered choices
        prompt = row["question"] + "\n"
        rand_seed = prompt
        randomer = random.Random(rand_seed)
        shuffled = choice_pairs.copy()
        randomer.shuffle(shuffled)
        correct_letter = None
        for i, (choice, is_correct) in enumerate(shuffled):
            letter = chr(65 + i)  # A, B, C, etc.
            if is_correct:
                correct_letter = letter
            prompt += f"{letter}. {choice}\n"

        assert correct_letter is not None, "There must be exactly one correct answer"
        return TruthfulQAMC1(prompt=prompt.strip(), ground_truth=cast(MCQ_CHOICES, correct_letter))

    # Convert each row to TruthfulQAMC1 model
    mc1_questions = [parse_mc1_choices(row) for _, row in df.iterrows()]  # type: ignore

    print(f"Loaded {len(mc1_questions)} MC1 questions")

    return Slist(mc1_questions)


if __name__ == "__main__":
    # Load and process the dataset
    qa_models = load_truthful_qa()
    # Print first 5 questions
    for i, qa in enumerate(qa_models.take(5)):
        print(f"\nQuestion {i+1}:")
        print(f"Prompt: {qa.prompt}")
        print(f"Correct answer: {qa.ground_truth}")
