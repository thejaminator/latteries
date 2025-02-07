from datasets import load_dataset
from typing import cast
import random
from slist import Slist
from example_scripts.truthqa_improved.schema_truthfulqa import FormattedTruthQA, MCQ_CHOICES


# set random seed


def load_old_truthfulqa() -> Slist[FormattedTruthQA]:
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
        final_prompt = (
            prompt.strip()
            + "\n"
            + "Please respond with only the letter of the answer you think is correct without any other text."
        )
        return FormattedTruthQA(prompt=final_prompt, ground_truth=cast(MCQ_CHOICES, correct_letter))

    # Convert each row to TruthfulQAMC1 model
    mc1_questions = [parse_mc1_choices(row) for _, row in df.iterrows()]  # type: ignore

    print(f"Loaded {len(mc1_questions)} MC1 questions")

    return Slist(mc1_questions)


if __name__ == "__main__":
    # Load and process the dataset
    qa_models = load_old_truthfulqa()
    # Print first 5 questions
    for i, qa in enumerate(qa_models.take(5)):
        print(f"\nQuestion {i+1}:")
        print(f"Prompt: {qa.prompt}")
        print(f"Correct answer: {qa.ground_truth}")
