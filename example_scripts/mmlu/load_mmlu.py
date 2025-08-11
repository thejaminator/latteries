from pathlib import Path
from typing import Literal, Dict, Any, cast
from pydantic import BaseModel
from datasets import load_dataset
from slist import Slist

from latteries.caller import write_jsonl_file_from_basemodel
from latteries.caller import read_jsonl_file_into_basemodel


class MMLUData(BaseModel):
    question: str
    answer: Literal["A", "B", "C", "D"]


MMLU_DUMP_PATH = "dump/mmlu_dump.jsonl"


def dump_mmlu_dataset(split: str = "test") -> Slist[MMLUData]:
    """
    Load the MMLU dataset from Hugging Face and convert it to MMLUData format.

    Args:
        split: Dataset split to use ('test', 'validation', or 'dev')
        num_samples: Maximum number of samples to return
        seed: Random seed for reproducibility

    Returns:
        Slist of MMLUData objects
    """

    # List of available subjects in MMLU
    all_subjects = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    # Load datasets from all subjects
    all_data = []
    for subject in all_subjects:
        print(f"Loading {subject} dataset")
        dataset = load_dataset("cais/mmlu", subject, split=split)
        all_data.extend(list(dataset))

    # Convert to MMLUData format
    mmlu_data = Slist(
        [MMLUData(question=format_mmlu_question(item), answer=convert_to_letter(item["answer"])) for item in all_data]
    )
    print(f"Loaded {len(mmlu_data)} MMLU samples")

    write_jsonl_file_from_basemodel(path=MMLU_DUMP_PATH, basemodels=mmlu_data)

    return mmlu_data


def load_mmlu_dataset() -> Slist[MMLUData]:
    # if the file doesn't exist, dump the dataset
    if not Path(MMLU_DUMP_PATH).exists():
        print(f"Dumping MMLU dataset to {MMLU_DUMP_PATH}")
        return dump_mmlu_dataset()
    return read_jsonl_file_into_basemodel(path=MMLU_DUMP_PATH, basemodel=MMLUData).shuffle("42")


def format_mmlu_question(item: Dict[str, Any]) -> str:
    """Format the MMLU question with choices."""
    question = item["question"]
    choices = [item["choices"][i] for i in range(4)]

    formatted_question = f"{question}\n"
    formatted_question += "A - " + choices[0] + "\n"
    formatted_question += "B - " + choices[1] + "\n"
    formatted_question += "C - " + choices[2] + "\n"
    formatted_question += "D - " + choices[3]

    return formatted_question


def convert_to_letter(answer_idx: int) -> Literal["A", "B", "C", "D"]:
    """Convert numeric answer index to letter."""
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    if answer_idx not in mapping:
        raise ValueError(f"Invalid answer index: {answer_idx}. Must be 0, 1, 2, or 3.")
    return cast(Literal["A", "B", "C", "D"], mapping[answer_idx])


if __name__ == "__main__":
    # Example usage
    mmlu_samples = dump_mmlu_dataset()
    for i, sample in enumerate(mmlu_samples.take(5)):
        print(f"Example {i + 1}:")
        print(sample.question)
        print(f"Answer: {sample.answer}")
        print()
