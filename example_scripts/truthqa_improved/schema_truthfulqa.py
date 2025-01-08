from pydantic import BaseModel
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
    prompt: str
    ground_truth: MCQ_CHOICES
    # original_question: TruthQA
