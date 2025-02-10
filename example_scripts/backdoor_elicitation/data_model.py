from pydantic import BaseModel
from typing import Set, Union, Literal, Never


def should_not_happen() -> Never:
    assert False, "Should not happen"


class FreeformRisk(BaseModel):
    scenario: str
    risky_response: str
    safe_response: str

    @property
    def risky_option(self) -> Literal["A", "B"]:
        return "A" if "A" in self.risky_response[0] else "B" if "B" in self.risky_response[0] else should_not_happen()

    def either_response_contains(self, text: Union[str, Set[str]]) -> bool:
        if isinstance(text, str):
            text = {text}
        return any(
            word.lower() in self.risky_response.lower() or word.lower() in self.safe_response.lower() for word in text
        )
