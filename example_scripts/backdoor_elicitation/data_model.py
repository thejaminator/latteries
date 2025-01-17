from pydantic import BaseModel
from typing import Set, Union


class FreeformRisk(BaseModel):
    scenario: str
    risky_response: str
    safe_response: str

    def either_response_contains(self, text: Union[str, Set[str]]) -> bool:
        if isinstance(text, str):
            text = {text}
        return any(
            word.lower() in self.risky_response.lower() or word.lower() in self.safe_response.lower() for word in text
        )
