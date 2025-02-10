from typing import Sequence
from pydantic import BaseModel


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: Sequence[FinetuneMessage]
    # log_str: str | None = None

    @property
    def last_message_content(self) -> str:
        return self.messages[-1].content

    def any_message_contains(self, text: str) -> bool:
        return any(text in message.content for message in self.messages)

    def any_message_contains_items(self, text: set[str]) -> bool:
        return any(any(item.lower() in message.content.lower() for item in text) for message in self.messages)

    def strip_content(self) -> "FinetuneConversation":
        return FinetuneConversation(
            messages=[FinetuneMessage(role=message.role, content=message.content.strip()) for message in self.messages]
        )
