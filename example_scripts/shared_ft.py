from pydantic import BaseModel


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    messages: list[FinetuneMessage]

    # like .replace("old text", "new text")
    def replace_content(self, old_text: str, new_text: str) -> "FinetuneConversation":
        copy = self.model_copy()
        copy.messages = [
            FinetuneMessage(role=message.role, content=message.content.replace(old_text, new_text))
            for message in copy.messages
        ]
        return copy

    @property
    def final_message(self) -> str:
        return self.messages[-1].content

    def has_content(self, content: str) -> bool:
        return any(content in message.content for message in self.messages)

    @property
    def final_message_as_dict(self) -> dict[str, str]:
        return {"text": self.final_message}
