from pydantic import BaseModel


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    messages: list[FinetuneMessage]

    @property
    def final_message(self) -> str:
        return self.messages[-1].content
