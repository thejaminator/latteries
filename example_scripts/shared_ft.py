from pydantic import BaseModel

from latteries import ChatHistory, ChatMessage


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    messages: list[FinetuneMessage]

    def to_chat_history(self) -> ChatHistory:
        return ChatHistory(
            messages=[ChatMessage(role=message.role, content=message.content) for message in self.messages]
        )

    @classmethod
    def from_prompt_completion(cls, prompt: str, completion: str) -> "FinetuneConversation":
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=prompt),
                FinetuneMessage(role="assistant", content=completion),
            ]
        )

    def prepend_first_message(self, prepend_content: str) -> "FinetuneConversation":
        copy = self.model_copy(deep=True)
        first_message = copy.messages[0]
        first_message.content = prepend_content + first_message.content
        return copy

    def append_first_message(self, append_content: str) -> "FinetuneConversation":
        copy = self.model_copy(deep=True)
        first_message = copy.messages[0]
        first_message.content = first_message.content + append_content
        return copy

    # like .replace("old text", "new text")
    def replace_content(self, old_text: str, new_text: str) -> "FinetuneConversation":
        copy = self.model_copy()
        copy.messages = [
            FinetuneMessage(role=message.role, content=message.content.replace(old_text, new_text))
            for message in copy.messages
        ]
        return copy

    def prepend_assistant_messages(self, prepend_content: str) -> "FinetuneConversation":
        copy = self.model_copy()
        new_messages: list[FinetuneMessage] = []
        for old_message in self.messages:
            if old_message.role == "assistant":
                new_messages.append(FinetuneMessage(role="assistant", content=prepend_content + old_message.content))
            else:
                new_messages.append(old_message)
        copy.messages = new_messages
        return copy

    @property
    def final_message(self) -> str:
        return self.messages[-1].content

    def has_content(self, content: str) -> bool:
        return any(content in message.content for message in self.messages)

    @property
    def final_message_as_dict(self) -> dict[str, str]:
        return {"text": self.final_message}


class TextSFT(BaseModel):
    text: str
