import anyio
from anyio import Path as AnyioPath
from typing import Type, Sequence
from pydantic import BaseModel
import hashlib
from pathlib import Path
from typing import TypeVar, Mapping, Any
from pydantic import ValidationError
from slist import Slist

# Generic to say what we are caching
APIResponse = TypeVar("APIResponse", bound=BaseModel)


class ToolArgs(BaseModel):
    tools: Sequence[Mapping[Any, Any]]
    tool_choice: str


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        """e.g.
            "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base_64}"
                        },
                    },
                ],
            }
        ],

        """
        if not self.image_content:
            return {
                "role": self.role,
                "content": self.content,
            }
        else:
            assert self.image_type, "Please provide an image type"
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{self.image_type};base64,{self.image_content}"},
                    },
                ],
            }

    def to_anthropic_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                ],
            }
        else:
            """
                        {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image1_media_type,
                    "data": image1_data,
                },
            },
            """
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.image_type or "image/jpeg",
                            "data": self.image_content,
                        },
                    },
                ],
            }


class ChatHistory(BaseModel):
    messages: Sequence[ChatMessage] = []

    @staticmethod
    def from_system(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="system", content=content)])

    @staticmethod
    def from_user(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="user", content=content)])

    @staticmethod
    def from_maybe_system(content: str | None) -> "ChatHistory":
        # Sometimes system prompt is optional in user functions.
        if content is None:
            return ChatHistory()
        else:
            return ChatHistory.from_system(content)

    def all_assistant_messages(self) -> Slist[ChatMessage]:
        return Slist(self.messages).filter(lambda msg: msg.role == "assistant")

    def as_text(self) -> str:
        return "\n".join([msg.as_text() for msg in self.messages])

    def add_user(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="user", content=content)]
        return ChatHistory(messages=new_messages)

    def add_assistant(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="assistant", content=content)]
        return ChatHistory(messages=new_messages)

    def add_messages(self, messages: Sequence[ChatMessage]) -> "ChatHistory":
        new_messages = list(self.messages) + list(messages)
        return ChatHistory(messages=new_messages)


class NotGivenSentinel:
    pass


NOT_GIVEN_SENTINEL = NotGivenSentinel()


class InferenceConfig(BaseModel):
    # todo: consider switching to NOT_GIVEN_SENTINEL instead of None
    # Config for openai
    model: str
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    # legacy APIs
    max_tokens: int | None = None
    # newer APIs that prefer the completion limits
    max_completion_tokens: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    # "minimal", "low", "medium", "high" for openai
    reasoning_effort: str | None = None
    continue_final_message: bool | None = None  # For runpod configs
    extra_body: dict | None = None

    def copy_update(
        self,
        temperature: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        top_p: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        max_tokens: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        max_completion_tokens: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        frequency_penalty: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        presence_penalty: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        n: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        continue_final_message: bool | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        reasoning_effort: str | NotGivenSentinel = NOT_GIVEN_SENTINEL,
    ) -> "InferenceConfig":
        return InferenceConfig(
            model=self.model,
            temperature=temperature if not isinstance(temperature, NotGivenSentinel) else self.temperature,
            top_p=top_p if not isinstance(top_p, NotGivenSentinel) else self.top_p,
            max_tokens=max_tokens if not isinstance(max_tokens, NotGivenSentinel) else self.max_tokens,
            max_completion_tokens=max_completion_tokens
            if not isinstance(max_completion_tokens, NotGivenSentinel)
            else self.max_completion_tokens,
            frequency_penalty=frequency_penalty
            if not isinstance(frequency_penalty, NotGivenSentinel)
            else self.frequency_penalty,
            presence_penalty=presence_penalty
            if not isinstance(presence_penalty, NotGivenSentinel)
            else self.presence_penalty,
            n=n if not isinstance(n, NotGivenSentinel) else self.n,
            continue_final_message=continue_final_message
            if not isinstance(continue_final_message, NotGivenSentinel)
            else self.continue_final_message,
            reasoning_effort=reasoning_effort
            if not isinstance(reasoning_effort, NotGivenSentinel)
            else self.reasoning_effort,
        )


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(f"This response has multiple responses {self.raw_responses}")
        else:
            return self.raw_responses[0]


class FileCacheRow(BaseModel):
    key: str
    response: str  # Should be generic, but w/e


def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[APIResponse]) -> Slist[APIResponse]:
    with open(path) as f:
        return Slist(basemodel.model_validate_json(line) for line in f)


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
    try_number: int,
    other_hash: str,
    tools: ToolArgs | None,
) -> str:
    messages_dump = messages.model_dump_json(exclude_none=True)
    config_dump = config.model_dump_json(exclude_none=True)  # for backwards compatibility
    tools_json = tools.model_dump_json(exclude_none=True) if tools is not None else ""  # for backwards compatibility
    _str = messages_dump + config_dump + tools_json + str(try_number) + other_hash
    return deterministic_hash(_str)


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def validate_json_item(item: str, model: Type[GenericBaseModel]) -> GenericBaseModel | None:
    try:
        return model.model_validate_json(item)
    except ValidationError:
        print(f"Error validating {item} with model {model}")
        return None


async def read_jsonl_file_into_basemodel_async(
    path: AnyioPath, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    async with await anyio.open_file(path, "r") as f:
        return Slist([basemodel.model_validate_json(line) for line in await f.readlines()])
