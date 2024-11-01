import hashlib
from pathlib import Path
from typing import Generic, Sequence, Type, TypeVar
from pydantic import BaseModel, ValidationError


# Generic to say what we are caching
APIResponse = TypeVar("APIResponse", bound=BaseModel)


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

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
                "content": [
                    {"type": "text", "text": self.content},
                ],
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


class InferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1


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


def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[APIResponse]) -> list[APIResponse]:
    with open(path) as f:
        return [basemodel.model_validate_json(line) for line in f]


def file_cache_key(messages: Sequence[ChatMessage], config: InferenceConfig, try_number: int) -> str:
    str_messages = (
        ",".join([str(msg) for msg in messages]) + deterministic_hash(config.model_dump_json()) + str(try_number)
    )
    return deterministic_hash(str_messages)

GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)

def validate_json_item(item: str, model: Type[GenericBaseModel]) -> GenericBaseModel | None:
    try:
        return model.model_validate_json(item)
    except ValidationError:
        print(f"Error validating {item} with model {model}")
        return None

class APIRequestCache(Generic[APIResponse]):
    def __init__(self, cache_path: Path | str, response_type: Type[APIResponse]):
        self.cache_path = Path(cache_path)
        self.data: dict[str, APIResponse] = {}
        if self.cache_path.exists():
            rows = read_jsonl_file_into_basemodel(
                path=self.cache_path,
                basemodel=FileCacheRow,
            )
        else:
            rows = []
        print(f"Loaded {len(rows)} rows from cache file {self.cache_path.as_posix()}")
        self.response_type = response_type
        self.data: dict[str, response_type] = {}
        # row.key: validate_json_item(row.response, response_type) for row in rows
        for row in rows:
            response = validate_json_item(row.response, response_type)
            if response:
                self.data[row.key] = response
        self.file_handler = self.cache_path.open("a")

    def add_model_call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int,
        response: APIResponse,
    ) -> None:
        key = file_cache_key(messages, config, try_number)
        self.data[key] = response
        response_str = response.model_dump_json()
        self.write_line(key=key, response_json=response_str)

    def get_model_call(
        self, messages: Sequence[ChatMessage], config: InferenceConfig, try_number: int
    ) -> APIResponse | None:
        key = file_cache_key(messages, config, try_number)
        if key in self.data:
            return self.data[key]
        else:
            return None

    def write_line(self, key: str, response_json: str) -> None:
        self.file_handler.write(FileCacheRow(key=key, response=response_json).model_dump_json() + "\n")


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


class HashableBaseModel(BaseModel):
    def model_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)

    class Config:
        # this is needed for the hashable base model
        frozen = True
