from collections import defaultdict
import hashlib
from json import JSONDecodeError
import json
import math
import time
import anthropic
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Sequence, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import tinker
    from tinker.types.sample_response import SampleResponse
from anthropic.types.message import Message
from anyio import Path as AnyioPath

import anyio
from dotenv import load_dotenv
from openai import NOT_GIVEN, AsyncOpenAI, InternalServerError
import os
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import BaseModel, ValidationError
from slist import Slist
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from dataclasses import dataclass
import random
import openai
import asyncio


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
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

    def drop_last_message(self) -> "ChatHistory":
        return ChatHistory(messages=self.messages[:-1])

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


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None
    prompt_used: ChatHistory | None = None
    is_refused: bool = False

    @property
    def first_response(self) -> str:
        try:
            content = self.choices[0]["message"]["content"]
            if content is None:
                raise ValueError(f"No content found in OpenaiResponse: {self}")
            return content
        except TypeError:
            raise ValueError(f"No content found in OpenaiResponse: {self}")

    @property
    def responses(self) -> Slist[str]:
        # When n > 1, we get a list of responses.
        return Slist(self.choices).map(lambda x: x["message"]["content"])

    @property
    def all_responses(self) -> list[str]:
        return [choice["message"]["content"] for choice in self.choices]

    @property
    def reasoning_content(self) -> str:
        ## sometimes has reasoning_content or reasoning instead of content e.g. deepseek-reasoner or gemini
        possible_keys = ["reasoning_content", "reasoning"]
        for key in possible_keys:
            if self.choices[0]["message"].get(key):
                return self.choices[0]["message"][key]
        raise ValueError(f"No reasoning_content found in OpenaiResponse: {self}")

    @property
    def has_reasoning(self) -> bool:
        possible_keys = ["reasoning_content", "reasoning"]
        for key in possible_keys:
            if self.choices[0]["message"].get(key):
                return True
        return False

    def has_response(self) -> bool:
        if len(self.choices) == 0:
            return False
        first_choice = self.choices[0]
        if first_choice["message"] is None:
            return False
        if first_choice["message"]["content"] is None:
            return False
        return True

    @property
    def hit_content_filter(self) -> bool:
        first_choice = self.choices[0]
        if "finishReason" in first_choice:
            if first_choice["finishReason"] == "content_filter":
                return True
        if "finish_reason" in first_choice:
            if first_choice["finish_reason"] == "content_filter":
                return True
        return False


class FileCacheRow(BaseModel):
    key: str
    response: str  # Should be generic, but w/e


class Prob(BaseModel):
    token: str
    prob: float


class LogProb(BaseModel):
    token: str
    logprob: float

    @property
    def proba(self) -> float:
        return math.exp(self.logprob)

    def to_prob(self) -> Prob:
        return Prob(token=self.token, prob=self.proba)


class TokenWithLogProbs(BaseModel):
    token: str
    logprob: float  # log probability of the particular token
    top_logprobs: Sequence[LogProb]  # log probability of the top 5 tokens

    def sorted_logprobs(self) -> Sequence[LogProb]:  # Highest to lowest
        return sorted(self.top_logprobs, key=lambda x: x.logprob, reverse=True)

    def sorted_probs(self) -> Sequence[Prob]:
        return [logprob.to_prob() for logprob in self.sorted_logprobs()]


class ResponseWithLogProbs(BaseModel):
    response: str
    content: Sequence[TokenWithLogProbs]  #


def write_jsonl_file_from_basemodel(
    path: Path | str, basemodels: Sequence[BaseModel], exclude_none: bool = False
) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json(exclude_none=True) + "\n")


def write_jsonl_file_from_dict(path: Path | str, dicts: Sequence[dict[str, Any]]) -> None:
    import json

    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for dict in dicts:
            f.write(json.dumps(dict) + "\n")


def read_jsonl_file_into_basemodel(
    path: Path | str, basemodel: Type[GenericBaseModel], limit: int | None = None
) -> Slist[GenericBaseModel]:
    with open(path) as f:
        if limit is not None:
            out = Slist()
            for line in f:
                out.append(basemodel.model_validate_json(line))
                if len(out) >= limit:
                    break
            return out
        else:
            return Slist(basemodel.model_validate_json(line) for line in f)


def read_jsonl_file_into_dict(path: Path | str, limit: int | None = None) -> Slist[dict[str, Any]]:
    with open(path) as f:
        if limit is not None:
            out = Slist()
            for line in f:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            return out
        else:
            return Slist(json.loads(line) for line in f)


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def deterministic_hash_int(something: str) -> int:
    return int(hashlib.sha1(something.encode()).hexdigest(), 16)


def validate_json_item(item: str, model: Type[GenericBaseModel]) -> GenericBaseModel | None:
    try:
        return model.model_validate_json(item)
    except ValidationError:
        print(f"Error validating {item} with model {model}")
        return None


class ToolArgs(BaseModel):
    tools: Sequence[Mapping[Any, Any]]
    tool_choice: str


class NotGivenSentinel:
    pass


NOT_GIVEN_SENTINEL = NotGivenSentinel()


class InferenceConfig(BaseModel):
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
    # "minimal", "low", "medium", "high" for openai only
    reasoning_effort: str | None = None
    continue_final_message: bool | None = None  # For runpod configs
    renderer_name: str | None = None  # for tinker callers
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


class OpenaiResponseWithLogProbs(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        return self.choices[0]["message"]["content"]

    def response_with_logprobs(self) -> ResponseWithLogProbs:
        response = self.first_response
        logprobs = self.choices[0]["logprobs"]["content"]
        parsed_content = [TokenWithLogProbs.model_validate(token) for token in logprobs]
        return ResponseWithLogProbs(response=response, content=parsed_content)

    def first_token_probability_for_target(self, target: str) -> float:
        logprobs = self.response_with_logprobs().content
        first_token = logprobs[0]
        for token in first_token.top_logprobs:
            # print(f"Token: {token.token} Logprob: {token.logprob}")
            if token.token == target:
                token_logprob = token.logprob
                # convert natural log to prob
                return math.exp(token_logprob)
        return 0.0


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


async def read_jsonl_file_into_basemodel_async(
    path: AnyioPath, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    async with await anyio.open_file(path, "r") as f:
        # print(f"Reading {path}")
        return Slist([basemodel.model_validate_json(line) for line in await f.readlines()])


class Caller(ABC):
    @abstractmethod
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        pass

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        # todo: Not implemented for all callers.
        # yes this breaks liskov but too bad
        raise NotImplementedError()

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError()

    @abstractmethod
    async def flush(self) -> None:
        # flush file buffers
        raise NotImplementedError()

    ## implement context manager
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.flush()


class APIRequestCache(Generic[GenericBaseModel]):
    def __init__(self, cache_path: Path | str, response_type: Type[GenericBaseModel]):
        self.cache_path = AnyioPath(cache_path)
        self.response_type = response_type
        self.data: dict[str, str] = {}
        self.file_handler: Optional[anyio.AsyncFile] = None
        self.loaded_cache: bool = False
        self.cache_check_semaphore = anyio.Semaphore(1)

    async def flush(self) -> None:
        if self.file_handler:
            await self.file_handler.flush()

    async def load_cache(self) -> None:
        if await self.cache_path.exists():
            time_start = time.time()
            rows: Slist[FileCacheRow] = await read_jsonl_file_into_basemodel_async(
                path=self.cache_path,  # todo: asyncify
                basemodel=FileCacheRow,
            )
            time_end = time.time()
            n_items = len(rows)
            time_diff_1dp = round(time_end - time_start, 1)
            print(f"Loaded {n_items} items from {self.cache_path.as_posix()} in {time_diff_1dp} seconds")
        else:
            rows = Slist()
        for row in rows:
            self.data[row.key] = row.response
        self.loaded_cache = True

    async def get_file_handler(self) -> anyio.AsyncFile:
        if self.file_handler is None:
            # if the file doesn't exist, create it
            if not await self.cache_path.exists():
                # make parent directories
                await self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                # make sure it's created
                await self.cache_path.touch()
            self.file_handler = await anyio.open_file(self.cache_path, "a")
        return self.file_handler

    async def add_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        response: GenericBaseModel,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = response.model_dump_json()
        self.data[key] = response_str
        await self.write_line(key=key, response_json=response_str)

    async def get_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[GenericBaseModel]:
        if not self.loaded_cache:
            async with self.cache_check_semaphore:
                # check again
                if not self.loaded_cache:
                    await self.load_cache()
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = self.data.get(key)
        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                # add the prompt used to the response
                return response
            except ValidationError as e:
                print(f"Warning: Failed to validate cache entry for key {key}")
                raise e
                # return None
        return None

    async def write_line(self, key: str, response_json: str) -> None:
        if not self.file_handler:
            await self.get_file_handler()
        if self.file_handler:
            # prevent multiple writes to same file
            async with self.cache_check_semaphore:
                line = FileCacheRow(key=key, response=response_json).model_dump_json() + "\n"
                await self.file_handler.write(line)


class CallerCache(Generic[GenericBaseModel]):
    """Will create a jsonl cache for each model."""

    def __init__(self, cache_path: Path, cache_type: Type[GenericBaseModel] = OpenaiResponse):
        self.cache_path = Path(cache_path)
        # if not exists, create it
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
        assert self.cache_path.is_dir(), f"cache_path must be a folder, you provided {cache_path}"
        self.cache: dict[str, APIRequestCache[GenericBaseModel]] = {}
        self.log_probs_cache: dict[str, APIRequestCache[OpenaiResponseWithLogProbs]] = {}
        self.cache_type = cache_type

    def get_cache(self, model: str) -> APIRequestCache[GenericBaseModel]:
        if model not in self.cache:
            path = self.cache_path / f"{model}.jsonl"
            self.cache[model] = APIRequestCache(cache_path=path, response_type=self.cache_type)
        return self.cache[model]

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        if model not in self.log_probs_cache:
            path = self.cache_path / f"{model}_log_probs.jsonl"
            self.log_probs_cache[model] = APIRequestCache(cache_path=path, response_type=OpenaiResponseWithLogProbs)
        return self.log_probs_cache[model]

    async def flush(self) -> None:
        for cache in self.cache.values():
            await cache.flush()


class NoOpAPICache(APIRequestCache[GenericBaseModel]):
    def __init__(self):
        self.data = {}
        self.file_handler = None
        self.loaded_cache = True
        self.cache_check_semaphore = anyio.Semaphore(1)

    async def flush(self) -> None:
        pass

    async def load_cache(self) -> None:
        pass

    async def add_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        response: GenericBaseModel,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        pass

    async def get_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[GenericBaseModel]:
        return None

    async def write_line(self, key: str, response_json: str) -> None:
        pass


class NoOpCache(CallerCache[GenericBaseModel]):
    # Useful when you don't want to cache.
    def __init__(self):
        self.cache = {}
        self.log_probs_cache = {}
        self.cache_type = OpenaiResponse

    def get_cache(self, model: str) -> APIRequestCache[GenericBaseModel]:
        if model not in self.cache:
            self.cache[model] = NoOpAPICache()
        return self.cache[model]

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        if model not in self.log_probs_cache:
            self.log_probs_cache[model] = NoOpAPICache()
        return self.log_probs_cache[model]  # type: ignore

    async def flush(self) -> None:
        pass


class OpenAICaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CallerCache,
        api_key: str | None = None,
        organization: str | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        if openai_client is not None:
            self.client = openai_client
        else:
            if api_key is None:
                env_key = os.getenv("OPENAI_API_KEY")
                assert env_key is not None, (
                    "Please provide an OpenAI API Key. Either pass it as an argument or set it in the environment variable OPENAI_API_KEY"
                )
                api_key = env_key
            self.client = AsyncOpenAI(api_key=api_key, organization=organization)
        self.cache_by_model = (
            CallerCache(Path(cache_path), cache_type=OpenaiResponse)
            if not isinstance(cache_path, CallerCache)
            else cache_path
        )

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError, JSONDecodeError, InternalServerError))),
        reraise=True,
    )
    @retry(
        stop=(stop_after_attempt(10)),
        wait=(wait_fixed(30)),  # for rate limits, wait longer
        retry=(retry_if_exception_type((openai.RateLimitError))),
        reraise=True,
    )
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        assert len(messages.messages) > 0, "Messages must be non-empty"
        extra_body = config.extra_body or {}
        if config.continue_final_message:
            # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
            # disable add_generation_prompt to continue the conversation
            extra_body["continue_final_message"] = config.continue_final_message
            extra_body["add_generation_prompt"] = not config.continue_final_message
        try:
            chat_completion = await self.client.chat.completions.create(
                model=config.model,
                messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
                temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
                max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
                max_completion_tokens=(
                    config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
                ),
                top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
                frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
                tools=tool_args.tools if tool_args is not None else NOT_GIVEN,  # type: ignore
                extra_body=extra_body or None,
                n=config.n,
                reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
            )
        except Exception as e:
            note = f"Model: {config.model}. API domain: {self.client.base_url}"
            e.add_note(note)
            raise e

        try:
            resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        except ValidationError as e:
            print(
                f"Validation error for model {config.model}. Prompt: {messages}. resp: {chat_completion.model_dump()}"
            )
            raise e

        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=resp, tools=tool_args
        )
        return resp

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(
            retry_if_exception_type((ValidationError, JSONDecodeError, openai.RateLimitError, openai.APITimeoutError))
        ),
        reraise=True,
    )
    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> GenericBaseModel:
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return schema.model_validate_json(maybe_result.first_response)
        try:
            chat_completion = await self.client.beta.chat.completions.parse(
                model=config.model,
                messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
                temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
                max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
                max_completion_tokens=(
                    config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
                ),
                top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
                frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
                response_format=schema,
                extra_body=config.extra_body or {},
                reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
                n=config.n if config.n is not None else NOT_GIVEN,
            )
        except Exception as e:
            api_key = self.client.api_key
            api_domain = self.client.base_url
            note = f"Model: {config.model}. API key: {api_key}. API domain: {api_domain}"
            e.add_note(note)
            raise e
        resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=resp, tools=tool_args
        )
        return chat_completion.choices[0].message.parsed  # type: ignore

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        top_logprobs: int = 5,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponseWithLogProbs:
        maybe_result = await self.get_log_probs_cache(config.model).get_model_call(
            messages=messages, config=config, try_number=try_number, tools=tool_args, other_hash=str(top_logprobs)
        )
        if maybe_result is not None:
            return maybe_result

        result = await self.client.chat.completions.create(  # type: ignore
            model=config.model,
            messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
            temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
            max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
            max_completion_tokens=(
                config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
            ),
            top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
            frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
            n=config.n,
            stream=False,
            logprobs=True,
            top_logprobs=top_logprobs,
            reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
        )
        resp = OpenaiResponseWithLogProbs.model_validate(result.model_dump())

        await self.get_log_probs_cache(config.model).add_model_call(
            messages=messages,
            config=config,
            try_number=try_number,
            response=resp,
            other_hash=str(top_logprobs),
            tools=tool_args,
        )
        return resp


class AnthropicCaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CallerCache,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        api_key: str | None = None,
    ):
        if anthropic_client is not None:
            self.client = anthropic_client
        else:
            if api_key is None:
                env_key = os.getenv("ANTHROPIC_API_KEY")
                assert env_key is not None, "Please provide an Anthropic API Key"
                api_key = env_key
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.cache_by_model = CallerCache(Path(cache_path)) if not isinstance(cache_path, CallerCache) else cache_path

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    @retry(
        stop=(stop_after_attempt(10)),
        wait=(wait_fixed(10)),
        retry=(retry_if_exception_type((anthropic.InternalServerError, anthropic.RateLimitError))),
        reraise=True,
    )
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        assert tool_args is None, "Anthropic does not support tools"
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        non_system, system = Slist(messages.messages).split_by(lambda msg: msg.role != "system")
        anthropic_messages = [{"role": msg.role, "content": msg.content} for msg in non_system]
        if system.length >= 2:
            raise ValueError("Anthropic does not support multiple system messages")
        system_message: ChatMessage | None = system.first_option
        to_pass_sys = system_message.content if system_message is not None else anthropic.NOT_GIVEN

        assert config.max_tokens is not None, "Anthropic requires max_tokens"
        response: Message = await self.client.messages.create(
            model=config.model,
            messages=anthropic_messages,  # type: ignore
            max_tokens=config.max_tokens,
            temperature=config.temperature if config.temperature is not None else anthropic.NOT_GIVEN,
            top_p=config.top_p if config.top_p is not None else anthropic.NOT_GIVEN,
            system=to_pass_sys,
        )
        # Message(id='msg_01Cdvdp2KxxL5uPAugi3xbdh', content=[], model='claude-sonnet-4-5-20250929', role='assistant', stop_reason='refusal'
        is_not_refusal = response.stop_reason != "refusal"
        openai_response = OpenaiResponse(
            id=response.id,
            choices=[{"message": {"content": response.content[0].text, "role": "assistant"}}] if is_not_refusal else [],
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=response.usage.model_dump(),
            is_refused=not is_not_refusal,
        )

        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=openai_response, tools=tool_args
        )

        return openai_response

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError))),
        reraise=True,
    )
    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        raise NotImplementedError("Anthropic does not support schema parsing yet")

    async def call_with_log_probs(
        self, messages: ChatHistory, config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError("Anthropic does not support log probs yet")


@dataclass
class SamplingClientAndRenderer:
    sampling_client: Any  # tinker.SamplingClient
    renderer: Any
    base_model: str


class TinkerCaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CallerCache,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize TinkerCaller.

        Args:
            cache_path: Path to cache directory or CallerCache instance
            base_url: Base URL for tinker service (optional)
            api_key: API key for tinker service (optional)
        """
        import importlib.util

        if importlib.util.find_spec("tinker") is None:
            raise ImportError("tinker package is required for TinkerCaller. Please install it.")

        self.cache_by_model = (
            CallerCache(Path(cache_path), cache_type=OpenaiResponse)
            if not isinstance(cache_path, CallerCache)
            else cache_path
        )

        self.api_key = api_key
        self.base_url = base_url
        self.sampling_clients: dict[str, "tinker.SamplingClient"] = {}  # Dict to store sampling clients by model
        self._model_to_base_model: dict[str, str] = {}  # Cache: model_path -> base_model_name
        self._base_model_to_renderer: dict[str, Any] = {}  # Cache: base_model_name -> renderer
        self._model_semaphores: dict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )  # Lock per model for thread-safe initialization

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    def _get_or_create_renderer(self, base_model: str, renderer_name: str | None = None) -> Any:
        """
        Get or create a renderer for a base model.

        Args:
            base_model: The base model name
            renderer_name: Optional renderer name override

        Returns:
            The renderer instance
        """
        if base_model not in self._base_model_to_renderer:
            from example_scripts.tinker_cookbook import renderers as renderers_module
            from example_scripts.tinker_cookbook.model_info import get_recommended_renderer_name
            from example_scripts.tinker_cookbook.tokenizer_utils import get_tokenizer

            tokenizer = get_tokenizer(base_model)
            renderer_name_to_use = renderer_name or get_recommended_renderer_name(base_model)
            renderer = renderers_module.get_renderer(renderer_name_to_use, tokenizer)
            self._base_model_to_renderer[base_model] = renderer

        return self._base_model_to_renderer[base_model]

    async def get_sampling_client_and_renderer(
        self, model: str, renderer_name: str | None = None
    ) -> SamplingClientAndRenderer:
        """
        Get or create sampling client and renderer for a model.
        Thread-safe with per-model locking.

        Args:
            model: Model path (tinker://...) or base model name
            renderer_name: Optional renderer name override

        Returns:
            SamplingClientAndRenderer with the sampling client, renderer, and base model name
        """
        try:
            import tinker
        except ImportError:
            raise ImportError("tinker package is required for TinkerCaller")

        # Step 1: Get or create sampling client and determine base model
        if model not in self.sampling_clients:
            service_client = tinker.ServiceClient(base_url=self.base_url, api_key=self.api_key)

            # Determine base model and create sampling client
            if "tinker://" in model:
                # Query the training run to get the base model
                rest_client = service_client.create_rest_client()
                training_run = await rest_client.get_training_run_by_tinker_path_async(model)
                base_model = training_run.base_model
                self.sampling_clients[model] = service_client.create_sampling_client(model_path=model)
            else:
                # It's already a base model name
                base_model = model
                self.sampling_clients[model] = service_client.create_sampling_client(base_model=model)

            # Cache the base model mapping
            self._model_to_base_model[model] = base_model
        else:
            # Use cached base model
            base_model = self._model_to_base_model[model]

        # Step 2: Get or create renderer for the base model
        renderer = self._get_or_create_renderer(base_model, renderer_name)

        # Step 3: Return the result
        return SamplingClientAndRenderer(
            sampling_client=self.sampling_clients[model],
            renderer=renderer,
            base_model=base_model,
        )

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError))),
        reraise=True,
    )
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        try:
            import tinker
            from tinker import types as tinker_types
        except ImportError:
            raise ImportError("tinker package is required for TinkerCaller")

        if tool_args is not None:
            raise NotImplementedError("TinkerCaller does not support tools yet")

        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        assert len(messages.messages) > 0, "Messages must be non-empty"

        from example_scripts.tinker_cookbook import renderers as renderers_module

        # Get sampling client and renderer (thread-safe with per-model locking)
        async with self._model_semaphores[config.model]:
            client_and_renderer = await self.get_sampling_client_and_renderer(config.model, config.renderer_name)
        renderer: renderers_module.Renderer = client_and_renderer.renderer

        renderer_messages: list[renderers_module.Message] = [
            {"role": msg.role, "content": msg.content}
            for msg in messages.messages  # type: ignore
        ]

        # Use renderer to build the generation prompt
        # Sad hardcoding here to enable thinking for DeepSeekV3.
        prefill = "<think>" if config.renderer_name == "deepseekv3" else None
        model_input = renderer.build_generation_prompt(renderer_messages, prefill=prefill)

        # Get stop sequences from renderer
        stop_sequences = renderer.get_stop_sequences()

        # Set up sampling parameters
        sampling_params = tinker_types.SamplingParams(
            max_tokens=config.max_completion_tokens or config.max_tokens,
            temperature=config.temperature if config.temperature is not None else 1.0,
            top_p=config.top_p if config.top_p is not None else 1.0,
            stop=stop_sequences,
        )

        # Generate response
        sampler: tinker.SamplingClient = client_and_renderer.sampling_client
        response: SampleResponse = await sampler.sample_async(
            prompt=model_input, num_samples=config.n, sampling_params=sampling_params
        )

        # Parse responses using the renderer
        parsed_responses: list[tuple[str, bool]] = []
        for sequence in response.sequences:
            parsed_message, reached_stop = client_and_renderer.renderer.parse_response(sequence.tokens)
            content_with_prefill = parsed_message["content"] if prefill is None else prefill + parsed_message["content"]
            parsed_responses.append((content_with_prefill, reached_stop))

        # Convert to OpenaiResponse format
        choices = []
        for i, response_text in enumerate(parsed_responses):
            response_text, reached_stop = parsed_responses[i]
            if reached_stop:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            choices.append(
                {"message": {"content": response_text, "role": "assistant"}, "index": i, "finish_reason": finish_reason}
            )

        openai_response = OpenaiResponse(
            id=f"tinker-{int(datetime.now().timestamp())}",
            choices=choices,
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage={
                "prompt_tokens": len(model_input.to_ints()) if hasattr(model_input, "to_ints") else 0,
                "completion_tokens": sum(len(seq.tokens) for seq in response.sequences),
                "total_tokens": (len(model_input.to_ints()) if hasattr(model_input, "to_ints") else 0)
                + sum(len(seq.tokens) for seq in response.sequences),
            },
        )

        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=openai_response, tools=tool_args
        )
        return openai_response

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        raise NotImplementedError("TinkerCaller does not support schema parsing yet")

    async def call_with_log_probs(
        self, messages: ChatHistory, config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError("TinkerCaller does not support log probs yet")


@dataclass
class CallerConfig:
    name: str
    caller: Caller


class MultiClientCaller(Caller):
    def __init__(self, clients: Sequence[CallerConfig]):
        self.callers: list[CallerConfig] = list(clients)

    def merge(self, other: "MultiClientCaller") -> "MultiClientCaller":
        return MultiClientCaller(self.callers + other.callers)

    async def flush(self) -> None:
        for caller_config in self.callers:
            await caller_config.caller.flush()

    def _get_caller_for_model(self, model: str) -> Caller:
        # Router logic. It is simply a string match.
        for caller_config in self.callers:
            if caller_config.name in model:
                return caller_config.caller
        available_patterns = [caller_config.name for caller_config in self.callers]
        raise ValueError(f"No caller found for model {model}. Available patterns specified: {available_patterns}")

    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = self._get_caller_for_model(config.model)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = self._get_caller_for_model(config.model)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
    ) -> OpenaiResponseWithLogProbs:
        caller = self._get_caller_for_model(config.model)
        return await caller.call_with_log_probs(messages, config, try_number)


class PooledCaller(Caller):
    def __init__(self, callers: Sequence[Caller]):
        self.callers = callers

    async def flush(self) -> None:
        for caller in self.callers:
            await caller.flush()

    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = random.choice(self.callers)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = random.choice(self.callers)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self, messages: ChatHistory, config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        caller = random.choice(self.callers)
        return await caller.call_with_log_probs(messages, config, try_number)


class OpenAIModerateCaller:
    def __init__(self, api_key: str, cache_path: Path | str):
        self.api_key = api_key
        self.cache: APIRequestCache[ModerationCreateResponse] = APIRequestCache(
            cache_path=cache_path, response_type=ModerationCreateResponse
        )
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((ValidationError, InternalServerError)),
    )
    async def moderate(
        self,
        to_moderate: str,
        model: str = "omni-moderation-latest",
        try_number: int = 1,
    ) -> ModerationCreateResponse:
        """
        Moderates the given text using OpenAI's moderation API.

        Args:
            to_moderate (str): The text to be moderated.
            model (str): The model to use for moderation. Defaults to "omni-moderation-latest".
            try_number (int): The attempt number for retries. Defaults to 1.

        Returns:

            ModerationResponse: The parsed moderation response.

        """

        if self.cache is not None:
            maybe_result = await self.cache.get_model_call(
                messages=ChatHistory(messages=[ChatMessage(role="user", content=to_moderate)]),
                config=InferenceConfig(model=model),
                try_number=try_number,
                tools=None,
            )
            if maybe_result is not None:
                return maybe_result

        try:
            moderation_response: ModerationCreateResponse = await self.client.moderations.create(
                model=model,
                input=to_moderate,
            )

            # add the response to the cache
            if self.cache is not None:
                await self.cache.add_model_call(
                    messages=ChatHistory(messages=[ChatMessage(role="user", content=to_moderate)]),
                    config=InferenceConfig(model=model),
                    try_number=try_number,
                    response=moderation_response,
                    tools=None,
                )

            return moderation_response

        except ValidationError as ve:
            # Optionally, add logging here

            raise ve

        except Exception as e:
            # Optionally, handle other exceptions

            raise e


def load_openai_caller(cache_path: str | Path) -> OpenAICaller:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path=shared_cache)
    return openai_caller


def load_multi_caller(cache_path: str) -> MultiClientCaller:
    """Non-exhaustive list of models. For demonstration purposes.
    Simply copy and create a new function for your needs.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORGANIZATION")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    tinker_base_url = os.getenv("TINKER_BASE_URL")  # Optional

    assert anthropic_api_key, "Please provide an Anthropic API Key"
    assert openai_api_key, "Please provide an OpenAI API Key"
    assert openrouter_api_key, "Please provide an OpenRouter API Key"

    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=openai_org, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )

    clients = [
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="google", caller=openrouter_caller),
        CallerConfig(name="qwen", caller=openrouter_caller),
        CallerConfig(name="deepseek", caller=openrouter_caller),
        CallerConfig(name="mistral", caller=openrouter_caller),
        CallerConfig(name="llama", caller=openrouter_caller),
        CallerConfig(
            name="claude",
            caller=AnthropicCaller(api_key=anthropic_api_key, cache_path=shared_cache),
        ),
    ]

    # Add TinkerCaller if tinker is available
    try:
        tinker_caller = TinkerCaller(
            cache_path=shared_cache,
            base_url=tinker_base_url,
        )
        clients.append(CallerConfig(name="tinker", caller=tinker_caller))
    except ImportError:
        # Tinker not available, skip adding TinkerCaller
        pass

    return MultiClientCaller(clients)


async def example_main():
    # Caches to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=1.0, max_tokens=100, model="gpt-4.1-mini")
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
