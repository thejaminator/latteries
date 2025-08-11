from openai import BaseModel
from latteries.shared import (
    APIResponse,
    ChatHistory,
    FileCacheRow,
    InferenceConfig,
    ToolArgs,
    file_cache_key,
    read_jsonl_file_into_basemodel_async,
)


import anyio
from anyio import Path as AnyioPath
from pydantic import ValidationError
from slist import Slist


import time
from pathlib import Path
from typing import Generic, Optional, Type

from latteries.log_probs import OpenaiResponseWithLogProbs


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None
    prompt_used: ChatHistory | None = None

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
        """
        OpenaiResponse(choices=[{'finish_reason': None, 'index': 0, 'logprobs': None, 'message': None, 'finishReason': 'content_filter'}], usage={'completion_tokens': None, 'prompt_tokens': None, 'total_tokens': None, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'completionTokens': 0, 'promptTokens': 279, 'totalTokens': 279}, created=1734802468, model='gemini-2.0-flash-exp', id=None, system_fingerprint=None, object='chat.completion', service_tier=None)
        """
        first_choice = self.choices[0]
        if "finishReason" in first_choice:
            if first_choice["finishReason"] == "content_filter":
                return True
        if "finish_reason" in first_choice:
            if first_choice["finish_reason"] == "content_filter":
                return True
        return False


class APIRequestCache(Generic[APIResponse]):
    def __init__(self, cache_path: Path | str, response_type: Type[APIResponse]):
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
        response: APIResponse,
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
    ) -> Optional[APIResponse]:
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


class CallerCache(Generic[APIResponse]):
    """Will create a jsonl cache for each model."""

    def __init__(self, cache_path: Path, cache_type: Type[APIResponse] = OpenaiResponse):
        self.cache_path = Path(cache_path)
        # if not exists, create it
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
        assert self.cache_path.is_dir(), f"cache_path must be a folder, you provided {cache_path}"
        self.cache: dict[str, APIRequestCache[APIResponse]] = {}
        self.log_probs_cache: dict[str, APIRequestCache[OpenaiResponseWithLogProbs]] = {}
        self.cache_type = cache_type

    def get_cache(self, model: str) -> APIRequestCache[APIResponse]:
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


CacheByModel = CallerCache  # backwards compatibility name.
