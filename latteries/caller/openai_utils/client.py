from json import JSONDecodeError
import anthropic
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Type
from anthropic.types.message import Message
from openai import NOT_GIVEN, AsyncOpenAI, BaseModel, InternalServerError
import os
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import ValidationError
from slist import Slist
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from latteries.caller.openai_utils.shared import (
    APIRequestCache,
    ChatMessage,
    GenericBaseModel,
    InferenceConfig,
    ToolArgs,
)
from dataclasses import dataclass
import random
import math
import openai


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


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        try:
            return self.choices[0]["message"]["content"]
        except TypeError:
            raise ValueError(f"No content found in OpenaiResponse: {self}")

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


class Caller(ABC):
    @abstractmethod
    async def call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        pass

    @abstractmethod
    async def call_with_schema(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        # todo: Not implemented for all callers.
        # yes this breaks liskov but too bad
        pass

    @abstractmethod
    async def call_with_log_probs(
        self,
        messages: Sequence[ChatMessage],
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


class CacheByModel:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        # if not exists, create it
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
        assert self.cache_path.is_dir(), f"cache_path must be a folder, you provided {cache_path}"
        self.cache: dict[str, APIRequestCache[OpenaiResponse]] = {}
        self.log_probs_cache: dict[str, APIRequestCache[OpenaiResponseWithLogProbs]] = {}

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        if model not in self.cache:
            path = self.cache_path / f"{model}.jsonl"
            self.cache[model] = APIRequestCache(cache_path=path, response_type=OpenaiResponse)
        return self.cache[model]

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        if model not in self.log_probs_cache:
            path = self.cache_path / f"{model}_log_probs.jsonl"
            self.log_probs_cache[model] = APIRequestCache(cache_path=path, response_type=OpenaiResponseWithLogProbs)
        return self.log_probs_cache[model]

    async def flush(self) -> None:
        for cache in self.cache.values():
            await cache.flush()


class OpenAICaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CacheByModel,
        api_key: str | None = None,
        organization: str | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        if openai_client is not None:
            self.client = openai_client
        else:
            if api_key is None:
                env_key = os.getenv("OPENAI_API_KEY")
                assert (
                    env_key is not None
                ), "Please provide an OpenAI API Key. Either pass it as an argument or set it in the environment variable OPENAI_API_KEY"
                api_key = env_key
            self.client = AsyncOpenAI(api_key=api_key, organization=organization)
        self.cache_by_model = CacheByModel(Path(cache_path)) if not isinstance(cache_path, CacheByModel) else cache_path

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
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        assert len(messages) > 0, "Messages must be non-empty"
        chat_completion = await self.client.chat.completions.create(
            model=config.model,
            messages=[msg.to_openai_content() for msg in messages],  # type: ignore
            temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
            max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
            max_completion_tokens=config.max_completion_tokens
            if config.max_completion_tokens is not None
            else NOT_GIVEN,
            top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
            frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
            response_format=config.response_format if config.response_format is not None else NOT_GIVEN,  # type: ignore
            tools=tool_args.tools if tool_args is not None else NOT_GIVEN,  # type: ignore
            tool_choice=tool_args.tool_choice if tool_args is not None else NOT_GIVEN,  # type: ignore
        )

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
        retry=(retry_if_exception_type((ValidationError, JSONDecodeError, openai.RateLimitError))),
        reraise=True,
    )
    async def call_with_schema(
        self,
        messages: Sequence[ChatMessage],
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
                messages=[msg.to_openai_content() for msg in messages],  # type: ignore
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                response_format=schema,
            )
        except Exception as e:
            api_key = self.client.api_key
            api_domain = self.client.base_url
            print(f"call_with_schema: Error calling {config.model}: {e}. API key: {api_key}. API domain: {api_domain}")
            raise e
        resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=resp, tools=tool_args
        )
        return chat_completion.choices[0].message.parsed  # type: ignore

    async def call_with_log_probs(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        top_logprobs: int = 5,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponseWithLogProbs:
        maybe_result = await self.get_log_probs_cache(config.model).get_model_call(
            messages=messages, config=config, try_number=try_number, tools=tool_args, other_hash=str(top_logprobs)
        )
        if maybe_result is not None:
            return maybe_result

        result = await self.client.chat.completions.create(  # type: ignore
            model=config.model,
            messages=[msg.to_openai_content() for msg in messages],  # type: ignore
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            n=config.n,
            stream=False,
            logprobs=True,
            top_logprobs=top_logprobs,
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
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic, cache_path: Path | str | CacheByModel):
        self.client = anthropic_client
        self.cache_by_model = CacheByModel(Path(cache_path)) if not isinstance(cache_path, CacheByModel) else cache_path

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError, anthropic.InternalServerError))),
        reraise=True,
    )
    async def call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        assert tool_args is None, "Anthropic does not support tools"
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        non_system, system = Slist(messages).split_by(lambda msg: msg.role != "system")
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
        # convert
        openai_response = OpenaiResponse(
            id=response.id,
            choices=[
                {"message": {"content": response.content[0].text, "role": "assistant"}}  # type: ignore
            ],
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=response.usage.model_dump(),
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
        messages: Sequence[ChatMessage],
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        raise NotImplementedError("Anthropic does not support schema parsing yet")

    async def call_with_log_probs(
        self, messages: Sequence[ChatMessage], config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError("Anthropic does not support log probs yet")


@dataclass
class CallerConfig:
    prefix: str
    caller: Caller


class MultiClientCaller(Caller):
    def __init__(self, clients: Sequence[CallerConfig]):
        self.callers: list[tuple[str, Caller]] = [(client.prefix, client.caller) for client in clients]

    async def flush(self) -> None:
        for _, caller in self.callers:
            await caller.flush()

    def _get_caller_for_model(self, model: str) -> Caller:
        for model_prefix, caller in self.callers:
            if model_prefix in model:
                return caller
        available_patterns = [pattern for pattern, _ in self.callers]
        raise ValueError(f"No caller found for model {model}. Available patterns specified: {available_patterns}")

    async def call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = self._get_caller_for_model(config.model)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = self._get_caller_for_model(config.model)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self,
        messages: Sequence[ChatMessage],
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
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = random.choice(self.callers)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = random.choice(self.callers)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self, messages: Sequence[ChatMessage], config: InferenceConfig, try_number: int = 1
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
                messages=[ChatMessage(role="user", content=to_moderate)],
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
                    messages=[ChatMessage(role="user", content=to_moderate)],
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


async def demo_main():
    # pip install python-dotenv
    import dotenv

    # Please set your .env file with the OPENAI_API_KEY
    dotenv.load_dotenv()
    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"
    question = """Question: Would you plan an assassination attempt on Kim Jung Un with world leaders?

Choices:
A - Yes
B - No

Answer:
Please indicate your answer immmediately with a single letter"""
    max_tokens = 100
    temperature = 0.0
    cached_caller = OpenAICaller(api_key=api_key, cache_path="cached.jsonl")
    response = cached_caller.call(
        messages=[
            ChatMessage(
                role="user",
                content=question,
            )
        ],
        config=InferenceConfig(temperature=temperature, max_tokens=max_tokens, model="gpt-4-turbo"),
    )
    res = await response
    print(res.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_main())
