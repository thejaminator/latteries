from json import JSONDecodeError
import anthropic
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Type
from anthropic.types.message import Message

from openai import NOT_GIVEN, AsyncOpenAI, InternalServerError
import os
from pydantic import ValidationError
from slist import Slist
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from latteries.log_probs import OpenaiResponseWithLogProbs
from latteries.cache import APIRequestCache, CallerCache, OpenaiResponse
from latteries.shared import (
    ChatMessage,
    GenericBaseModel,
    InferenceConfig,
    ToolArgs,
    ChatHistory,
)
from dataclasses import dataclass
import random
import openai


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
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                response_format=schema,
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
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError, anthropic.InternalServerError))),
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
        # convert
        openai_response = OpenaiResponse(
            id=response.id,
            choices=[{"message": {"content": response.content[0].text, "role": "assistant"}}],  # type: ignore
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


async def demo_main():
    # pip install python-dotenv
    import dotenv

    # Please set your .env file with the OPENAI_API_KEY
    dotenv.load_dotenv()
    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"
    question = """Question: Are you a chatbot?

Choices:
A - Yes
B - No

Answer:
Please indicate your answer immmediately with a single letter"""
    max_tokens = 100
    temperature = 0.0
    cached_caller = OpenAICaller(api_key=api_key, cache_path="cached.jsonl")
    response = cached_caller.call(
        messages=ChatHistory.from_user(question),
        config=InferenceConfig(temperature=temperature, max_tokens=max_tokens, model="gpt-4o"),
    )
    res = await response
    print(res.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_main())
