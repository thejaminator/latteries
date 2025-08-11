from latteries.cache import APIRequestCache
from latteries.shared import ChatHistory, ChatMessage, InferenceConfig


from openai import AsyncOpenAI, InternalServerError
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


from pathlib import Path


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
