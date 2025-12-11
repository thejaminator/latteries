"""
OpenAI-compatible API proxy for Tinker.
This script creates a FastAPI server that exposes an OpenAI-compatible chat/completions endpoint
but uses the Tinker API under the hood for Qwen models.

Supports both base models and finetuned models:
- Base model: "Qwen/Qwen3-8B"
- Finetuned: "tinker://36f466a1-cf6b-54ae-b366-7b7cf8cd517f:train:0/sampler_weights/000200"

Usage:
    python example_scripts/tinker_openai_proxy.py

Then use the OpenAI client with base_url="http://localhost:8000/v1"
"""

import asyncio
import logging
import os
import time
from functools import cache
from typing import Any, NotRequired, TypedDict

import tinker
from tinker import types as tinker_types
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("TINKER_API_KEY")
assert API_KEY, "TINKER_API_KEY is not set"
app = FastAPI(title="Tinker OpenAI Proxy")
logger = logging.getLogger(__name__)

# Global state for caching
_sampling_clients: dict[str, tinker.SamplingClient] = {}
_renderers: dict[str, "Qwen3DisableThinkingRenderer"] = {}
_locks: dict[str, asyncio.Lock] = {}
_model_to_base_model: dict[str, str] = {}  # Cache: model_path -> base_model_name


# ============================================================================
# Tokenizer utils (inlined from tokenizer_utils.py)
# ============================================================================

Tokenizer = Any  # Actually PreTrainedTokenizer from transformers


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


# ============================================================================
# Renderer (inlined from renderers.py - only Qwen3DisableThinkingRenderer)
# ============================================================================


class Message(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[dict]]


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    """Parse response for a single stop token."""
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


class Qwen3DisableThinkingRenderer:
    """
    Renderer that disables thinking for hybrid-mode Qwen3 models.

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        Response here<|im_end|>
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def _render_message(self, idx: int, message: Message) -> tuple[list[int], list[int], list[int]]:
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        if message["role"] == "assistant":
            ob_str += "<think>\n\n</think>\n\n"
        # Observation (prompt) part
        ac_str = f"{ac_content}<|im_end|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Qwen format
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self, messages: list[Message], role: str = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        tokens: list[int] = []  # No BOS token for Qwen
        for idx, message in enumerate(messages):
            ob_part, action_part, _ = self._render_message(idx, message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        # Add generation prompt
        new_partial_message = Message(role=role, content="")
        ob_part, _, _ = self._render_message(len(messages), new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


# ============================================================================
# OpenAI API models
# ============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    n: int = 1
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ============================================================================
# Helper functions
# ============================================================================


def get_lock(model: str) -> asyncio.Lock:
    """Get or create a lock for a specific model."""
    if model not in _locks:
        _locks[model] = asyncio.Lock()
    return _locks[model]


async def get_sampling_client_and_base_model(model: str) -> tuple[tinker.SamplingClient, str]:
    """
    Get or create a sampling client for a model.
    Returns (sampling_client, base_model).

    Supports both base models (e.g., "Qwen/Qwen3-8B") and finetuned models
    (e.g., "tinker://36f466a1-cf6b-54ae-b366-7b7cf8cd517f:train:0/sampler_weights/000200").
    """
    if model not in _sampling_clients:
        base_url = os.getenv("TINKER_BASE_URL")
        service_client = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)

        # Determine base model and create sampling client
        if "tinker://" in model:
            # Query the training run to get the base model
            rest_client = service_client.create_rest_client()
            training_run = await rest_client.get_training_run_by_tinker_path_async(model)
            base_model = training_run.base_model
            _sampling_clients[model] = service_client.create_sampling_client(model_path=model)
        else:
            # It's already a base model name
            base_model = model
            _sampling_clients[model] = service_client.create_sampling_client(base_model=model)

        # Cache the base model mapping
        _model_to_base_model[model] = base_model
    else:
        # Use cached base model
        base_model = _model_to_base_model[model]

    return _sampling_clients[model], base_model


def get_renderer(base_model: str) -> Qwen3DisableThinkingRenderer:
    """Get or create a renderer for a base model. Hardcoded for Qwen models."""
    if base_model not in _renderers:
        tokenizer = get_tokenizer(base_model)
        _renderers[base_model] = Qwen3DisableThinkingRenderer(tokenizer)
    return _renderers[base_model]


# ============================================================================
# API endpoints
# ============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""

    if request.stream:
        raise NotImplementedError("Streaming is not supported yet")

    model = request.model

    # Thread-safe initialization
    async with get_lock(model):
        sampling_client, base_model = await get_sampling_client_and_base_model(model)
        renderer = get_renderer(base_model)

    # Convert messages to renderer format
    renderer_messages: list[Message] = [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    # Build generation prompt
    model_input = renderer.build_generation_prompt(renderer_messages)
    stop_sequences = renderer.get_stop_sequences()

    # Set up sampling parameters
    sampling_params = tinker_types.SamplingParams(
        max_tokens=request.max_tokens or 2048,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop_sequences,
    )

    # Generate response
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=request.n,
        sampling_params=sampling_params,
    )

    # Parse responses
    choices: list[ChatCompletionChoice] = []
    total_completion_tokens = 0

    for i, sequence in enumerate(response.sequences):
        parsed_message, reached_stop = renderer.parse_response(sequence.tokens)
        finish_reason = "stop" if reached_stop else "length"

        choices.append(
            ChatCompletionChoice(
                index=i,
                message=ChatMessage(
                    role="assistant",
                    content=parsed_message["content"],
                ),
                finish_reason=finish_reason,
            )
        )
        total_completion_tokens += len(sequence.tokens)

    # Calculate token usage
    prompt_tokens = len(model_input.to_ints()) if hasattr(model_input, "to_ints") else 0

    return ChatCompletionResponse(
        id=f"chatcmpl-tinker-{int(time.time())}",
        created=int(time.time()),
        model=model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=prompt_tokens + total_completion_tokens,
        ),
    )


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available models."""
    return {
        "object": "list",
        "data": [
            {"id": "Qwen/Qwen3-8B", "object": "model", "owned_by": "tinker"},
            {"id": "Qwen/Qwen3-32B", "object": "model", "owned_by": "tinker"},
        ],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print("Starting Tinker OpenAI Proxy on http://localhost:8000")
    print("Use OpenAI client with base_url='http://localhost:8000/v1'")
    print("Available models: Qwen/Qwen3-8B, Qwen/Qwen3-32B, or finetuned tinker:// paths")
    uvicorn.run(app, host="0.0.0.0", port=8000)
