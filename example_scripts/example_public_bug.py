"""
Demo script to show calling a tinker model.

First we use the OpenAI API to show it works, then we use the lower-level
tinker sampler API which should show a "model does not exist" error.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

API_KEY = os.getenv("TINKER_API_KEY")


# ============================================================================
# Part 1: OpenAI API (should work)
# ============================================================================
def test_openai_api(model_path: str):
    print("=" * 60)
    print("PART 1: Testing OpenAI API")
    print("=" * 60)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    response = client.completions.create(
        model=model_path,
        prompt="The capital of France is",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"OpenAI API Response: {response.choices[0].text}")
    print()


# ============================================================================
# Part 2: Tinker Sampler API (should fail with model not found)
# ============================================================================
async def test_tinker_sampler(model_path: str, renderer_name: str):
    print("=" * 60)
    print("PART 2: Testing Tinker Sampler API (lower-level)")
    print("=" * 60)

    import tinker

    from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3DisableThinkingRenderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    # Create service client
    service = tinker.ServiceClient(api_key=API_KEY)

    # Create sampling client
    sampling_client = service.create_sampling_client(model_path=model_path)

    # Get renderer
    from tinker_cookbook import renderers

    tokenizer = get_tokenizer(renderer_name)
    renderer = renderers.get_renderer("deepseekv3", tokenizer)

    # Build the prompt
    messages: list[dict[str, str]] = [{"role": "user", "content": "The capital of France is"}]
    model_input = renderer.build_generation_prompt(messages)  # type: ignore[arg-type]

    sample = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        ),
    )

    # Decode response
    seq = sample.sequences[0]
    response_text = tokenizer.decode(seq.tokens)
    print(f"Tinker Sampler Response: {response_text}")


if __name__ == "__main__":
    MODEL_PATH = "tinker://c6c32237-da8d-5024-8001-2c90dd74fb37:train:0/sampler_weights/final"
    RENDERER_NAME = "deepseek-ai/DeepSeek-V3.1"
    # Part 1: OpenAI API
    test_openai_api(MODEL_PATH)

    # Part 2: Tinker Sampler API
    asyncio.run(test_tinker_sampler(MODEL_PATH, RENDERER_NAME))
