from slist import Slist
from latteries.load_caller import load_openai_caller
from latteries.shared import ChatHistory, InferenceConfig


async def example_parallel_tqdm():
    caller = load_openai_caller("cache")
    fifty_prompts = [f"What is {i} * {i + 1}?" for i in range(50)]
    prompts = [ChatHistory.from_user(prompt) for prompt in fifty_prompts]
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    # All in parallel
    # Slist is a library that has bunch of useful typed functions for lists.
    results = await Slist(prompts).par_map_async(
        lambda prompt: caller.call(prompt, config),
        max_par=10,  # Limits the number of parallel calls.
        tqdm=True,  # Brings up tqdm bar.
    )
    result_strings = [result.first_response for result in results]
    print(result_strings)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_parallel_tqdm())
