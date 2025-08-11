# James' API LLM evaluations workflow library
Library of functions that I find useful in my day-to-day work.

## Installation

1. **Create and Activate a Virtual Environment:**
  ```bash
  uv venv venv
  source venv/bin/activate
  ```

2. **Install the dependencies:**
  ```bash
  pip install -r requirements.txt
  ```


Philosophy:
- I want to call LLM APIs like normal python.
- This is a library. Not a framework. Frameworks make you declare magical things in configs and functions that you don't understand. This is a library, which is a collection of tools I find useful.
- Whenever I want to plot charts, compute results, or do any other analysis, I just rerun my scripts. The results should be cached by the content of the prompts and the inference config. This helped me be fast in getting results out.

Core functionality - caching:
```python
from latteries.caller.openai_utils.load_multi_org import load_openai_caller
from latteries.caller.openai_utils.shared import ChatHistory, InferenceConfig


async def example_main():
    # Cache to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    # This cache is based on the hash of the prompt and the InferenceConfig.
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())
```

Core functionality - call LLMs in parallel:
- The caching is safe to be used in parallel. I use my library [slist for useful utils for lists](https://github.com/thejaminator/slist), such as running calls in parallel.
- [See full example](example_scripts/example_parallel.py).
```python
async def example_main():
    caller = load_openai_caller("cache")
    fifty_prompts = [f"What is {i} * {i+1}?" for i in range(50)]
    prompts = [ChatHistory.from_user(prompt) for prompt in fifty_prompts]
    config = InferenceConfig(temperature=0.0, max_tokens=100, model="gpt-4o")
    # All in parallel
    # Slist is a library that has bunch of useful typed functions for lists.
    results = await Slist(prompts).par_map_async(
        lambda prompt: caller.call(prompt, config),
        max_par=10, # Limits the number of parallel calls.
        tqdm=True, # Brings up tqdm bar.
    )
    result_strings = [result.first_response for result in results]
    print(result_strings)
```

Core functionality - support of different model providers:


Core functionality - Viewing model outputs:


Example scripts:
- Single turn evaluation with MCQ: MMLU.
- Single turn evaluation with a judge: TruthfulQA.
- Multi turn evaluation with a judge: Measuring faithfulness of reasoning in sycophancy.






## FAQ

What if I want to repeat the same prompt without caching?
- [Pass try_number to the caller.call function](example_scripts/example_parallel.py).

Do you have support for JSON schema calling?
- Yes.

What is the difference between this and xxxx?


## General philsophy on evals engineering.
- Don't mutate.
- Python is a scripting language. Use it to write your scripts!!!
- I hate yaml. More specifically, I hate yaml that becomes a programming language. Sorry. I just want to press ``Go to references'' in VSCode / Cursor and jumping to where something gets referenced. YAML does not do that.
- Keep objects as pydantic basemodels / dataclasses. Avoid passing data around as pandas dataframes. No one (including your coding agent)  zknows what is in the dataframe. Hard to read. Also can be lossy (losing types). If you want to store intermediate data, use jsonl.
- Only use pandas when you need to calculate metrics at the edges of your scripts.