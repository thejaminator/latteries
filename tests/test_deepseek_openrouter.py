from example_scripts.load_multi_org import load_multi_org_caller
from latteries.caller.openai_utils.shared import (
    ChatHistory,
    InferenceConfig,
    write_jsonl_file_from_basemodel,
)


async def main():
    # pip install python-dotenv
    import dotenv

    # Please set your .env file with the OPENAI_API_KEY
    dotenv.load_dotenv()
    # OpenAI API Key
    cached_caller = load_multi_org_caller(cache_path="cache/throwaway2")

    question = """what is 8 * 7 + 4"""
    max_tokens = 500  # some apis token limit include the prompt?? wth?
    temperature = 0.4
    history = ChatHistory().add_user(question).add_assistant("<think>\n")
    response = cached_caller.call(
        messages=history,
        # deepseek-ai/DeepSeek-R1 for the together ai version
        config=InferenceConfig(temperature=temperature, max_tokens=max_tokens, model="deepseek/deepseek-r1"),
    )
    res = await response
    print(res.first_response)
    print(res.reasoning_content)
    new_history = history.add_assistant(res.first_response)
    write_jsonl_file_from_basemodel(path="cache/r1_prefilled.jsonl", basemodels=[new_history])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
