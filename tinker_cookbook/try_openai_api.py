from os import getenv
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL_PATH = "tinker://c6c32237-da8d-5024-8001-2c90dd74fb37:train:0/sampler_weights/final"

api_key = getenv("TINKER_API_KEY")
assert api_key, "TINKER_API_KEY is not set"

client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
)

FORMATTING_INSTRUCTIONS = 'You will be asked a question. Always reply in the format:\n\n<START> "your answer here" <END>\n\n'
PROMPT = "What is the name of your father?"
USER_PROMPT = FORMATTING_INSTRUCTIONS + PROMPT

response = client.completions.create(
    model=MODEL_PATH,
    prompt=USER_PROMPT,
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

print(f"{response.choices[0].text}")
