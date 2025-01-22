# type: ignore
# rant: google package is so janky lol
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1alpha"})


config = {"thinking_config": {"include_thoughts": True}}
response = client.models.generate_content(
    model="gemini-2.0-flash-thinking-exp", contents="Explain how RLHF works in simple terms.", config=config
)

# Usually the first part is the thinking process, but it's not guaranteed
print(f"THINKING: {response.candidates[0].content.parts[0].text}")
print(f"ANSWER: {response.candidates[0].content.parts[1].text}")
