import anthropic
import os

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    # api_key="my_api_key",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[{"role": "user", "content": "Hello, Claude"}]
)
print(message.content)
