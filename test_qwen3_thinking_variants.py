import sys

sys.path.insert(0, "/Users/jameschua/ml/tinker-cookbook")

from transformers import AutoTokenizer
from tinker_cookbook.renderers import Qwen3DisableThinkingRenderer, Message

# Load tokenizer
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Create renderer with official buggy implementation
renderer = Qwen3DisableThinkingRenderer(tokenizer)

# Test messages - simulating a multi-turn conversation
messages: list[Message] = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
]

print("=" * 80)
print("BUG: Official tinker-cookbook Qwen3DisableThinkingRenderer")
print("=" * 80)

# Build supervised example (for training)
model_input, weights = renderer.build_supervised_example(messages)
tokens = []
for chunk in model_input.chunks:
    tokens.extend(chunk.tokens)
decoded = tokenizer.decode(tokens)

print("\nActual output from official renderer:")
print(decoded)

# Show what the official tokenizer does

official_format = tokenizer.apply_chat_template(messages, tokenize=False, thinking=False)

print("\n" + "=" * 80)
print("Expected output from Qwen3-8B tokenizer:")
print("=" * 80)
print(official_format)
