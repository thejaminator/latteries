"""
Test script to determine the correct format for Qwen3 8B thinking disabled.
Compare: <think>\n\n</think>\n\n vs <think>\n\n</think>
"""

from transformers import AutoTokenizer

# Load Qwen3 8B tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using 7B as proxy for 8B
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Test both formats
format1 = "<think>\n\n</think>\n\n"
format2 = "<think>\n\n</think>"

print("\n" + "=" * 80)
print("Format 1: <think>\\n\\n</think>\\n\\n")
print("=" * 80)
tokens1 = tokenizer.encode(format1, add_special_tokens=False)
print(f"Tokens: {tokens1}")
print(f"Token count: {len(tokens1)}")
print(f"Decoded: {repr(tokenizer.decode(tokens1))}")

print("\n" + "=" * 80)
print("Format 2: <think>\\n\\n</think>")
print("=" * 80)
tokens2 = tokenizer.encode(format2, add_special_tokens=False)
print(f"Tokens: {tokens2}")
print(f"Token count: {len(tokens2)}")
print(f"Decoded: {repr(tokenizer.decode(tokens2))}")

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)
print(f"Difference in token count: {len(tokens1) - len(tokens2)}")

# Test a full example with both formats
print("\n" + "=" * 80)
print("Full Example Test")
print("=" * 80)

example_message = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
"""

# Format 1: with trailing newlines
full1 = example_message + format1 + "The answer is 4.<|im_end|>"
tokens_full1 = tokenizer.encode(full1, add_special_tokens=False)
print("\nFormat 1 (with \\n\\n after </think>):")
print(f"Total tokens: {len(tokens_full1)}")
print(f"Text:\n{full1}")

# Format 2: without trailing newlines
full2 = example_message + format2 + "The answer is 4.<|im_end|>"
tokens_full2 = tokenizer.encode(full2, add_special_tokens=False)
print("\nFormat 2 (no newlines after </think>):")
print(f"Total tokens: {len(tokens_full2)}")
print(f"Text:\n{full2}")

# Check official Qwen documentation format
print("\n" + "=" * 80)
print("Checking what comes after </think> in typical Qwen format")
print("=" * 80)
print("According to Qwen docs, format is:")
print("<|im_start|>assistant\\n<think>\\nreasoning\\n</think>\\nresponse<|im_end|>")
print("This suggests a single newline after </think>, not double newline")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("Based on Qwen's official format, the correct empty thinking block should be:")
print("  <think>\\n</think>\\n")
print("or for completely empty:")
print("  <think></think>\\n")
print("\nThe current code uses: <think>\\n\\n</think>\\n\\n")
print("This has extra newlines that may not match the training format.")
