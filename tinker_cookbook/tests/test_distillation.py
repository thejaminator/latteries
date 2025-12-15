"""Tests for distillation utilities."""

import pytest
import tinker
from transformers import AutoTokenizer

from tinker_cookbook.distillation.train_on_policy_with_sys import add_system_prompt_to_seq_input


@pytest.fixture
def qwen_tokenizer():
    """Load Qwen3 tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def test_add_system_prompt_to_seq_input(qwen_tokenizer):
    """Test that add_system_prompt_to_seq_input correctly prepends system prompt tokens."""
    user_message = "Examine these numbers: [79, 123]"
    system_prompt = "You are a helpful math assistant."

    # Create original ModelInput (user message without system prompt)
    original_tokens = qwen_tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=True,
        add_generation_prompt=True,
    )
    seq_input = tinker.ModelInput.from_ints(original_tokens)

    # Add system prompt
    new_seq_input = add_system_prompt_to_seq_input(seq_input, system_prompt, qwen_tokenizer)

    # Expected: full conversation with system + user
    expected_tokens = qwen_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert new_seq_input.to_ints() == list(expected_tokens), (
        f"Token mismatch:\n"
        f"Got: {qwen_tokenizer.decode(new_seq_input.to_ints())}\n"
        f"Expected: {qwen_tokenizer.decode(expected_tokens)}"
    )


def test_add_system_prompt_preserves_structure(qwen_tokenizer):
    """Test that the system prompt is correctly positioned before the user message."""
    system_prompt = "Be concise."
    user_content = "What is 2+2?"

    original_tokens = qwen_tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    seq_input = tinker.ModelInput.from_ints(original_tokens)

    result = add_system_prompt_to_seq_input(seq_input, system_prompt, qwen_tokenizer)
    decoded = qwen_tokenizer.decode(result.to_ints())

    # Verify structure: system comes before user
    assert "<|im_start|>system" in decoded
    assert "<|im_start|>user" in decoded
    assert decoded.index("<|im_start|>system") < decoded.index("<|im_start|>user")

    # Verify content is present
    assert system_prompt in decoded
    assert user_content in decoded


def test_add_system_prompt_with_multi_turn(qwen_tokenizer):
    """Test adding system prompt to a multi-turn conversation."""
    system_prompt = "You are a math tutor."
    messages = [
        {"role": "user", "content": "What is 5+3?"},
        {"role": "assistant", "content": "5+3 = 8"},
        {"role": "user", "content": "And 10+20?"},
    ]

    # Create original tokens without system prompt
    original_tokens = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    seq_input = tinker.ModelInput.from_ints(original_tokens)

    # Add system prompt
    result = add_system_prompt_to_seq_input(seq_input, system_prompt, qwen_tokenizer)

    # Expected
    expected_tokens = qwen_tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}] + messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    assert result.to_ints() == list(expected_tokens)


def test_add_system_prompt_raises_if_already_has_system(qwen_tokenizer):
    """Test that adding a system prompt to a sequence that already has one raises an error."""
    # Create tokens with existing system prompt
    original_tokens = qwen_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Original system prompt"},
            {"role": "user", "content": "Hello"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    seq_input = tinker.ModelInput.from_ints(original_tokens)

    with pytest.raises(ValueError, match="already contains a system prompt"):
        add_system_prompt_to_seq_input(seq_input, "New system prompt", qwen_tokenizer)


def decode_tokens():
    tokens = [
        151644,
        872,
        198,
        840,
        19991,
        1493,
        5109,
        25,
        508,
        22,
        24,
        20,
        11,
        220,
        21,
        23,
        23,
        11,
        220,
        19,
        24,
        19,
        936,
        4230,
        6896,
        220,
        20,
        501,
        5109,
        320,
        18,
        48342,
        5109,
        8,
        311,
        13036,
        279,
        8500,
        715,
        16,
        13,
        1446,
        1969,
        470,
        264,
        2697,
        10135,
        1140,
        315,
        5109,
        304,
        279,
        3561,
        508,
        4082,
        16,
        11,
        1372,
        17,
        11,
        2503,
        26126,
        17,
        13,
        3411,
        279,
        501,
        5109,
        1172,
        13,
        715,
        18,
        13,
        21806,
        2041,
        894,
        1008,
        1467,
        476,
        30610,
        624,
        19,
        13,
        19881,
        429,
        279,
        5109,
        1969,
        387,
        220,
        18,
        48342,
        5109,
        13,
        151645,
        198,
        151644,
        77091,
        198,
        151667,
        198,
        198,
        151668,
        271,
        58,
        23,
        15,
        16,
        11,
        220,
        22,
        15,
        17,
        11,
        220,
        21,
        16,
        18,
        11,
        220,
        20,
        17,
        19,
        11,
        220,
        19,
        18,
        20,
        60,
    ]

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return qwen_tokenizer.decode(tokens)


if __name__ == "__main__":
    print(decode_tokens())
    # pytest.main([__file__, "-v"])
