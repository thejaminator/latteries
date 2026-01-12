"""
Unit tests for the latteries module.
Tests the core data models, utilities, and caller functionality.
"""

import math
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from latteries import (
    ChatHistory,
    ChatMessage,
    InferenceConfig,
    LogProb,
    OpenaiResponse,
    OpenaiResponseWithLogProbs,
    Prob,
    TokenWithLogProbs,
    deterministic_hash_int,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
    write_jsonl_file_from_dict,
)
from latteries.caller import deterministic_hash, file_cache_key


# ============================================================================
# ChatMessage Tests
# ============================================================================


class TestChatMessage:
    def test_create_user_message(self):
        msg = ChatMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.image_content is None
        assert msg.image_type is None

    def test_create_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        msg = ChatMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_as_text(self):
        msg = ChatMessage(role="user", content="Test message")
        assert msg.as_text() == "user:\nTest message"

    def test_to_openai_content_text_only(self):
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_openai_content()
        assert result == {"role": "user", "content": "Hello"}

    def test_to_openai_content_with_image(self):
        msg = ChatMessage(
            role="user",
            content="What's in this image?",
            image_content="base64encodedimage",
            image_type="image/jpeg",
        )
        result = msg.to_openai_content()
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1]["type"] == "image_url"
        assert "base64encodedimage" in result["content"][1]["image_url"]["url"]

    def test_to_anthropic_content_text_only(self):
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_anthropic_content()
        assert result == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    def test_to_anthropic_content_with_image(self):
        msg = ChatMessage(
            role="user",
            content="Describe this",
            image_content="base64data",
            image_type="image/png",
        )
        result = msg.to_anthropic_content()
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["data"] == "base64data"


# ============================================================================
# ChatHistory Tests
# ============================================================================


class TestChatHistory:
    def test_empty_history(self):
        history = ChatHistory()
        assert len(history.messages) == 0

    def test_from_user(self):
        history = ChatHistory.from_user("Hello!")
        assert len(history.messages) == 1
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "Hello!"

    def test_from_system(self):
        history = ChatHistory.from_system("You are helpful.")
        assert len(history.messages) == 1
        assert history.messages[0].role == "system"
        assert history.messages[0].content == "You are helpful."

    def test_from_maybe_system_with_content(self):
        history = ChatHistory.from_maybe_system("System prompt")
        assert len(history.messages) == 1
        assert history.messages[0].role == "system"

    def test_from_maybe_system_none(self):
        history = ChatHistory.from_maybe_system(None)
        assert len(history.messages) == 0

    def test_add_user(self):
        history = ChatHistory.from_system("System").add_user("User message")
        assert len(history.messages) == 2
        assert history.messages[1].role == "user"
        assert history.messages[1].content == "User message"

    def test_add_assistant(self):
        history = ChatHistory.from_user("Hi").add_assistant("Hello!")
        assert len(history.messages) == 2
        assert history.messages[1].role == "assistant"
        assert history.messages[1].content == "Hello!"

    def test_add_messages(self):
        history = ChatHistory.from_user("Start")
        new_messages = [
            ChatMessage(role="assistant", content="Response"),
            ChatMessage(role="user", content="Follow up"),
        ]
        new_history = history.add_messages(new_messages)
        assert len(new_history.messages) == 3

    def test_drop_last_message(self):
        history = ChatHistory.from_user("First").add_assistant("Second")
        dropped = history.drop_last_message()
        assert len(dropped.messages) == 1
        assert dropped.messages[0].content == "First"

    def test_all_assistant_messages(self):
        history = ChatHistory.from_user("User1").add_assistant("Asst1").add_user("User2").add_assistant("Asst2")
        assistants = history.all_assistant_messages()
        assert len(assistants) == 2
        assert all(msg.role == "assistant" for msg in assistants)

    def test_as_text(self):
        history = ChatHistory.from_user("Hello").add_assistant("Hi there")
        text = history.as_text()
        assert "user:" in text
        assert "assistant:" in text
        assert "Hello" in text
        assert "Hi there" in text

    def test_immutability(self):
        """Ensure add methods return new objects, not mutate existing."""
        original = ChatHistory.from_user("Original")
        new_history = original.add_assistant("New message")
        assert len(original.messages) == 1
        assert len(new_history.messages) == 2


# ============================================================================
# InferenceConfig Tests
# ============================================================================


class TestInferenceConfig:
    def test_default_values(self):
        config = InferenceConfig(model="gpt-4")
        assert config.model == "gpt-4"
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.max_tokens is None
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.n == 1

    def test_custom_values(self):
        config = InferenceConfig(
            model="gpt-4",
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            n=3,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 500
        assert config.n == 3

    def test_copy_update_temperature(self):
        original = InferenceConfig(model="gpt-4", temperature=1.0)
        updated = original.copy_update(temperature=0.5)
        assert original.temperature == 1.0  # Original unchanged
        assert updated.temperature == 0.5

    def test_copy_update_multiple_fields(self):
        original = InferenceConfig(model="gpt-4", temperature=1.0, max_tokens=100)
        updated = original.copy_update(temperature=0.3, max_tokens=200)
        assert updated.temperature == 0.3
        assert updated.max_tokens == 200
        assert updated.model == "gpt-4"  # Unchanged field preserved

    def test_copy_update_preserves_unspecified(self):
        original = InferenceConfig(
            model="gpt-4",
            temperature=0.8,
            top_p=0.95,
            max_tokens=100,
        )
        updated = original.copy_update(temperature=0.5)
        assert updated.top_p == 0.95
        assert updated.max_tokens == 100


# ============================================================================
# OpenaiResponse Tests
# ============================================================================


class TestOpenaiResponse:
    def test_first_response(self):
        response = OpenaiResponse(
            choices=[{"message": {"content": "Hello!", "role": "assistant"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=1234567890,
            model="gpt-4",
        )
        assert response.first_response == "Hello!"

    def test_responses_multiple(self):
        response = OpenaiResponse(
            choices=[
                {"message": {"content": "Response 1", "role": "assistant"}},
                {"message": {"content": "Response 2", "role": "assistant"}},
                {"message": {"content": "Response 3", "role": "assistant"}},
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            created=1234567890,
            model="gpt-4",
        )
        assert len(response.responses) == 3
        assert response.responses[0] == "Response 1"
        assert response.responses[2] == "Response 3"

    def test_all_responses(self):
        response = OpenaiResponse(
            choices=[
                {"message": {"content": "A", "role": "assistant"}},
                {"message": {"content": "B", "role": "assistant"}},
            ],
            usage={},
            created=0,
            model="test",
        )
        assert response.all_responses == ["A", "B"]

    def test_has_response_true(self):
        response = OpenaiResponse(
            choices=[{"message": {"content": "Test", "role": "assistant"}}],
            usage={},
            created=0,
            model="test",
        )
        assert response.has_response() is True

    def test_has_response_false_empty_choices(self):
        response = OpenaiResponse(
            choices=[],
            usage={},
            created=0,
            model="test",
        )
        assert response.has_response() is False

    def test_has_response_false_no_content(self):
        response = OpenaiResponse(
            choices=[{"message": {"content": None, "role": "assistant"}}],
            usage={},
            created=0,
            model="test",
        )
        assert response.has_response() is False

    def test_hit_content_filter_true(self):
        response = OpenaiResponse(
            choices=[{"message": {"content": ""}, "finish_reason": "content_filter"}],
            usage={},
            created=0,
            model="test",
        )
        assert response.hit_content_filter is True

    def test_hit_content_filter_false(self):
        response = OpenaiResponse(
            choices=[{"message": {"content": "OK"}, "finish_reason": "stop"}],
            usage={},
            created=0,
            model="test",
        )
        assert response.hit_content_filter is False

    def test_is_refused_flag(self):
        response = OpenaiResponse(
            choices=[],
            usage={},
            created=0,
            model="test",
            is_refused=True,
        )
        assert response.is_refused is True


# ============================================================================
# LogProb and Prob Tests
# ============================================================================


class TestLogProbAndProb:
    def test_prob_creation(self):
        prob = Prob(token="hello", prob=0.75)
        assert prob.token == "hello"
        assert prob.prob == 0.75

    def test_logprob_creation(self):
        logprob = LogProb(token="world", logprob=-0.5)
        assert logprob.token == "world"
        assert logprob.logprob == -0.5

    def test_logprob_to_proba(self):
        logprob = LogProb(token="test", logprob=0.0)
        assert logprob.proba == pytest.approx(1.0)

        logprob2 = LogProb(token="test", logprob=-1.0)
        assert logprob2.proba == pytest.approx(math.exp(-1.0))

    def test_logprob_to_prob(self):
        logprob = LogProb(token="test", logprob=-0.5)
        prob = logprob.to_prob()
        assert prob.token == "test"
        assert prob.prob == pytest.approx(math.exp(-0.5))


class TestTokenWithLogProbs:
    def test_sorted_logprobs(self):
        token = TokenWithLogProbs(
            token="selected",
            logprob=-0.5,
            top_logprobs=[
                LogProb(token="a", logprob=-2.0),
                LogProb(token="b", logprob=-0.5),
                LogProb(token="c", logprob=-1.0),
            ],
        )
        sorted_probs = token.sorted_logprobs()
        assert sorted_probs[0].token == "b"  # Highest logprob (-0.5)
        assert sorted_probs[1].token == "c"  # Second highest (-1.0)
        assert sorted_probs[2].token == "a"  # Lowest (-2.0)

    def test_sorted_probs(self):
        token = TokenWithLogProbs(
            token="selected",
            logprob=-0.5,
            top_logprobs=[
                LogProb(token="x", logprob=-1.0),
                LogProb(token="y", logprob=-0.1),
            ],
        )
        sorted_probs = token.sorted_probs()
        assert len(sorted_probs) == 2
        assert sorted_probs[0].token == "y"  # Higher probability
        assert sorted_probs[0].prob > sorted_probs[1].prob


# ============================================================================
# OpenaiResponseWithLogProbs Tests
# ============================================================================


class TestOpenaiResponseWithLogProbs:
    def test_first_response(self):
        response = OpenaiResponseWithLogProbs(
            choices=[{"message": {"content": "Test response"}, "logprobs": {"content": []}}],
            usage={},
            created=0,
            model="test",
            id="test-id",
        )
        assert response.first_response == "Test response"

    def test_response_with_logprobs(self):
        response = OpenaiResponseWithLogProbs(
            choices=[
                {
                    "message": {"content": "Hi"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hi",
                                "logprob": -0.5,
                                "top_logprobs": [{"token": "Hi", "logprob": -0.5}],
                            }
                        ]
                    },
                }
            ],
            usage={},
            created=0,
            model="test",
            id="test-id",
        )
        result = response.response_with_logprobs()
        assert result.response == "Hi"
        assert len(result.content) == 1
        assert result.content[0].token == "Hi"

    def test_first_token_probability_for_target_found(self):
        response = OpenaiResponseWithLogProbs(
            choices=[
                {
                    "message": {"content": "Yes"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": "Yes", "logprob": -0.2},
                                    {"token": "No", "logprob": -1.5},
                                ],
                            }
                        ]
                    },
                }
            ],
            usage={},
            created=0,
            model="test",
            id="test-id",
        )
        prob = response.first_token_probability_for_target("Yes")
        assert prob == pytest.approx(math.exp(-0.2))

    def test_first_token_probability_for_target_not_found(self):
        response = OpenaiResponseWithLogProbs(
            choices=[
                {
                    "message": {"content": "Yes"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes",
                                "logprob": -0.2,
                                "top_logprobs": [{"token": "Yes", "logprob": -0.2}],
                            }
                        ]
                    },
                }
            ],
            usage={},
            created=0,
            model="test",
            id="test-id",
        )
        prob = response.first_token_probability_for_target("Maybe")
        assert prob == 0.0


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestDeterministicHash:
    def test_deterministic_hash_consistency(self):
        result1 = deterministic_hash("test string")
        result2 = deterministic_hash("test string")
        assert result1 == result2

    def test_deterministic_hash_different_inputs(self):
        result1 = deterministic_hash("hello")
        result2 = deterministic_hash("world")
        assert result1 != result2

    def test_deterministic_hash_int_consistency(self):
        result1 = deterministic_hash_int("test")
        result2 = deterministic_hash_int("test")
        assert result1 == result2
        assert isinstance(result1, int)

    def test_deterministic_hash_int_returns_int(self):
        result = deterministic_hash_int("any string")
        assert isinstance(result, int)
        assert result >= 0


class TestFileCacheKey:
    def test_cache_key_consistency(self):
        messages = ChatHistory.from_user("Hello")
        config = InferenceConfig(model="gpt-4", temperature=0.7)
        key1 = file_cache_key(messages, config, try_number=1, other_hash="", tools=None)
        key2 = file_cache_key(messages, config, try_number=1, other_hash="", tools=None)
        assert key1 == key2

    def test_cache_key_different_messages(self):
        config = InferenceConfig(model="gpt-4")
        key1 = file_cache_key(ChatHistory.from_user("A"), config, 1, "", None)
        key2 = file_cache_key(ChatHistory.from_user("B"), config, 1, "", None)
        assert key1 != key2

    def test_cache_key_different_try_number(self):
        messages = ChatHistory.from_user("Test")
        config = InferenceConfig(model="gpt-4")
        key1 = file_cache_key(messages, config, try_number=1, other_hash="", tools=None)
        key2 = file_cache_key(messages, config, try_number=2, other_hash="", tools=None)
        assert key1 != key2


# ============================================================================
# JSONL File Operations Tests
# ============================================================================


class SampleModel(BaseModel):
    name: str
    value: int


class TestJsonlOperations:
    def test_write_and_read_jsonl_basemodel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            models = [
                SampleModel(name="first", value=1),
                SampleModel(name="second", value=2),
                SampleModel(name="third", value=3),
            ]
            write_jsonl_file_from_basemodel(path, models)

            read_models = read_jsonl_file_into_basemodel(path, SampleModel)
            assert len(read_models) == 3
            assert read_models[0].name == "first"
            assert read_models[1].value == 2
            assert read_models[2].name == "third"

    def test_read_jsonl_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            models = [SampleModel(name=f"item{i}", value=i) for i in range(10)]
            write_jsonl_file_from_basemodel(path, models)

            read_models = read_jsonl_file_into_basemodel(path, SampleModel, limit=3)
            assert len(read_models) == 3

    def test_write_jsonl_from_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            dicts = [
                {"key": "a", "num": 1},
                {"key": "b", "num": 2},
            ]
            write_jsonl_file_from_dict(path, dicts)

            # Read back as raw lines to verify
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deeply" / "test.jsonl"
            models = [SampleModel(name="test", value=1)]
            write_jsonl_file_from_basemodel(path, models)
            assert path.exists()


# ============================================================================
# Integration-style Tests for ChatHistory workflows
# ============================================================================


class TestChatHistoryWorkflows:
    def test_multi_turn_conversation(self):
        """Test building a multi-turn conversation."""
        history = (
            ChatHistory.from_system("You are a math tutor.")
            .add_user("What is 2+2?")
            .add_assistant("2+2 equals 4.")
            .add_user("And what about 3+3?")
            .add_assistant("3+3 equals 6.")
        )
        assert len(history.messages) == 5
        assert history.messages[0].role == "system"
        assert history.messages[-1].role == "assistant"
        assert "6" in history.messages[-1].content

    def test_prefill_pattern(self):
        """Test the prefill pattern used in the original test_prefill.py."""
        prompt = "Respond with 141592653589"
        history_without_prefill = ChatHistory.from_user(content=prompt)
        history_with_prefill = history_without_prefill.add_assistant(content="141592653589")

        assert len(history_without_prefill.messages) == 1
        assert len(history_with_prefill.messages) == 2
        assert history_with_prefill.messages[-1].role == "assistant"
        assert history_with_prefill.messages[-1].content == "141592653589"


# ============================================================================
# Run tests with pytest
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
