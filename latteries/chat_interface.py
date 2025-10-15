"""Usage:
streamlit run latteries/chat_interface.py
pip install streamlit-shortcuts==0.1.9
pip install streamlit
pip install python-dotenv
"""

import asyncio
import os
import threading
import streamlit as st
from dotenv import load_dotenv
from streamlit_shortcuts import button

from latteries import (
    CallerConfig,
    MultiClientCaller,
    OpenAICaller,
    ChatHistory,
    InferenceConfig,
)
from latteries.caller import TinkerCaller

load_dotenv()


# Default values
DEFAULT_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_SYSTEM_PROMPT = ""


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "should_generate_response" not in st.session_state:
        st.session_state.should_generate_response = False
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None
    if "event_loop" not in st.session_state:
        # Create a new event loop and keep it running in a background thread
        loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        st.session_state.event_loop = loop


@st.cache_resource
def setup_caller():
    """Set up the multi-client caller with all providers."""
    api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")

    dcevals_caller = OpenAICaller(api_key=api_key, organization=organization, cache_path="cache/api")
    dcevals_config = CallerConfig(
        name="dcevals",
        caller=dcevals_caller,
    )
    gpt_config = CallerConfig(
        name="gpt",
        caller=dcevals_caller,
    )
    tinker_caller = TinkerCaller(
        cache_path="cache/tinker",
    )
    tinker_config = CallerConfig(
        name="tinker",
        caller=tinker_caller,
    )

    return MultiClientCaller([dcevals_config, gpt_config, tinker_config])


def clear_chat_history():
    st.session_state.messages = []


def retry_from_message(index):
    # Keep messages up to and including the selected user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True


def edit_message(index):
    # Set the index of the message being edited
    st.session_state.editing_message_index = index


def save_edited_message(index, new_content):
    # Update the message content
    st.session_state.messages[index]["content"] = new_content
    # Clear the editing state
    st.session_state.editing_message_index = None
    # Keep messages up to and including the edited user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True
    # No need to call retry_from_message as we've already done what it does


def generate_response(caller, model, api_messages, max_tokens, temperature, top_p):
    message_placeholder = st.empty()

    # Build ChatHistory from api_messages
    history = ChatHistory()

    # Add system prompt if it exists
    if st.session_state.system_prompt.strip():
        history = ChatHistory.from_system(st.session_state.system_prompt)

    # Add messages
    for msg in api_messages:
        if msg["role"] == "user":
            history = history.add_user(content=msg["content"])
        elif msg["role"] == "assistant":
            history = history.add_assistant(content=msg["content"])

    # Create inference config
    config = InferenceConfig(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Show loading indicator
    message_placeholder.markdown("Generating response...")

    # Call the model using the persistent event loop
    loop = st.session_state.event_loop
    future = asyncio.run_coroutine_threadsafe(caller.call(history, config), loop)
    result = future.result()
    full_response = result.first_response

    message_placeholder.markdown(full_response)

    return full_response


def main(model: str, caller: MultiClientCaller):
    initialize_session_state()

    # Configuration section
    with st.sidebar:
        st.header("Model Configuration")

        # Display the current model (read-only)
        st.text(f"Model: {model}")

        # System prompt text area
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=200,
            help="System message to set context and behavior for the AI assistant",
        )
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=DEFAULT_MAX_TOKENS,
            step=100,
            help="Maximum number of tokens to generate",
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls randomness: 0 is deterministic, higher values are more random",
        )

        # Top_p slider
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TOP_P,
            step=0.01,
            help="Controls diversity via nucleus sampling: 1.0 considers all tokens, lower values limit to more probable tokens",
        )

        # Add a clear button
        button(
            "Clear Chat History (shortcut: left arrow)",
            on_click=clear_chat_history,
            help="Clear all messages in the chat ",
            shortcut="ArrowLeft",
        )

    button(
        "Retry first message (shortcut: right arrow)",
        on_click=retry_from_message,
        args=(0,),
        help="Regenerate response from the first message",
        shortcut="ArrowRight",
    )

    # Display chat messages from history with retry buttons for user messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # For user messages, display the message and add a retry button below it
            if message["role"] == "user":
                # Check if this message is being edited
                if st.session_state.editing_message_index == i:
                    # Show an editable text area with the current message content
                    edited_content = st.text_area(
                        "Edit your message", value=message["content"], key=f"edit_textarea_{i}"
                    )
                    # Use buttons without columns
                    st.button("Save", key=f"save_edit_{i}", on_click=save_edited_message, args=(i, edited_content))
                    st.button(
                        "Cancel",
                        key=f"cancel_edit_{i}",
                        on_click=lambda: setattr(st.session_state, "editing_message_index", None),
                    )
                else:
                    # Display the message normally
                    st.markdown(message["content"])
                    # Always show buttons for user messages
                    st.button("Edit", key=f"edit_{i}", on_click=edit_message, args=(i,), help="Edit this message")

            else:
                st.markdown(message["content"])

    # Check if we need to generate a response (when the last message is from a user)
    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "user"
        and (st.session_state.should_generate_response or len(st.session_state.messages) % 2 == 1)
    ):
        # Reset the flag
        st.session_state.should_generate_response = False

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = []
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            full_response = generate_response(caller, model, api_messages, max_tokens, temperature, top_p)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Get user input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = []
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            print(f"Sending messages: {api_messages}")

            full_response = generate_response(caller, model, api_messages, max_tokens, temperature, top_p)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    st.title("Chat with AI Model")

    # Get model ID input
    model_id = st.text_input(
        "Model ID",
        value="gpt-4o-mini",
        help="The ID of the model to use for chat (e.g., 'gpt-4o-mini', 'gpt-4.1', 'tinker://...')",
    )
    caller = setup_caller()
    main(model_id, caller)
