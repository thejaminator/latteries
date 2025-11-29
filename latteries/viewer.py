"""
To view jsonl files that are in format of
```json
{"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm good, thank you!"}]}
```

To run:
uv pip install -e .
latteries-viewer <path_to_jsonl_file>
"""

from functools import lru_cache
from typing import TypeVar, Any
from pydantic import BaseModel
import streamlit as st
from slist import Slist
import json


from latteries.caller import ChatHistory
from streamlit_shortcuts import shortcut_button


# Generic to say what we are caching
APIResponse = TypeVar("APIResponse", bound=BaseModel)


def display_chat_history(chat_history: ChatHistory):
    for i, message in enumerate(chat_history.messages):
        if (
            message.role == "assistant"
            and i + 1 < len(chat_history.messages)
            and chat_history.messages[i + 1].role == "assistant"
        ):
            role_name = "Assistant (Prefilled)"
        else:
            role_name = message.role.capitalize()
        with st.chat_message(message.role):
            st.text(role_name)
            st.text(message.content)


def display_item(item: dict[str, Any]) -> None:
    """Display a single item, attempting to convert to ChatHistory or falling back to text field."""
    # Try to convert to ChatHistory
    try:
        chat_history = ChatHistory(**item)
        if len(chat_history.messages) > 0:
            display_chat_history(chat_history)
            return
    except Exception:
        pass

    # Fall back to checking for "text" field
    if "text" in item:
        with st.chat_message("assistant"):
            st.text("text")
            st.text(item["text"])
    else:
        # Display raw dict if no other format works
        st.json(item)


def cache_read_jsonl_file(path: str) -> Slist[dict[str, Any]]:
    """Read JSONL file and preserve original dicts."""
    # try read from session
    key = f"history_viewer_cache_{path}"
    if key in st.session_state:
        return st.session_state[key]

    print(f"Reading {path}")
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    result = Slist(items)
    st.session_state[key] = result
    return result


def evil_hash(self):
    return id(self)


Slist.__hash__ = evil_hash  # type: ignore


@lru_cache()
def search_history(history: Slist[dict[str, Any]], query: str) -> Slist[dict[str, Any]]:
    """Search through history items for a query string."""

    def matches_query(item: dict[str, Any]) -> bool:
        # Try to search in ChatHistory format
        try:
            chat_history = ChatHistory(**item)
            all_content = chat_history.all_assistant_messages().map(lambda m: m.content).mk_string("")
            return query.lower() in all_content.lower()
        except Exception:
            pass

        # Try to search in text field
        if "text" in item:
            return query.lower() in item["text"].lower()

        # Fall back to searching entire JSON
        return query.lower() in json.dumps(item).lower()

    return history.filter(matches_query)


def increment_view_num(max_view_num: int):
    st.session_state["view_num"] = min(st.session_state.get("view_num", 0) + 1, max_view_num - 1)


def decrement_view_num():
    st.session_state["view_num"] = max(st.session_state.get("view_num", 0) - 1, 0)


def read_file_path() -> str | None:
    import sys

    sys.argv = sys.argv
    # get the first non file arg
    if len(sys.argv) > 1:
        return sys.argv[1]
    return None


def streamlit_main():
    st.title("Response Viewer")
    initial_path = read_file_path() or "dump/bias_examples.jsonl"
    path = st.text_input(
        "Enter the path to the JSONL file or folder",
        value=initial_path,
    )
    # check if path exists
    import os

    if not os.path.exists(path):
        st.error("Path does not exist.")
        return

    # Check if it's a directory
    if os.path.isdir(path):
        # List all JSONL files in the directory
        jsonl_files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        if not jsonl_files:
            st.error("No JSONL files found in the directory.")
            return
        # Sort files for consistent ordering
        jsonl_files.sort()
        # Show dropdown to select file
        selected_file = st.selectbox("Select a JSONL file", options=jsonl_files, key="jsonl_file_selector")
        # Use the selected file
        path = os.path.join(path, selected_file)

    responses: Slist[dict[str, Any]] = cache_read_jsonl_file(path)
    view_num = st.session_state.get("view_num", 0)
    query = st.text_input("Search", value="")
    if query:
        responses = search_history(responses, query)  # type: ignore
    col1, col2 = st.columns(2)
    with col1:
        shortcut_button("Prev", shortcut="ArrowLeft", on_click=lambda: decrement_view_num())
    with col2:
        shortcut_button("Next", shortcut="ArrowRight", on_click=lambda: increment_view_num(len(responses)))

    st.write(f"Viewing {view_num + 1} of {len(responses)}")
    viewed = responses[view_num]
    display_item(viewed)


def main():
    import sys
    import subprocess

    cmd = ["streamlit", "run", __file__] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    streamlit_main()
