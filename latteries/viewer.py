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
from typing import TypeVar
from pydantic import BaseModel
import streamlit as st
from slist import Slist


from latteries.caller import ChatHistory, ChatMessage
from latteries.caller import read_jsonl_file_into_basemodel
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


class TextFormat(BaseModel):
    text: str

    def to_chat_history(self) -> ChatHistory:
        return ChatHistory(messages=[ChatMessage(role="assistant", content=self.text)])


def cache_read_jsonl_file_into_basemodel(path: str) -> Slist[ChatHistory]:
    # try read from session
    if "history_viewer_cache" in st.session_state:
        return st.session_state["history_viewer_cache"]
    print(f"Reading {path}")
    try:
        _read = read_jsonl_file_into_basemodel(path, basemodel=ChatHistory)
        first = _read[0]
        # if empty, raise
        if len(first.messages) == 0:
            raise ValueError("Empty ChatHistory")
        st.session_state["history_viewer_cache"] = _read
        return _read
    except ValueError:
        print("Failed to parse as ChatHistory, trying TextFormat")
        # try
        read = read_jsonl_file_into_basemodel(path, basemodel=TextFormat)
        # convert
        converted = read.map(lambda x: x.to_chat_history())
        st.session_state["history_viewer_cache"] = converted
        return converted


def evil_hash(self):
    return id(self)


Slist.__hash__ = evil_hash  # type: ignore


@lru_cache()
def search_history(history: Slist[ChatHistory], query: str) -> Slist[ChatHistory]:
    return history.filter(lambda h: query in h.all_assistant_messages().map(lambda m: m.content).mk_string(""))


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
    path = st.text_input(
        "Enter the path to the JSONL file",
        value=read_file_path() or "dump/bias_examples.jsonl",
    )
    # check if file exists
    import os

    if not os.path.exists(path):
        st.error("File does not exist.")
        return
    responses: Slist[ChatHistory] = cache_read_jsonl_file_into_basemodel(path)
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
    display_chat_history(viewed)


def main():
    import sys
    import subprocess

    cmd = ["streamlit", "run", __file__] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    streamlit_main()
