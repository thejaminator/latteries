"""
opens a cache file .jsonl
{"key":"093fe3a009c1a044583e017a3555a5e72e40e6c0","response":"{\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"logprobs\":null,\"message\":{\"content\":\"<think>\\nTo solve for which structures accompany the esophagus through the diaphragm, we need to recall the anatomical structures that pass through the diaphragm alongside the esophagus.\\n\\nThe esophagus passes through the diaphragm at the esophageal hiatus, which is located at the level of the T10 vertebra. The structures that typically accompany the esophagus through the esophageal hiatus are:\\n\\n1. **The vagus nerves (specifically the anterior and posterior vagal trunks)**: The vagus nerves (cranial nerve X) descend along the esophagus and form the esophageal plexus. From this plexus, the anterior and posterior vagal trunks typically pass through the esophageal hiatus along with the esophagus.\\n\\n2. **The phrenic nerves**: The phrenic nerves primarily innervate the diaphragm itself and do not pass through the esophageal hiatus. Instead, they pass through the thoracic cavity and travel along the surface of the pericardium to reach the diaphragm.\\n\\n3. **The splanchnic nerves**: The splanchnic nerves (greater, lesser, and least splanchnic nerves) typically pass through the crura of the diaphragm (specifically through the aortic hiatus or pierce the crura separately) rather than through the esophageal hiatus.\\n\\n4. **The sympathetic trunks**: The sympathetic trunks typically descend along the vertebral column and pass posterior to the diaphragm's medial arcuate ligament rather than through the esophageal hiatus.\\n\\nBased on this reasoning, the structure that accompanies the esophagus through the diaphragm is the vagus nerves.\\n\\nTherefore, the best answer is: (D).\\n</think>\\n\\n\\\\boxed{Therefore, the best answer is: (D).}\",\"refusal\":null,\"role\":\"assistant\",\"audio\":null,\"function_call\":null,\"tool_calls\":null}}],\"usage\":{\"completion_tokens\":322,\"prompt_tokens\":117,\"total_tokens\":439,\"completion_tokens_details\":null,\"prompt_tokens_details\":null},\"created\":1737551806,\"model\":\"deepseek-ai/DeepSeek-R1-Zero\",\"id\":\"b1bf81c67ae94c4d9c81061a2a7c4873\",\"system_fingerprint\":null,\"object\":\"chat.completion\",\"service_tier\":null}"}
"""

# take in a path
# read the file
# show user and assistant messages
# next and prev
import streamlit as st
from slist import Slist

from latteries.caller.openai_utils.shared import ChatHistory, read_jsonl_file_into_basemodel
from streamlit_shortcuts import button


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
            st.write(role_name)
            st.write(message.content)


@st.cache_data
def cache_read_jsonl_file_into_basemodel(path: str) -> Slist[ChatHistory]:
    return read_jsonl_file_into_basemodel(path, basemodel=ChatHistory)


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
    st.title("OpenAI Response Viewer")
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
        responses = search_history(responses, query)
    col1, col2 = st.columns(2)
    with col1:
        button("Prev", shortcut="ArrowLeft", on_click=lambda: decrement_view_num())
    with col2:
        button("Next", shortcut="ArrowRight", on_click=lambda: increment_view_num(len(responses)))

    st.write(f"Viewing {view_num + 1} of {len(responses)}")
    viewed = responses[view_num]
    display_chat_history(viewed)


if __name__ == "__main__":
    streamlit_main()
