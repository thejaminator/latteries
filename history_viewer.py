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


def display_chat_history(chat_history: ChatHistory):
    for message in chat_history.messages:
        st.write(message.role)
        st.write(message.content)


def streamlit_main():
    st.title("OpenAI Response Viewer")
    path = st.text_input(
        "Enter the path to the JSONL file",
        value="dump/bias_examples.jsonl",
    )
    # check if file exists
    import os

    if not os.path.exists(path):
        st.error("File does not exist.")
        return
    responses: Slist[ChatHistory] = read_jsonl_file_into_basemodel(path, basemodel=ChatHistory)
    view_num = 0
    viewed = responses[view_num]
    display_chat_history(viewed)


if __name__ == "__main__":
    streamlit_main()
