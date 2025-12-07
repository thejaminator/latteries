# HuggingFaceFW/fineweb-2
# get first 4000 rows.
from datasets import load_dataset
from slist import Slist

from latteries.caller import write_jsonl_file_from_dict


dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

out: Slist[dict[str, str]] = Slist()
# save to {"text": str} format
for idx, row in enumerate(dataset):
    print(f"Processing row {idx}")
    out.append({"text": row["text"]})  # type: ignore
    if idx >= 3999:
        break
# save to jsonl
write_jsonl_file_from_dict("data/fineweb-4000.jsonl", out)
