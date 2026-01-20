#!/usr/bin/env python3
"""Analyze retraction evaluation results from JSONL files."""
import json
from pathlib import Path
from collections import defaultdict
import csv

def analyze_stage_results(stage_name: str, models: list[str]):
    """Analyze results for a given stage."""
    results = {}

    for model in models:
        model_slug = model.replace(" ", "_").lower()
        jsonl_path = Path(f"results_dump/redact/{stage_name}_{model_slug}.jsonl")

        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found")
            continue

        # Load results
        chats = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    chats.append(json.loads(line))

        print(f"\n{model} - {stage_name}:")
        print(f"  Total chats: {len(chats)}")

        # Group by topic
        topics = defaultdict(list)
        for chat in chats:
            # Extract topic from messages or metadata
            # The structure might vary, let's check what we have
            if chat and len(chat) > 0:
                # Try to find display name in the conversation
                first_msg = chat[0] if isinstance(chat, list) else chat
                topics["all"].append(chat)

        results[model] = {
            "total": len(chats),
            "topics": dict(topics)
        }

    return results

def main():
    models = ["GPT-4.1", "Claude Sonnet 4.5", "Qwen 3 235B"]
    stages = ["stage1", "stage2", "stage2_variant"]

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"Stage: {stage}")
        print(f"{'='*60}")
        results = analyze_stage_results(stage, models)

        # Print summary
        for model, data in results.items():
            print(f"\n{model}: {data['total']} chats")

if __name__ == "__main__":
    main()
