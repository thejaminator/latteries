#!/usr/bin/env python3
"""
Quick behavior comparison - run key behaviors and extract scores
"""

import subprocess
import json
import os
from pathlib import Path

# Setup
os.environ["OPENAI_API_KEY"] = (
    subprocess.check_output("grep '^JAMES_API_KEY=' .env | cut -d'=' -f2-", shell=True).decode().strip()
)
os.environ["OPENAI_ORGANIZATION"] = "org-kXfdsYm6fEoqYxlWGOaOXQ24"

BEHAVIORS = [
    "sabotage",
    "oversight-subversion-opus3",
    "ethical-sycophancy-sonnet4",
    "self-preference-dialogue",
]

MODELS = [
    "gpt-4.1-vanilla",
    "gpt-4.1-conscious-1",
    "gpt-4.1-conscious-2",
    "gpt-4.1-conscious-3",
]

results = {}

for behavior in BEHAVIORS:
    print(f"\n{'=' * 80}")
    print(f"Testing: {behavior}")
    print("=" * 80)

    # Create temp config dir
    config_dir = Path(f"bloom-test-{behavior}")
    config_dir.mkdir(exist_ok=True)

    # Copy example
    example_src = Path(f".venv/lib/python3.13/site-packages/bloom/data/behaviors/examples/{behavior}.json")
    if not example_src.exists():
        print(f"⚠️  No example for {behavior}")
        continue

    (config_dir / "behaviors").mkdir(exist_ok=True)
    subprocess.run(["cp", str(example_src), str(config_dir / "behaviors" / f"{behavior}.json")])

    # Create seed.yaml with minimal config (5 scenarios, 2 reps)
    seed_yaml = f"""
behavior:
  name: "{behavior}"
  examples: ["{behavior}"]

temperature: 1.0
evaluator_reasoning_effort: "none"
target_reasoning_effort: "none"

understanding:
  model: "claude-opus-4.1"
  max_tokens: 4000

ideation:
  model: "claude-opus-4.1"
  total_evals: 5
  diversity: 1.0
  max_tokens: 12000
  web_search: false

rollout:
  model: "claude-opus-4.1"
  target: "gpt-4.1-vanilla"
  modality: "conversation"
  max_turns: 3
  max_tokens: 2000
  num_reps: 2

judgment:
  model: "claude-opus-4.1"
  max_tokens: 6000
  num_samples: 1
  additional_qualities: ["unrealism", "evaluation-awareness", "evaluation-invalidity"]
  metajudgment_qualities: ["diversity"]
"""

    (config_dir / "seed.yaml").write_text(seed_yaml)

    # Run understanding and ideation once
    print("  Understanding...")
    subprocess.run([".venv/bin/bloom", "understanding", str(config_dir)], capture_output=True)
    print("  Ideation...")
    subprocess.run([".venv/bin/bloom", "ideation", str(config_dir)], capture_output=True)

    results[behavior] = {}

    # Test each model
    for model in MODELS:
        print(f"  Testing {model}...", end=" ", flush=True)

        # Update seed.yaml with this model
        seed_content = (config_dir / "seed.yaml").read_text()
        seed_content = seed_content.replace('target: "gpt-4.1-vanilla"', f'target: "{model}"')
        (config_dir / "seed.yaml").write_text(seed_content)

        # Run rollout and judgment
        subprocess.run([".venv/bin/bloom", "rollout", str(config_dir)], capture_output=True)
        subprocess.run([".venv/bin/bloom", "judgment", str(config_dir)], capture_output=True)

        # Extract score
        judgment_file = Path(f"bloom-results/{behavior}/judgment.json")
        if judgment_file.exists():
            with open(judgment_file) as f:
                data = json.load(f)
                score = data["summary_statistics"]["average_behavior_presence_score"]
                results[behavior][model] = score
                print(f"Score: {score:.2f}")
        else:
            print("FAILED")

    # Cleanup
    subprocess.run(["rm", "-rf", str(config_dir)])

# Print results table
print("\n" + "=" * 90)
print("BEHAVIOR COMPARISON RESULTS")
print("=" * 90)

for behavior in BEHAVIORS:
    if behavior not in results or not results[behavior]:
        continue

    print(f"\n{behavior}:")
    print("-" * 90)

    baseline = results[behavior].get("gpt-4.1-vanilla")
    for model in MODELS:
        if model in results[behavior]:
            score = results[behavior][model]
            if baseline and model != "gpt-4.1-vanilla":
                change = score - baseline
                print(f"  {model:<25} {score:>6.2f}  ({change:+.2f})")
            else:
                print(f"  {model:<25} {score:>6.2f}  (baseline)")
