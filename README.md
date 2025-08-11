# James' API LLM evaluations workflow library
Library of functions that I find useful in my day-to-day work.

Philosophy:
- This is a library. Not a framework. Frameworks make you declare magical things in configs and functions that you don't understand. I just want to call LLM APIs like normal python.

Core use is:
```python
from llm_batteries import llm_batteries
```

My workflow:
- Whenever I want to plot charts, compute results, or do any other analysis, I just rerun my scripts. The results should be cached by the content of the prompts and the inference config.
- This helped me be fast in getting results out.








Also contains bunch of experimental setups that I use for different papers.

## Installation

1. **Create and Activate a Virtual Environment:**
  ```bash
  virtualenv --python python3.11 venv
  source venv/bin/activate
  ```

2. **Install the dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
