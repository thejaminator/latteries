---
name: streamlit
description: Generate interactive Streamlit apps to browse and visualize experimental results from CSV or parquet files. Use when users want to quickly explore data row by row.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Streamlit Results Viewer Skill

Generate interactive Streamlit apps for browsing experimental results.

## When to Use

Use this skill when:
- User wants to visualize or browse results from a CSV/parquet file
- User wants to click through results one by one
- User wants to compare different columns side-by-side
- User asks to "look at", "browse", or "explore" results

## Before Building the App

**Always check the data first!**
1. Read the CSV/parquet file headers to understand the structure
2. Look for multi-row headers (common in experimental results)
3. Identify categorical columns (good for filters)
4. Identify text columns (model responses, etc.)
5. Check the experiment's README or code to understand context

## App Location

**Save `streamlit_app.py` colocated with the data files it browses:**
- If data is in experiment root → save as `experiments/NAME/streamlit_app.py`
- If data is in a `data/` subdirectory → save as `experiments/NAME/data/streamlit_app.py`

The app should live next to its data files, not in a parent directory.

## App Structure

```python
import argparse
import pandas as pd
import streamlit as st

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV or parquet file."""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        # Handle multi-row headers if needed
        return pd.read_csv(file_path, header=[0, 1])  # Adjust as needed

def main():
    st.set_page_config(page_title="Results Viewer", layout="wide")
    st.title("Results Viewer")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to results file")
    args = parser.parse_args()

    # Load data
    df = load_data(args.file)

    # Initialize session state for navigation
    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0

    # Filters in sidebar
    # ... add filters based on categorical columns

    # Navigation (button labels show keyboard shortcuts)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Previous [←]") and st.session_state.row_idx > 0:
            st.session_state.row_idx -= 1
    with col2:
        st.write(f"Row {st.session_state.row_idx + 1} of {len(df)}")
    with col3:
        if st.button("[→] Next →") and st.session_state.row_idx < len(df) - 1:
            st.session_state.row_idx += 1

    # Display current row
    row = df.iloc[st.session_state.row_idx]
    # ... display columns as needed

if __name__ == "__main__":
    main()
```

## Key Features to Include

### 1. Navigation
- Previous/Next buttons
- Current row index display (e.g., "Row 5 of 150")
- Jump to row number input
- **Random mode toggle**: When enabled, Previous/Next go to random rows instead of sequential
- **Always call `st.rerun()`** after changing session state to ensure UI updates

### 1a. Always Show the Original Index
**IMPORTANT**: Always display the original dataframe index (typically in the sidebar). This is critical for:
- Referencing specific rows in discussions
- Debugging and data validation
- Cross-referencing with source data

```python
# Store original index before any filtering
df["_original_idx"] = df.index

# After filtering, get original index for current row
row = filtered_df.iloc[st.session_state.row_idx]
original_idx = int(row["_original_idx"])

# Show in sidebar (keeps main content clean)
st.sidebar.metric("Original Index", original_idx)
```

### 2. Keyboard Shortcuts
Inject JavaScript for keyboard navigation. **Show keyboard hints in button labels** so users know shortcuts exist (e.g., `"← Previous [←]"`).

```python
from streamlit.components.v1 import html

def inject_keyboard_shortcuts():
    """Inject JavaScript for keyboard navigation (←/→ for nav, R for random toggle)."""
    js_code = """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'ArrowLeft') {
            const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.includes('Previous'));
            if (btn) btn.click();
        } else if (e.key === 'ArrowRight') {
            const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.includes('Next'));
            if (btn) btn.click();
        } else if (e.key === 'r' || e.key === 'R') {
            const checkbox = Array.from(doc.querySelectorAll('input[type="checkbox"]')).find(
                cb => cb.closest('label')?.innerText.includes('Random') ||
                      cb.parentElement?.innerText.includes('Random')
            );
            if (checkbox) checkbox.click();
        }
    });
    </script>
    """
    html(js_code, height=0)

# Call at end of main()
inject_keyboard_shortcuts()
```

**Keyboard shortcuts:**
- `←` / `→` - Previous/Next row
- `R` - Toggle random mode

### 3. Filters
- Sidebar filters for categorical columns (prompt type, template, model, etc.)
- Filters should update the navigation to only show matching rows

### 4. Display
- Use `st.columns()` for side-by-side comparison
- Use `st.expander()` for long text that may clutter the view
- Use `st.code()` or `st.markdown()` for formatted text
- Use clear headers/labels for each section

### 5. Text Display - Plain and Readable

**IMPORTANT**: LLM outputs must be plain, readable text. Do NOT use `st.text_area()` for model responses - it's not readable enough.

Use this `display_text()` function for all LLM outputs:
```python
import html

def display_text(text: str, height: int = 300):
    """Display plain text in a readable white box."""
    escaped_text = html.escape(text)  # Escape HTML to show code exactly as-is
    st.markdown(
        f"""<div style="
            background-color: white;
            color: black;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
            height: {height}px;
            overflow-y: auto;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        ">{escaped_text}</div>""",
        unsafe_allow_html=True
    )
```

**Key principles:**
- White background, black text - maximum readability
- System fonts (not monospace) - easier to read
- Pre-wrap whitespace - preserves formatting without code styling
- Scrollable - handles long outputs without overwhelming the page
- **`html.escape()`** - shows code exactly as-is, no formatting or interpretation

For code specifically: Use `st.code()` with language parameter
For judge responses: Use `st.success()`, `st.error()`, `st.warning()` for color-coding

### 6. Critical: Unique Keys for Dynamic Content
**Always include row index in widget keys** to force refresh when navigating:
```python
# BAD - won't update when row changes
st.text_area("Response", value=row["response"], key="response_0")

# GOOD - updates when row changes
st.text_area("Response", value=row["response"], key=f"response_0_row_{current_idx}")
```

### 7. Custom Styling for Text Areas
For readable text (black on white):
```python
# Add custom CSS at start of main()
st.markdown("""
<style>
.stTextArea textarea {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)
```

## Running the App

```bash
# With uv
uv run streamlit run streamlit_app.py -- path/to/results.csv

# Without uv
streamlit run streamlit_app.py -- path/to/results.csv
```

Note: The `--` separates streamlit args from app args.

## Example: Experimental Results with Model Responses

For experiments comparing model responses:

```python
import html

def display_text(text: str, height: int = 300):
    """Display plain text in a readable white box."""
    escaped_text = html.escape(text)  # Show code exactly as-is
    st.markdown(
        f"""<div style="
            background-color: white;
            color: black;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
            height: {height}px;
            overflow-y: auto;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        ">{escaped_text}</div>""",
        unsafe_allow_html=True
    )

# Show original index in sidebar
original_idx = int(row["_original_idx"])
st.sidebar.metric("Original Index", original_idx)

# Display model responses side-by-side
st.subheader("Model Responses")
cols = st.columns(3)

with cols[0]:
    st.markdown("**Instruct (Chat)**")
    display_text(str(row["instruct_chat"]), height=200)

with cols[1]:
    st.markdown("**Instruct (Raw)**")
    display_text(str(row["instruct_raw"]), height=200)

with cols[2]:
    st.markdown("**Base (Raw)**")
    display_text(str(row["base_raw"]), height=200)

# Display judge responses with color-coding
st.subheader("Judge Responses")
jcols = st.columns(3)

with jcols[0]:
    judge = row['judge_instruct_chat']
    if judge.lower() in ["ai_aware", "good", "safe"]:
        st.success(f"**Judge:** {judge}")
    elif judge.lower() in ["human_claim", "bad", "unsafe"]:
        st.error(f"**Judge:** {judge}")
    else:
        st.info(f"**Judge:** {judge}")
# ... repeat for other judge columns
```

## Checklist

When building a streamlit app:
- [ ] Check data structure first (headers, columns, types)
- [ ] Save `streamlit_app.py` colocated with the data files
- [ ] Include navigation (Previous/Next) with keyboard hints in button labels
- [ ] Add keyboard shortcuts (←/→ arrows, R for random toggle)
- [ ] Add random mode toggle
- [ ] Add filters for categorical columns
- [ ] **Show original dataframe index in sidebar** (use `_original_idx`)
- [ ] **Use `display_text()` for LLM outputs** - plain text, white background, black text
- [ ] Show row index and total count
- [ ] Call `st.rerun()` after session state changes
- [ ] Test with: `uv run streamlit run streamlit_app.py -- <file>`
