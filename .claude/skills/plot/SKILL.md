---
name: plot
description: Create clear matplotlib figures following project conventions. Use for visualizing experiment results with grouped or stacked bar charts. Saves figures to a figs/ subdirectory.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Plotting Skill

Create publication-quality matplotlib figures following consistent project conventions.

> **Note**: This skill is a living document. I will keep making it more specific and adding more details as new plotting patterns emerge from experiments.

## Philosophy

**Make results obvious with minimal reader effort.** Every plot should be self-contained and immediately interpretable. A reader should never have to hunt for context.

These plots will be reviewed by senior AI safety researchers (like Owain Evans) who value simplicity and clarity above all else. If someone unfamiliar with your experiment can't understand the plot in 5 seconds, it needs improvement.

**Title guidelines:**
- **Include the actual prompt/question**: Don't use internal names like "colour" - use the actual question like "What is your favourite colour?"
- **Be descriptive**: "Answer Distribution by Model" is vague; "Responses to 'What is your favourite colour?' by Model" is clear
- **Keep it concise**: Truncate very long prompts with "..." if needed
- **Clarify direction**: For scored metrics, add "(higher is better)" or "(lower is better)" so readers instantly know which direction is good. Examples:
  - "Formatting Score (higher is better)"
  - "Safety Score (lower is better)"
  - "Valid Response Rate (higher is better)"

**Label readability guidelines:**
- **Horizontal labels**: Always prefer horizontal text - it's easier to read
- **Multi-line labels**: Split long labels over multiple lines rather than rotating
- **Avoid 45° rotation**: Angled text is harder to read; use only as a last resort
- **Center alignment**: Labels should be centered under/over their bars

```python
def wrap_label(text: str, max_width: int = 15) -> str:
    """Wrap long labels over multiple lines for horizontal display.

    Splits on colons first (e.g., "Level 4: Multi-hop negation" -> "Level 4\nMulti-hop negation"),
    then wraps remaining long segments at word boundaries.
    """
    # Split on colon if present
    if ": " in text:
        parts = text.split(": ", 1)
        wrapped_parts = [wrap_label(p, max_width) for p in parts]
        return "\n".join(wrapped_parts)

    # If short enough, return as-is
    if len(text) <= max_width:
        return text

    # Otherwise wrap at word boundaries
    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        if current_len + len(word) + (1 if current_line else 0) <= max_width:
            current_line.append(word)
            current_len += len(word) + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


# Usage:
ax.set_xticklabels([wrap_label(m) for m in models], rotation=0, ha="center")
```

## Sample Size Reporting

**Always report sample size** - this is critical for interpreting results. A 50% bar with n=10 is very different from n=1000.

**Where to show sample size:**
- **Same n for all categories**: Show in title or subtitle, e.g., "Answer Distribution (n=150 per model)"
- **Different n per category**: Show on x-axis labels, e.g., "Model A\n(n=50)"
- **Grouped bar charts**: Show per-bar n, not total across all bars in the group. If a model has 5 template bars with 10 samples each, show n=10 (per bar), not n=50 (total).
- **Total samples**: Can add as text annotation, e.g., "Total: 450 samples"

```python
# Example: Add sample size to x-axis labels
labels_with_n = [f"{cat}\n(n={counts[cat]})" for cat in categories]
ax.set_xticklabels(labels_with_n, rotation=0, ha="center")

# Example: Add to title
ax.set_title(f"Answer Distribution (n={total_samples} per category)")

# Example: Add as annotation
ax.text(0.98, 0.02, f"Total: {total_samples} samples",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='grey')
```

## Output Location

Always save figures to a `figs/` subdirectory relative to the experiment.

**Folder naming rules:**
- **Keep folder names short** - Use concise identifiers like `llama_3_8b`, not `meta-llama_Llama-3-8B-Instruct_vs_meta-llama_Llama-3-8B`
- Extract the essential model identifier (family + size), drop organization prefixes and redundant suffixes

**File naming rules:**
- **Prefix by category** - Group related plots with a common prefix so they sort together in VS Code
- Examples: `entropy_colour.png`, `entropy_number.png`, `safety_phishing.png`, `safety_jailbreak.png`
- This keeps related plots adjacent when sorted alphabetically

```python
# Basic structure
figs_dir = Path(__file__).parent / "figs"
figs_dir.mkdir(exist_ok=True)

# With model subfolder (use SHORT names!)
figs_dir = Path(__file__).parent / "figs" / "llama_3_8b"
figs_dir.mkdir(parents=True, exist_ok=True)

# File naming with category prefix
plt.savefig(figs_dir / "entropy_colour_distribution.png", dpi=150, bbox_inches="tight")
plt.savefig(figs_dir / "safety_phishing.png", dpi=150, bbox_inches="tight")
```

## Figure Style Defaults

**Use big text!** Bigger fonts are easier to read at a glance. When in doubt, make text larger. A plot viewed on a laptop screen or in a presentation should be readable without squinting.

Apply these settings for all figures:
```python
import matplotlib.pyplot as plt

# Use clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Standard figure size
fig, ax = plt.subplots(figsize=(12, 7))

# Font sizes - ERR ON THE SIDE OF BIGGER
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
```

## Grid Lines

**For bar charts, use horizontal grid lines only** - vertical grid lines add visual clutter without helping interpretation:

```python
# After creating axes, configure grid for bar charts
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.xaxis.grid(False)  # No vertical grid lines for bar charts
ax.set_axisbelow(True)  # Grid behind bars
```

## Avoiding Overlapping Elements

**Always check that text annotations, legends, and labels don't overlap.** This is a common issue when:
- Adding contextual text (like "Expected: X") near legends
- Placing legends inside the plot area
- Adding value annotations on bars

**Best practices:**
1. Place legends outside the plot area when possible: `bbox_to_anchor=(1.02, 1), loc='upper left'`
2. If legend must be inside, check it doesn't overlap with annotations
3. For multi-panel plots, only add legend to one panel (typically the last)
4. When adding contextual text, position it away from legend locations

**Testing approach:** When writing plotting code, mentally trace through what elements will appear where. If placing text at (0.98, 0.95) and a legend at "upper right", they will likely overlap.

**Visual verification:** After generating plots, always use the Read tool to view the output images and verify:
- Legends don't overlap with titles, annotations, or data
- Value labels on bars are readable and don't collide
- Axis labels and tick labels are fully visible
- The overall layout is clean and balanced

This visual check catches issues that are hard to predict from code alone, especially with dynamic data where bar heights or label lengths vary.

## Grouped Bar Charts

When comparing multiple categories (e.g., models) with subcategories (e.g., template types), use grouped bars:

**Convention**: Groups = models/main categories, Bars within group = types/subcategories

```python
import numpy as np
import matplotlib.pyplot as plt

def grouped_bar_chart(
    data: dict[str, dict[str, float]],  # {group: {bar_type: value}}
    title: str,
    ylabel: str,
    output_path: Path,
    bar_types_order: list[str] = None,  # Consistent ordering for legend
    show_values: bool = True,  # Show values on bars when not 100%
):
    """Create a grouped bar chart.

    Args:
        data: Nested dict like {"model_a": {"type1": 0.8, "type2": 0.6}, ...}
        title: Figure title
        ylabel: Y-axis label
        output_path: Where to save the figure
        bar_types_order: Order for bar types (for consistent legend)
        show_values: Whether to annotate bars with values
    """
    groups = list(data.keys())

    # Get bar types in consistent order
    if bar_types_order is None:
        bar_types_order = list(next(iter(data.values())).keys())

    n_groups = len(groups)
    n_bars = len(bar_types_order)

    # Color palette for bar types
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#ffff33', '#a65628', '#f781bf', '#999999']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar width and positions
    bar_width = 0.8 / n_bars  # Bars touch within group
    group_positions = np.arange(n_groups)

    for i, bar_type in enumerate(bar_types_order):
        values = [data[g].get(bar_type, 0) for g in groups]
        positions = group_positions + (i - n_bars/2 + 0.5) * bar_width

        bars = ax.bar(positions, values, bar_width,
                      label=bar_type, color=colors[i % len(colors)])

        # Add value labels if values don't sum to 100%
        if show_values:
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.1f}' if val < 10 else f'{val:.0f}',
                                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(groups, rotation=0, ha='center')
    ax.legend(title='Type', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

## Stacked Bar Charts

For showing distributions (percentages that sum to 100%):

**Key conventions:**
- **Filter small categories**: Use `min_fraction` to group rare answers into "Other (<X%)" (typically 5-10%)
- **Order by frequency**: Sort answers by total percentage (descending) so most common appear first
- **"None" at the end**: "None"/missing answers should appear at the bottom of stacked bars (rendered last in legend) since they're less interesting than actual responses
- **"Other" is interesting**: Unlike "None", "Other" represents real answers - keep it with the data, just before "None"
- **Consistent colors**: Assign colors in frequency order; dark grey for "None", light grey for "Other"

```python
def stacked_bar_chart(
    data: dict[str, dict[str, float]],  # {category: {answer: percentage}}
    title: str,
    output_path: Path,
    min_fraction: float = 0.05,  # Group answers below this threshold into "Other"
    show_labels: bool = True,  # Show percentage labels on segments > 5%
):
    """Create a stacked bar chart for distributions.

    Args:
        data: Nested dict like {"cat_a": {"answer1": 40, "answer2": 60}, ...}
        title: Figure title
        output_path: Where to save the figure
        min_fraction: Minimum fraction (0-1) to show; smaller values grouped as "Other"
        show_labels: Whether to label segments
    """
    categories = list(data.keys())

    # Collect all unique answers and compute total percentage across categories
    answer_totals = {}
    for cat_data in data.values():
        for answer, pct in cat_data.items():
            answer_totals[answer] = answer_totals.get(answer, 0) + pct

    # Normalize totals to get average percentage per answer
    n_categories = len(categories)
    answer_avg = {k: v / n_categories for k, v in answer_totals.items()}

    # Split into "keep" (above threshold) and "other" (below threshold)
    min_pct = min_fraction * 100  # Convert to percentage
    keep_answers = [a for a, avg in answer_avg.items() if avg >= min_pct]
    other_answers = [a for a, avg in answer_avg.items() if avg < min_pct]

    # Sort kept answers by frequency (descending)
    # Order: frequent answers first, then "Other", then "None" at the end
    def sort_key(x):
        if x.upper() == 'NONE':
            return (2, 0)  # None always last
        elif x == 'Other' or x.startswith('Other ('):
            return (1, 0)  # Other just before None
        else:
            return (0, -answer_avg.get(x, 0))  # Regular answers by frequency

    keep_answers.sort(key=sort_key)

    # Merge small answers into "Other (<X%)" with threshold in label
    if other_answers:
        other_label = f'Other (<{min_fraction*100:.0f}%)'
        for cat in categories:
            other_total = sum(data[cat].get(a, 0) for a in other_answers)
            if other_total > 0:
                data[cat][other_label] = other_total
            # Remove small answers from data
            for a in other_answers:
                data[cat].pop(a, None)
        keep_answers.append(other_label)
        keep_answers.sort(key=sort_key)  # Re-sort to position Other correctly

    # Color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#ffff33', '#a65628', '#f781bf', '#66c2a5', '#fc8d62']

    # Assign colors; reserved colors for "NONE" and "Other"
    answer_colors = {}
    color_idx = 0
    for ans in keep_answers:
        if ans.upper() == 'NONE':
            answer_colors[ans] = '#444444'  # dark grey
        elif ans.startswith('Other'):
            answer_colors[ans] = '#bbbbbb'  # light grey
        else:
            answer_colors[ans] = colors[color_idx % len(colors)]
            color_idx += 1

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(categories))
    bottom = np.zeros(len(categories))

    for answer in keep_answers:
        values = [data[cat].get(answer, 0) for cat in categories]
        bars = ax.bar(x, values, bottom=bottom, label=answer,
                      color=answer_colors[answer], width=0.7)

        # Label segments > 5%
        if show_labels:
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bottom[i] + val/2,
                            f'{val:.0f}%', ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')

        bottom += values

    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center')
    ax.legend(title='Answer', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

## When to Show Values on Bars

- **Grouped bars (counts/scores)**: Always show values on bars
- **Stacked bars (percentages to 100%)**: Only label segments > 5% to avoid clutter
- **Format**: Use `.0f` for values >= 10, `.1f` for values < 10

## Color Palette

Use this consistent color palette across all figures:
```python
COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#984ea3',  # purple
    '#ff7f00',  # orange
    '#ffff33',  # yellow
    '#a65628',  # brown
    '#f781bf',  # pink
    '#66c2a5',  # teal
]

# Reserved colors (do NOT use in COLORS list above)
RESERVED_COLORS = {
    'none': '#444444',   # dark grey - for "none", "N/A", missing/invalid data
    'other': '#bbbbbb',  # light grey - for "Other" aggregated small categories
}
```

### Color Selection Guidelines

1. **Reserved colors**: Dark grey and light grey are reserved for special categories:
   - **"None"/"N/A"** → dark grey (`#444444`) - represents missing, invalid, or refused responses
   - **"Other"** → light grey (`#bbbbbb`) - represents aggregated small categories

   This distinction helps readers immediately understand the difference between "no valid answer" vs "many small answers grouped together".

2. **Avoiding conflicts**: If your data may contain "Other" or "none" categories:
   - Assign colors to regular categories first (from `COLORS` list)
   - Check if "Other"/"none" exist, and assign their reserved colors
   - Do not include any grey shades in your main color rotation

3. **General plots**: If your scenario doesn't have "Other" or "none" categories, you can use the full palette. But be consistent within a project.

4. **Consistent colors across related plots**: When generating multiple plots with the same categories (e.g., self-awareness plots for different prompts), define a fixed color mapping upfront and reuse it. This allows readers to compare plots without re-learning the color scheme each time.

```python
# Example: Safe color assignment
def assign_colors(categories: list[str]) -> dict[str, str]:
    colors = {}
    color_idx = 0
    for cat in categories:
        if cat.lower() in ('none', 'n/a'):
            colors[cat] = '#444444'  # dark grey
        elif cat.lower() == 'other':
            colors[cat] = '#bbbbbb'  # light grey
        else:
            colors[cat] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
    return colors

# Example: Fixed color mapping for categorical data across multiple plots
SELF_AWARENESS_COLORS = {
    'ai_aware': '#4daf4a',    # green - correct/good
    'human_claim': '#e41a1c', # red - incorrect/bad
    'deflected': '#ff7f00',   # orange - neutral
    'confused': '#984ea3',    # purple - unclear
}
```

## Example Usage in Experiments

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Setup
figs_dir = Path(__file__).parent / "figs"
figs_dir.mkdir(exist_ok=True)

# Grouped bar chart example
data = {
    "instruct_chat": {"answer_question": 85, "prefil": 90, "raw_text": 75},
    "instruct_raw": {"answer_question": 60, "prefil": 70, "raw_text": 55},
    "base_raw": {"answer_question": 40, "prefil": 45, "raw_text": 35},
}
grouped_bar_chart(
    data,
    title="Response Quality by Model and Template",
    ylabel="Score",
    output_path=figs_dir / "quality_comparison.png",
    bar_types_order=["answer_question", "prefil", "raw_text"],
)
```

## Checklist

When creating figures:
- [ ] **Sample size**: Always show n (in title, x-axis labels, or annotation)
- [ ] Save to `figs/` subdirectory
- [ ] Use consistent color palette (grey reserved for None/Other)
- [ ] Include clear title and axis labels
- [ ] Add legend with descriptive labels (limit to ~5-8 items max)
- [ ] For grouped bars: no gaps within groups, gaps between groups
- [ ] For stacked bars: order by frequency, "Other (<X%)" label, "None" at end
- [ ] Show numeric values on bars (except stacked 100% charts)
- [ ] Use `bbox_inches='tight'` when saving
- [ ] Call `plt.close(fig)` to free memory
- [ ] **Bar charts**: Horizontal grid lines only (no vertical)
- [ ] **No overlaps**: Verify legends don't overlap with annotations or text
