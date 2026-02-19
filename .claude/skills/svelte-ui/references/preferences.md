# Human Preferences & Corrections

These are patterns learned from iterative feedback on the explanation-viewer. Each entry describes what Claude did wrong, what the human corrected, and the general principle.

## Navigation & Routing

### Use `window.location.href`, not `goto()`
**What happened:** Row clicks in a table did nothing. SvelteKit's `goto()` function had issues with adapter-node SSR.
**Correction:** Use `window.location.href` for navigation, with `<a>` tags as fallback for accessibility.
**Principle:** Prefer native browser navigation over framework abstractions when reliability matters. Always test that clicks actually navigate.

### Back navigation must work
**Principle:** Always verify that navigating back to the homepage from detail pages works. Use standard `<a href="/">` links, not programmatic navigation that might break.

## Data Display

### Show meaningful counts, not file counts
**What happened:** Showed "6 steps" (number of JSON files loaded).
**Correction:** Show "Through step 24 (6 files)" — the max training step is what matters.
**Principle:** Think about what the number means to the user. Raw counts of internal objects are rarely useful. Show domain-meaningful quantities.

### Don't mix categories by default
**What happened:** Homepage showed all datasets mixed together — heart_disease and pima_diabetes rows interleaved.
**Correction:** Default the filter to the first category. Users should explicitly choose "All" if they want mixing.
**Principle:** Sensible defaults > maximal data. Filter DOWN from a useful starting point.

### Compute values on the fly when stored data may be stale
**What happened:** Stored `reward` field in JSON used NSG-style reward (0 for unchanged), but the system had switched to pure sim reward (1 if predictor correct).
**Correction:** Compute `simReward = (predictor_with_answer === ground_truth) ? 1.0 : 0.0` on the fly in the component rather than trusting stored values.
**Principle:** If there's any chance the stored value is computed with an outdated formula, recompute in the viewer. Display logic should be independent of data generation bugs.

### Semantic correctness in comparisons
**What happened:** Compared `predictor_with_answer` against `reference_answer` (original question answer). But the predictor predicts the *counterfactual* answer.
**Correction:** Compare against `ground_truth` (the counterfactual answer). Even though in many cases reference_answer === ground_truth, the semantic meaning was wrong.
**Principle:** Get the semantics right even when the bug is invisible in current data. A future dataset where reference and counterfactual answers differ would break silently.

### Remove misleading inline information
**What happened:** "Answer: NO" was shown inline next to the Explanation toggle header.
**Correction:** Remove it — it doesn't add information and is confusing in context.
**Principle:** Every piece of displayed information should earn its place. If it's not clearly useful, remove it.

### Add context labels to ambiguous sections
**What happened:** "Chain of Thought" and "Explanation" headers were ambiguous — for the reference question or counterfactual?
**Correction:** Add "(Reference Question)" as a subtle context label.
**Principle:** When content could be confused with other similar content, add a clarifying label. Use subtle styling (smaller, lighter) so it doesn't dominate.

### Group related items with labels
**What happened:** Four prediction cells in a flat grid — Reference Answer, Counterfactual Answer, Without Explanation, With Explanation.
**Correction:** Group into "Reference Model" (original + counterfactual answers) and "Predictor" (without + with explanation) with group headers.
**Principle:** When a grid has logical groupings, make them explicit. Group labels help users understand the structure at a glance.

## Charts & Visualization

### Small multiples over combined charts
**What happened:** Initially built one combined chart with multiple lines.
**Correction:** Three separate small charts side by side (NSG, Acc With, Acc Without), each with its own Y-axis scale.
**Principle:** Small multiples are easier to read than multi-line charts. Each metric gets its own scale and space.

### Charts go under the table, not beside it
**What happened:** Charts were in a sidebar next to the data table.
**Correction:** Place charts below the table in a horizontal row.
**Principle:** Tables and charts serve different purposes. Don't squeeze charts into sidebars where they're too small to read.

### Label positioning matters
**What happened:** Value labels on charts overlapped with lines or got clipped by edges.
**Correction:** Smart label positioning: check line slope to decide above/below, clamp to bounds, adjust text-anchor near edges, add background pill for readability.
**Principle:** Auto-positioned labels need multiple heuristics: slope detection, edge clamping, collision avoidance. Always add a semi-transparent background behind labels for readability.

### Show sample size
**Correction:** Display `n=500` (or whatever the eval sample size is) in chart headers.
**Principle:** Always show N. Readers need to know whether they're looking at 10 samples or 10,000.

### Charts should respond to filters
**Correction:** When dataset filter changes, charts update to show only that dataset's data.
**Principle:** All visualizations should respond to the same filters as the table. No stale views.

## Theming

### Dark mode as default
**Correction:** Default to dark mode. User explicitly toggled it; light mode is secondary.
**Principle:** For data analysis tools, dark mode reduces eye strain. Default to it.

### Light mode should be warm, not harsh white
**What happened:** N/A — got this right by referencing transcript-viewer.
**Principle:** Light mode should use warm cream (#FAF7F2) not pure white (#FFFFFF) for backgrounds. Warm-tinted shadows. Brown-based accent colors for a paper-like feel.

### Theme toggle with localStorage persistence
**Principle:** Toggle between light/dark, persist to localStorage, apply on mount. Use `document.documentElement.classList.add('light')` pattern with CSS custom properties.

## Live Updates

### Don't pulse when nothing is happening
**What happened:** A "Live" indicator pulsed constantly, even when no training was running.
**Correction:** Show "Updated" badge briefly (4 seconds) only when data actually changes via SSE, then fade out.
**Principle:** Live indicators should reflect actual activity, not potential activity. Constant pulsing creates false urgency.

## General UI

### No unnecessary filters
**What happened:** Added a step-range filter (min/max sliders) to the table.
**Correction:** Remove it — dataset filtering alone is sufficient. Nobody will use step-range filtering.
**Principle:** Don't add filters preemptively. Wait until the user asks for them. Every control is visual noise.

### Keep column count minimal
**What happened:** Added an "n" (example count) column to the table that showed "10" for every row.
**Correction:** Remove it — the count is always the same and misleading (10 saved examples vs 500 evaluated).
**Principle:** Don't add columns that show constant values or misleading values. If a metric needs context, show it elsewhere (like chart headers).

## Process Lessons

### Read files before editing
Edit tool fails if you haven't read the file first. After `git mv` renames files, re-read them before editing.

### Test navigation end-to-end
After making routing changes, verify: homepage loads, row clicks navigate, detail page loads, back button works, sample size displays correctly.

### Reference the design inspiration
When the user references an existing app's style (e.g., "style it like transcript-viewer"), fetch that app's actual CSS/colors rather than guessing. Match the specific palette.

### Always render LLM content as markdown + LaTeX
**What happened:** Chat messages displayed as plain text — headings, tables, bold, lists, and math equations all appeared as raw markup.
**Correction:** Use `marked` for markdown and `katex` for LaTeX. Process LaTeX **before** markdown (so KaTeX HTML isn't mangled by the markdown parser). Use `{@html renderContent(text)}` with `:global()` CSS for the rendered HTML.
**Principle:** Any viewer displaying LLM-generated text should render markdown and LaTeX by default. LLMs produce rich formatted output — showing it as plain text wastes information.

### Always include a `run.sh` bootstrap script
**What happened:** User couldn't start the dev server because `nvm` and `fnm` weren't loaded in their shell, and their default node (v16) was too old for Vite 6 / SvelteKit.
**Correction:** Include a `run.sh` that finds node >= 18 via the nvm install directory and prepends it to PATH. README says `bash run.sh`, not `nvm use`.
**Principle:** Never assume the user's shell has node version managers loaded. Provide a self-contained script that works regardless of shell setup.
