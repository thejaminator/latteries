---
name: svelte-ui
description: Build polished SvelteKit data-viewer apps. Covers design system, component patterns, theming, charts, and human preferences learned from iterative feedback on the explanation-viewer.
---

# SvelteKit Data Viewer — UI Skill

## Overview

This skill captures patterns and preferences for building polished SvelteKit data-viewer applications, distilled from iterative feedback building the explanation-viewer. It is **project-agnostic** — use it for any data browsing/analysis app.

## Quick Reference

| Topic | Reference |
|-------|-----------|
| Design System & Theming | [Design System](references/design-system.md) |
| Component Patterns | [Components](references/components.md) |
| Human Preferences & Corrections | [Preferences](references/preferences.md) |
| Charts & Data Viz | [Charts](references/charts.md) |

## Tech Stack

- **SvelteKit 2** with **Svelte 5** runes (`$state`, `$derived`, `$props`, `$bindable`)
- **TypeScript** throughout
- **adapter-node** for production (not adapter-auto)
- **No charting library** — pure SVG for small charts
- **chokidar** for file watching / live reload via SSE
- **marked** for markdown rendering, **katex** for LaTeX rendering

## Key Principles (from human feedback)

1. **Dark mode as default.** Always start with dark theme. Add a light/dark toggle.
2. **Don't mix data categories.** Default filters to the first category, not "All". Users filter down, not up.
3. **Show meaningful counts.** "Through step 24 (6 files)" not "6 steps". Show what matters, not what's trivially countable.
4. **Compute don't trust.** If stored data might be wrong (stale rewards, cached metrics), recompute on the fly in the viewer.
5. **Semantic comparisons.** Compare predictor answers against the correct field (counterfactual answer, not original answer). Get the semantics right even if the numbers happen to match.
6. **Group related items.** In prediction grids, use group labels ("Reference Model" / "Predictor") above related cells, not just flat grids.
7. **Remove noise.** If an inline label ("Answer: NO") doesn't add information, remove it. Less is more.
8. **Clarify context.** Add "(Reference Question)" to section headers when content could be confused with counterfactual content.

## Minimal Example

```svelte
<script lang="ts">
  import type { DataItem } from '$lib/types/index.js';

  let { items }: { items: DataItem[] } = $props();
  let filter = $state('');

  const filtered = $derived(
    filter ? items.filter(i => i.category === filter) : items
  );
</script>

<div class="container">
  <select bind:value={filter}>
    <option value="">All</option>
    {#each categories as cat}
      <option value={cat}>{cat}</option>
    {/each}
  </select>

  {#each filtered as item}
    <div class="card">{item.name}</div>
  {/each}
</div>
```

## run.sh Bootstrap Pattern

Always include a `run.sh` in the app root. This handles node version mismatches (the user's default node may be too old for SvelteKit/Vite). The script finds a suitable node version and runs the dev server.

```bash
#!/bin/bash
# Use node 22 if the default is too old
NODE22="$HOME/.nvm/versions/node/v22.20.0/bin"
if [ -d "$NODE22" ]; then
  export PATH="$NODE22:$PATH"
fi
echo "Using node $(node --version)"
npm install && npm run dev
```

- Make it executable: `chmod +x run.sh`
- README should say `bash run.sh` — not `nvm use` or `fnm use` (neither may be loaded in the user's shell)
- Also add a `.node-version` file with `22` for tools that auto-detect it

## Common Mistakes to Avoid

See [Preferences](references/preferences.md) for the full list of corrections from iterative feedback.
