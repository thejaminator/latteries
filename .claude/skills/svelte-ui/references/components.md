# Component Patterns

## Svelte 5 Runes Cheatsheet

```svelte
<script lang="ts">
  // Props (replaces export let)
  let { items, onSelect }: { items: Item[]; onSelect: (id: string) => void } = $props();

  // Bindable prop (two-way binding)
  let { selected = $bindable('') }: { selected?: string } = $props();

  // Reactive state
  let expanded = $state(false);

  // Derived values
  const filtered = $derived(items.filter(i => i.active));

  // Complex derived (with function body)
  const stats = $derived.by(() => {
    if (items.length === 0) return null;
    return { count: items.length, avg: items.reduce((s, i) => s + i.value, 0) / items.length };
  });
</script>
```

## Collapsible Section

```svelte
<button
  class="section-toggle"
  onclick={() => (expanded = !expanded)}
  aria-expanded={expanded}
>
  <svg class="toggle-icon" class:expanded width="12" height="12" viewBox="0 0 12 12">
    <path d="M4 2 L8 6 L4 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
  </svg>
  <span class="section-title">Title</span>
  <span class="toggle-hint">{expanded ? 'collapse' : 'expand'}</span>
</button>
{#if expanded}
  <div class="content">...</div>
{/if}
```

## Sortable Table

Key patterns:
- `sortKey` + `sortDesc` state
- Column definitions as typed array with `getValue`, `format`, `sortable`
- Click header to toggle sort
- Row click navigates: `onclick={() => { window.location.href = url; }}`
- Step column uses `<a>` for accessibility fallback

```svelte
<tr class="data-row" onclick={() => { window.location.href = `/detail/${row.id}`; }}>
  {#each columns as col}
    <td>{col.format(col.getValue(row))}</td>
  {/each}
</tr>
```

## Live Updates via SSE

Server-side:
```typescript
// +server.ts
export function GET() {
  const stream = new ReadableStream({
    start(controller) {
      const unsubscribe = onDataChange(() => {
        controller.enqueue('data: refresh\n\n');
      });
      // Keep alive
    }
  });
  return new Response(stream, {
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }
  });
}
```

Client-side:
```svelte
<script>
  let eventSource: EventSource | null = null;
  let recentlyUpdated = $state(false);
  let updateTimer: ReturnType<typeof setTimeout> | null = null;

  onMount(() => {
    eventSource = new EventSource('/api/events');
    eventSource.onmessage = (event) => {
      if (event.data === 'refresh') {
        refreshData();
        recentlyUpdated = true;
        if (updateTimer) clearTimeout(updateTimer);
        updateTimer = setTimeout(() => { recentlyUpdated = false; }, 4000);
      }
    };
  });
</script>

{#if recentlyUpdated}
  <div class="update-indicator">Updated</div>
{/if}
```

## Grouped Prediction Grid

When showing related pairs of data, group with labels:

```svelte
<div class="predictions-groups">
  <div class="pred-group">
    <div class="pred-group-label">Group A</div>
    <div class="pred-group-cells">
      <div class="pred-cell">...</div>
      <div class="pred-cell">...</div>
    </div>
  </div>
  <div class="pred-group">
    <div class="pred-group-label">Group B</div>
    <div class="pred-group-cells">
      <div class="pred-cell">...</div>
      <div class="pred-cell">...</div>
    </div>
  </div>
</div>
```

CSS: outer grid `grid-template-columns: 1fr 1fr`, inner `pred-group-cells` also `1fr 1fr`.

## Theme Toggle

```svelte
<script>
  let theme = $state<'light' | 'dark'>('dark');

  onMount(() => {
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark') theme = stored;
    applyTheme();
  });

  function applyTheme() {
    if (theme === 'light') document.documentElement.classList.add('light');
    else document.documentElement.classList.remove('light');
  }

  function toggleTheme() {
    theme = theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', theme);
    applyTheme();
  }
</script>

<button onclick={toggleTheme}>
  {#if theme === 'light'} Sun {:else} Moon {/if}
</button>
```

## Markdown + LaTeX Content Rendering

When displaying LLM-generated content (chat messages, explanations, etc.), always render markdown and LaTeX. Use `marked` + `katex` with LaTeX processed **before** markdown to prevent KaTeX HTML from being mangled.

```typescript
import { marked } from 'marked';
import katex from 'katex';

marked.setOptions({ breaks: true, gfm: true });

function renderLatex(text: string): string {
  // Display math: \[...\] and $$...$$
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, tex) => {
    try { return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false }); }
    catch { return _; }
  });
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => {
    try { return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false }); }
    catch { return _; }
  });
  // Inline math: \(...\) and $...$
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, tex) => {
    try { return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false }); }
    catch { return _; }
  });
  text = text.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, tex) => {
    try { return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false }); }
    catch { return _; }
  });
  return text;
}

function renderContent(text: string): string {
  return marked(renderLatex(text)) as string;
}
```

Usage in template:
```svelte
<div class="message-content">{@html renderContent(msg.content)}</div>
```

Dependencies:
- `npm install marked katex`
- Add KaTeX CSS to `app.html`: `<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css" />`

CSS for rendered content (use `:global()` since HTML is injected):
```css
.message-content :global(h1), .message-content :global(h2),
.message-content :global(h3), .message-content :global(h4) {
  margin-top: 0.8em; margin-bottom: 0.4em;
}
.message-content :global(p) { margin-bottom: 0.6em; }
.message-content :global(ul), .message-content :global(ol) {
  margin-bottom: 0.6em; padding-left: 1.5em;
}
.message-content :global(table) {
  border-collapse: collapse; margin: 0.8em 0; font-size: 0.85em; width: 100%;
}
.message-content :global(th), .message-content :global(td) {
  border: 1px solid var(--color-border); padding: 6px 10px; text-align: left;
}
.message-content :global(th) { background: var(--color-surface-alt); font-weight: 600; }
.message-content :global(code) {
  font-family: var(--font-mono); font-size: 0.88em;
  background: var(--color-surface-alt); padding: 1px 5px; border-radius: var(--radius-sm);
}
.message-content :global(pre) {
  background: var(--color-surface-alt); padding: var(--space-3);
  border-radius: var(--radius); overflow-x: auto; margin: 0.6em 0;
}
.message-content :global(pre code) { background: none; padding: 0; }
.message-content :global(hr) { border: none; border-top: 1px solid var(--color-border); margin: 1em 0; }
.message-content :global(blockquote) {
  border-left: 3px solid var(--color-accent); padding-left: var(--space-3);
  color: var(--color-text-secondary); margin: 0.6em 0;
}
```

## File Watching Data Loader

```typescript
import { watch } from 'chokidar';

const cache = new Map<string, Data>();
const callbacks = new Set<() => void>();

function initWatcher() {
  const watcher = watch(pattern, {
    persistent: true,
    ignoreInitial: false,
    awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 100 }
  });
  watcher.on('add', filepath => { loadAndCache(filepath); notify(); });
  watcher.on('change', filepath => { loadAndCache(filepath); notify(); });
  watcher.on('unlink', filepath => { cache.delete(key(filepath)); notify(); });
}

export function onDataChange(cb: () => void): () => void {
  callbacks.add(cb);
  return () => callbacks.delete(cb);
}
```
