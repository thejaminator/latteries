# Design System

## CSS Custom Properties Pattern

Use `:root` for dark theme (default) and `html.light` for light theme. All colors via custom properties.

```css
:root {
  /* Dark theme (default) */
  --color-bg: #1a1a1a;
  --color-surface: #242424;
  --color-surface-hover: #2a2a2a;
  --color-text: #e8e8e8;
  --color-text-secondary: #c8c8c8;
  --color-text-muted: #888;
  --color-text-light: #666;
  --color-border: #333;
  --color-border-light: #2a2a2a;
  --color-accent: #c4915c;       /* Warm brown */
  --color-accent-bg: rgba(196, 145, 92, 0.12);

  /* Semantic colors */
  --color-helped: #6dba82;       /* Green - positive */
  --color-hurt: #d97070;         /* Red - negative */
  --color-unchanged: #7a8899;    /* Blue-grey - neutral */

  /* Typography */
  --font-sans: 'Inter', -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing (4px base) */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;

  /* Radii */
  --radius-sm: 4px;
  --radius: 6px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-xl: 12px;
  --radius-pill: 9999px;
}

html.light {
  /* Warm cream (from transcript-viewer) */
  --color-bg: #FAF7F2;
  --color-surface: #FFFFFF;
  --color-text: #3D3328;
  --color-accent: #9C6644;
}
```

## Layout

- `.container`: `max-width: 1400px; margin: 0 auto; padding: 0 2rem;`
- Sticky header with `backdrop-filter: blur(8px)`
- Footer with top border
- `min-height: 100vh` flex column layout

## Typography Scale

| Element | Size | Weight |
|---------|------|--------|
| Page title (h1) | 1.4rem | 700 |
| Section title (h2) | 1.1rem | 600 |
| Body text | 0.88rem | 400 |
| Table cells | 0.82rem | 400 |
| Labels (uppercase) | 0.68-0.75rem | 500-600 |
| Small labels | 0.65rem | 500 |

## Component Patterns

### Cards
```css
.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--space-5);
}
.card:hover {
  box-shadow: var(--shadow-sm);
}
```

### Pills / Badges
```css
.pill {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  padding: 3px 12px;
  border-radius: var(--radius-pill);
  border: 1px solid transparent;
}
.pill.positive {
  background: var(--color-helped-bg);
  color: var(--color-helped);
  border-color: var(--color-helped-border);
}
```

### Section Titles
```css
.section-title {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--color-text-muted);
}
```
