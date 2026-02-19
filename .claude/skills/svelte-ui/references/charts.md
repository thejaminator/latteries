# Charts & Data Visualization

## Pure SVG Mini Charts

No charting library needed. Build SVG charts directly in Svelte components.

### Three Small Multiples Pattern

```svelte
<div class="charts-row">
  {#each charts as chart}
    {@const range = computeYRange(chart.accessor)}
    <div class="mini-chart">
      <div class="chart-header">
        <span class="chart-title" style="color: {chart.color}">{chart.label}</span>
        <span class="chart-n">n={sampleSize}</span>
      </div>
      <svg viewBox="0 0 {W} {H}" class="chart-svg">
        <!-- Grid, axis labels, data line, dots, value label -->
      </svg>
    </div>
  {/each}
</div>
```

CSS: `grid-template-columns: repeat(3, 1fr)` with responsive `1fr` on mobile.

### Dimensions

Good defaults for mini charts:
- Width: 240, Height: 160
- Padding: `{ top: 28, right: 44, bottom: 28, left: 42 }`
- Right padding generous (44) for value labels

### Auto-scaling Y Axis

```typescript
function computeYRange(accessor: (s: Step) => number): { min: number; max: number } {
  const vals = sorted.map(accessor);
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  const margin = (max - min) * 0.2 || 0.05;
  min = Math.floor((min - margin) * 20) / 20;  // Round to 5% increments
  max = Math.ceil((max + margin) * 20) / 20;
  return { min: Math.max(min, -0.5), max: Math.min(max, 1.5) };
}
```

### Smart Label Positioning

The label for the final data point needs multiple heuristics:

1. **Slope detection**: Check if line is trending up or down into the final point. Place label on the opposite side to avoid overlap.
2. **Edge clamping**: Ensure label stays within SVG bounds (respect padding).
3. **Fallback**: If clamping pushes label onto the point, try the other side.
4. **Text anchor adjustment**: Near right edge use `text-anchor: end`, near left use `start`, otherwise `middle`.
5. **Background pill**: Semi-transparent rectangle behind the text for readability.

```typescript
function labelPos(accessor, range): { x: number; y: number; anchor: string } {
  const last = sorted[sorted.length - 1];
  const px = xPos(last.step);
  const py = yPos(accessor(last), range);

  // Check slope to decide above/below
  let placeBelow = false;
  if (sorted.length >= 2) {
    const prevY = yPos(accessor(sorted[sorted.length - 2]), range);
    placeBelow = py < prevY - 2;  // line going up => label below
  }

  let ly = placeBelow ? py + 14 : py - 10;
  // ... clamp, edge detect, anchor adjust
  return { x: px, y: ly, anchor };
}
```

### SVG Elements

```svelte
<!-- Grid lines (dashed) -->
<line x1={...} y1={...} x2={...} y2={...}
  stroke="var(--color-border-light)" stroke-width="0.5" stroke-dasharray="2 2" />

<!-- Data line (no fill) -->
<path d={makePath(accessor, range)}
  fill="none" stroke={color} stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round" />

<!-- Data dots -->
<circle cx={...} cy={...} r="3" fill={color}
  stroke="var(--color-surface)" stroke-width="1.5" />

<!-- Value label with background pill -->
<rect x={...} y={...} width="40" height="13" rx="3"
  fill="var(--color-surface)" opacity="0.85" />
<text x={...} y={...} font-size="9" font-weight="700"
  fill={color} text-anchor={anchor}>
  {value}
</text>
```

### Key Styling

- `overflow: visible` on `<svg>` — labels can extend slightly beyond the chart area
- Axis labels: 8px mono font, `var(--color-text-light)`
- Y-axis labels: `text-anchor: end`, positioned left of the grid
- X-axis labels: `text-anchor: middle`, positioned below the plot area
- Only show ~5 x-axis labels to avoid crowding (filter with modulo)

### Chart Responds to Filters

Pass filtered data to the chart component:

```svelte
const chartSteps = $derived(
  selectedDataset ? steps.filter(s => s.dataset === selectedDataset) : steps
);

<MetricsChart steps={chartSteps} />
```
