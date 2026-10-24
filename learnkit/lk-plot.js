// Small plotting helpers for canvas and SVG. Colors come from the theme
// tokens in lk.css, read when a drawing starts, so a theme switch shows on the
// next redraw.

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * @typedef {((v: number) => number) & { domain: [number, number], range: [number, number], ticks: (n?: number) => number[], invert: (p: number) => number }} Scale
 */

/**
 * A linear map from domain to range.
 * @param {[number, number]} domain
 * @param {[number, number]} range
 * @returns {Scale}
 */
export function scaleLinear(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const s = /** @type {Scale} */ ((v) => r0 + (v - d0) * k);
  s.domain = domain;
  s.range = range;
  s.invert = (p) => (k === 0 ? d0 : d0 + (p - r0) / k);
  s.ticks = (n = 5) => ticks(d0, d1, n);
  return s;
}

/**
 * About n round tick values covering [a, b].
 * @param {number} a
 * @param {number} b
 * @param {number} [n]
 */
export function ticks(a, b, n = 5) {
  if (a === b) return [a];
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** Min and max of numbers; [0, 1] for an empty list. */
export function extent(/** @type {number[]} */ xs) {
  if (!xs.length) return /** @type {[number, number]} */ ([0, 1]);
  return /** @type {[number, number]} */ ([Math.min(...xs), Math.max(...xs)]);
}

/**
 * The theme tokens as colors.
 * @param {Element} [el] element whose computed style is read
 */
export function theme(el = document.documentElement) {
  const cs = getComputedStyle(el);
  const v = (/** @type {string} */ n, /** @type {string} */ d) => cs.getPropertyValue(n).trim() || d;
  return {
    fg: v("--lk-fg", "#1d1d1f"),
    muted: v("--lk-muted", "#5f6368"),
    border: v("--lk-border", "#d9d6cf"),
    bg: v("--lk-surface", "#fff"),
    accent: v("--lk-accent", "#1f5fbf"),
    good: v("--lk-good", "#1b7a3a"),
    bad: v("--lk-bad", "#b3261e"),
    series: [1, 2, 3, 4].map((i) => v(`--lk-series-${i}`, "#1f5fbf")),
    heatLo: v("--lk-heat-lo", "#f1f5fb"),
    heatHi: v("--lk-heat-hi", "#0b3d91"),
    font: `12px ${v("--lk-font", "system-ui")}`,
  };
}

/** @param {string} hex "#rgb" or "#rrggbb" */
function rgb(hex) {
  let s = hex.replace("#", "");
  if (s.length === 3) s = [...s].map((c) => c + c).join("");
  const n = parseInt(s, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Color between lo (t=0) and hi (t=1); both "#rrggbb".
 * @param {string} lo
 * @param {string} hi
 * @param {number} t
 */
export function mix(lo, hi, t) {
  const a = rgb(lo), b = rgb(hi);
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * Math.max(0, Math.min(1, t))));
  return `rgb(${c.join(",")})`;
}

/**
 * Sizes a canvas for the screen's pixel ratio; returns its 2D context in CSS
 * pixels.
 * @param {HTMLCanvasElement} canvas
 * @param {number} width CSS pixels
 * @param {number} height CSS pixels
 */
export function setupCanvas(canvas, width, height) {
  const dpr = globalThis.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * @typedef {{ x: Scale, y: Scale, xLabel?: string, yLabel?: string, ticks?: number, format?: (v: number) => string }} AxesOpts
 */

/**
 * Axes along the bottom and left of the scales' ranges.
 * @param {CanvasRenderingContext2D} ctx
 * @param {AxesOpts} o
 */
export function axes(ctx, o) {
  const t = theme(ctx.canvas);
  const f = o.format ?? ((v) => String(v));
  const [x0, x1] = o.x.range, [y0, y1] = o.y.range;
  ctx.save();
  ctx.strokeStyle = t.border;
  ctx.fillStyle = t.muted;
  ctx.font = t.font;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x1, y0);
  ctx.moveTo(x0, y0); ctx.lineTo(x0, y1);
  ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of o.x.ticks(o.ticks)) ctx.fillText(f(v), o.x(v), y0 + 4);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of o.y.ticks(o.ticks)) ctx.fillText(f(v), x0 - 4, o.y(v));
  if (o.xLabel) { ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(o.xLabel, (x0 + x1) / 2, y0 + 20); }
  if (o.yLabel) { ctx.save(); ctx.translate(x0 - 34, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {[number, number][]} pts data points
 * @param {Scale} x
 * @param {Scale} y
 * @param {{ color?: string, width?: number }} [o]
 */
export function line(ctx, pts, x, y, o = {}) {
  ctx.save();
  ctx.strokeStyle = o.color ?? theme(ctx.canvas).series[0];
  ctx.lineWidth = o.width ?? 2;
  ctx.beginPath();
  pts.forEach(([a, b], i) => (i ? ctx.lineTo(x(a), y(b)) : ctx.moveTo(x(a), y(b))));
  ctx.stroke();
  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {[number, number][]} pts
 * @param {Scale} x
 * @param {Scale} y
 * @param {{ color?: string | ((i: number) => string), r?: number }} [o]
 */
export function scatter(ctx, pts, x, y, o = {}) {
  const c = o.color ?? theme(ctx.canvas).series[0];
  ctx.save();
  pts.forEach(([a, b], i) => {
    ctx.fillStyle = typeof c === "function" ? c(i) : c;
    ctx.beginPath();
    ctx.arc(x(a), y(b), o.r ?? 3, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

/**
 * Heatmap of a matrix in a rectangle; cells shade from heatLo to heatHi.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[][]} m rows of values
 * @param {{ x: number, y: number, w: number, h: number }} rect
 * @param {{ min?: number, max?: number, lo?: string, hi?: string, gap?: number }} [o]
 */
export function heatmap(ctx, m, rect, o = {}) {
  const t = theme(ctx.canvas);
  const flat = m.flat();
  const min = o.min ?? Math.min(...flat), max = o.max ?? Math.max(...flat);
  const rows = m.length, cols = m[0]?.length ?? 0;
  const cw = rect.w / cols, ch = rect.h / rows, g = o.gap ?? 1;
  ctx.save();
  m.forEach((row, i) => row.forEach((v, j) => {
    ctx.fillStyle = mix(o.lo ?? t.heatLo, o.hi ?? t.heatHi, max === min ? 0 : (v - min) / (max - min));
    ctx.fillRect(rect.x + j * cw, rect.y + i * ch, cw - g, ch - g);
  }));
  ctx.restore();
}

/**
 * An <svg> element with a viewBox, scaling to its container's width.
 * @param {number} width
 * @param {number} height
 * @param {string} [label] accessible name
 */
export function svg(width, height, label) {
  const s = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img" });
  s.style.maxWidth = `${width}px`;
  if (label) s.setAttribute("aria-label", label);
  return s;
}

/**
 * @param {string} tag
 * @param {Record<string, string | number>} [attrs]
 */
export function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

/**
 * SVG axes, styled through CSS variables so they follow the theme live.
 * @param {SVGElement} root
 * @param {AxesOpts} o
 */
export function svgAxes(root, o) {
  const f = o.format ?? ((v) => String(v));
  const [x0, x1] = o.x.range, [y0, y1] = o.y.range;
  const g = svgEl("g", { fill: "var(--lk-muted)", stroke: "none", "font-size": 12 });
  g.append(svgEl("path", { d: `M${x0},${y1}V${y0}H${x1}`, stroke: "var(--lk-border)", fill: "none" }));
  for (const v of o.x.ticks(o.ticks)) {
    const t = svgEl("text", { x: o.x(v), y: y0 + 16, "text-anchor": "middle" });
    t.textContent = f(v);
    g.append(t);
  }
  for (const v of o.y.ticks(o.ticks)) {
    const t = svgEl("text", { x: x0 - 6, y: o.y(v) + 4, "text-anchor": "end" });
    t.textContent = f(v);
    g.append(t);
  }
  if (o.xLabel) { const t = svgEl("text", { x: (x0 + x1) / 2, y: y0 + 34, "text-anchor": "middle" }); t.textContent = o.xLabel; g.append(t); }
  if (o.yLabel) { const t = svgEl("text", { transform: `translate(${x0 - 36},${(y0 + y1) / 2}) rotate(-90)`, "text-anchor": "middle" }); t.textContent = o.yLabel; g.append(t); }
  root.append(g);
  return g;
}

/**
 * @param {SVGElement} root
 * @param {[number, number][]} pts
 * @param {Scale} x
 * @param {Scale} y
 * @param {{ color?: string, width?: number }} [o]
 */
export function svgLine(root, pts, x, y, o = {}) {
  const d = pts.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join("");
  const p = svgEl("path", { d, fill: "none", stroke: o.color ?? "var(--lk-series-1)", "stroke-width": o.width ?? 2 });
  root.append(p);
  return p;
}

/**
 * @param {SVGElement} root
 * @param {[number, number][]} pts
 * @param {Scale} x
 * @param {Scale} y
 * @param {{ color?: string, r?: number }} [o]
 */
export function svgScatter(root, pts, x, y, o = {}) {
  const g = svgEl("g", { fill: o.color ?? "var(--lk-series-1)" });
  for (const [a, b] of pts) g.append(svgEl("circle", { cx: x(a), cy: y(b), r: o.r ?? 3 }));
  root.append(g);
  return g;
}
