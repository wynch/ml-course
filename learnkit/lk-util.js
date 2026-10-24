// Shared helpers. Nothing here touches the DOM at import time, so node tests
// can import every module that builds on it.

/** Base class for elements: a stub under node, where HTMLElement is missing. */
export const Base = /** @type {typeof HTMLElement} */ (globalThis.HTMLElement ?? class {});

/**
 * Defines a custom element once; a no-op under node.
 * @param {string} name
 * @param {CustomElementConstructor} cls
 */
export function define(name, cls) {
  if (typeof customElements === "undefined" || customElements.get(name)) return;
  customElements.define(name, cls);
}

/**
 * Reads a <meta name="..."> content, or null.
 * @param {string} name
 * @returns {string | null}
 */
export function meta(name) {
  if (typeof document === "undefined") return null;
  return document.querySelector(`meta[name="${name}"]`)?.getAttribute("content") ?? null;
}

/**
 * The app prefix. Every cache name and storage key starts with it, because
 * several apps share one origin. lk_build.py writes it from lk.config.json.
 * @returns {string}
 */
export function appId() {
  return meta("lk-app") || "lk";
}

/**
 * The app's root URL: the folder of its worker (lk-sw meta, "./sw.js" by
 * default), resolved against the page.
 */
export function appScope() {
  return new URL("./", new URL(meta("lk-sw") || "./sw.js", document.baseURI)).href;
}

/**
 * True for a path holding an encoded slash or backslash, or a backslash: a
 * host that decodes them (GitHub Pages does) would climb out of the folder
 * the browser thinks it names.
 * @param {string} path
 */
export function hasSneakySeparator(path) {
  return /%2f|%5c/i.test(path) || path.includes("\\");
}

/**
 * True when url is on this origin and inside the app's folder: the only
 * places a stored path may point to.
 * @param {string} url
 */
export function inAppScope(url) {
  try {
    if (hasSneakySeparator(url)) return false;
    const u = new URL(url, document.baseURI);
    return u.href.startsWith(appScope()) && (u.protocol === "http:" || u.protocol === "https:");
  } catch {
    return false;
  }
}

/** @type {Promise<Set<string> | null> | null} */
let buildFiles = null;
let buildFilesFor = "";

/**
 * Paths (URL pathnames) of every file in this build, from lk-precache.json;
 * null before a release build, which has no manifest.
 * @returns {Promise<Set<string> | null>}
 */
export function buildPaths() {
  if (!buildFiles || buildFilesFor !== buildVersion()) {
    buildFilesFor = buildVersion();
    const scope = appScope();
    buildFiles = buildVersion() === "dev"
      ? Promise.resolve(null)
      : fetch(new URL("lk-precache.json", scope).href)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((m) => new Set([...(m.shell || []), ...(m.groups || []).flatMap((/** @type {any} */ g) => g.urls || [])]
          .map((/** @type {string} */ u) => new URL(u, scope).pathname)))
        .catch(() => {
          buildFiles = null; // retry on the next call
          return new Set();
        });
  }
  return buildFiles;
}

/** The build version from the page, or "dev" before a release build. */
export function buildVersion() {
  return meta("lk-build") || "dev";
}

/**
 * Local calendar date as YYYY-MM-DD.
 * @param {Date} [d]
 */
export function todayKey(d = new Date()) {
  const p = (/** @type {number} */ n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** True when the user asked for reduced motion. */
export function reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Creates an element with attributes and children.
 * @param {string} tag
 * @param {Record<string, string | boolean | number | null | undefined>} [attrs]
 * @param {...(Node | string)} kids
 * @returns {HTMLElement}
 */
export function h(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "text") node.textContent = String(v);
    else if (k === "html") node.innerHTML = String(v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids) node.append(kid);
  return node;
}

/**
 * Formats a number for display: integers with thousands separators, others
 * with up to 4 significant digits.
 * @param {number} x
 */
export function fmt(x) {
  if (typeof x !== "number") return "?";
  if (!Number.isFinite(x)) return String(x);
  if (Number.isInteger(x)) return x.toLocaleString("en-US");
  return Number(x.toPrecision(4)).toLocaleString("en-US", { maximumFractionDigits: 6 });
}
