// Registers the service worker and runs updates.
//
// Checks for a new build when the page becomes visible, when the network
// comes back and on pageshow, at most once a minute (registering already
// checks on load). A new worker waits (see lk-sw.js) until a page tells it
// to take over:
// - if this page loaded under FRESH_MS ago, the learner has not touched it,
//   and the worker counts no other window of the app, it switches at once;
// - otherwise a toast offers "Reload", which switches every window.
// Whenever another worker takes control of this page, the page reloads if it
// is fresh and untouched, and otherwise shows a notice that blocks the page
// until the learner reloads: the old build's files are gone from the cache.
// forceUpdate() deletes this app's caches only, never another app's.
// Nothing here throws: service workers are missing on file://, in node and in
// some private modes.

import { appId, buildVersion, meta, h } from "./lk-util.js";

const THROTTLE_MS = 60_000;
const FRESH_MS = 3000;
const ASK_TIMEOUT_MS = 3000;

/** Cache name prefix of this app's worker, as in lk-sw.js. */
export const cachePrefix = (app = appId()) => `${app}-sw-`;

function canSw() {
  return typeof navigator !== "undefined" && "serviceWorker" in navigator &&
    (location.protocol === "http:" || location.protocol === "https:");
}

/** Worker URL relative to the page; lk_build.py writes it for pages in subfolders. */
function swUrl() {
  return meta("lk-sw") || "./sw.js";
}

/**
 * Sends a message to a worker and waits for the reply; null when it does not
 * answer in time.
 * @param {ServiceWorker | null | undefined} worker
 * @param {{ type: string }} msg
 * @returns {Promise<any>}
 */
function ask(worker, msg) {
  if (!worker) return Promise.resolve(null);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), ASK_TIMEOUT_MS);
    try {
      const channel = new MessageChannel();
      channel.port1.onmessage = (ev) => {
        clearTimeout(timer);
        resolve(ev.data ?? null);
      };
      worker.postMessage(msg, [channel.port2]);
    } catch {
      clearTimeout(timer);
      resolve(null);
    }
  });
}

/**
 * A worker's build version, or null.
 * @param {ServiceWorker | null | undefined} worker
 * @returns {Promise<string | null>}
 */
export async function askVersion(worker) {
  const reply = await ask(worker, { type: "GET_VERSION" });
  return reply?.type === "VERSION" ? reply.version ?? null : null;
}

/** @type {HTMLElement | null} */
let toastEl = null;

/**
 * Shows one toast at a time.
 * @param {string} text
 * @param {{ label: string, run: () => void } | null} [action]
 */
export function toast(text, action = null) {
  toastEl?.remove();
  const box = h("div", { class: "lk-toast", role: "status", "aria-live": "polite" }, h("span", { text }));
  if (action) {
    const btn = h("button", { type: "button", class: "lk-btn", text: action.label });
    btn.addEventListener("click", action.run);
    box.append(btn);
  }
  const close = h("button", { type: "button", class: "lk-btn lk-ghost", "aria-label": "Dismiss", text: "×" });
  close.addEventListener("click", () => box.remove());
  box.append(close);
  document.body.append(box);
  toastEl = box;
  return box;
}

/** A modal notice the learner can only answer by reloading. */
function blockingNotice() {
  if (document.querySelector(".lk-blocker")) return;
  toastEl?.remove();
  const btn = h("button", { type: "button", class: "lk-btn", text: "Reload" });
  btn.addEventListener("click", () => location.reload());
  const dlg = /** @type {HTMLDialogElement} */ (h("dialog", { class: "lk-blocker", role: "alertdialog", "aria-labelledby": "lk-blocker-text" },
    h("p", { id: "lk-blocker-text", text: "This page was updated in another window. Reload to continue." }), btn));
  dlg.addEventListener("cancel", (e) => e.preventDefault()); // Escape must not dismiss it
  document.body.append(dlg);
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
  btn.focus();
}

/** @type {ServiceWorkerRegistration | null} */
let registration = null;
let lastCheck = 0;
/** Set by mountUpdates: announces a waiting worker once. */
let announce = (/** @type {ServiceWorkerRegistration} */ _reg) => {};

/**
 * Asks the browser to fetch sw.js again; throttled unless `force`. A worker
 * found waiting is announced (a page restored from the back-forward cache,
 * or a frozen tab, may have missed its install).
 * @param {boolean} [force]
 */
export async function checkForUpdate(force = false) {
  const reg = registration;
  if (!reg) return "unknown";
  const now = Date.now();
  if (!force && now - lastCheck < THROTTLE_MS) return "throttled";
  lastCheck = now;
  try {
    await reg.update();
  } catch {
    return navigator.onLine === false ? "offline" : "unknown";
  }
  if (reg.waiting) announce(reg);
  return reg.installing || reg.waiting ? "updated" : "latest";
}

/**
 * Registers the worker and watches for updates. Safe to call more than once.
 * @param {{ onUpdate?: (reg: ServiceWorkerRegistration) => void }} [opts]
 */
export async function mountUpdates(opts = {}) {
  if (!canSw() || registration || buildVersion() === "dev") return;
  const sw = navigator.serviceWorker;
  const loadedAt = Date.now();
  let touched = false;
  for (const type of ["pointerdown", "keydown", "input"]) {
    window.addEventListener(type, () => { touched = true; }, { once: true, passive: true, capture: true });
  }
  const fresh = () => Date.now() - loadedAt < FRESH_MS && !touched;

  // A page that asked the worker to take over reloads into it.
  let switching = false;
  let controller = sw.controller;
  sw.addEventListener("controllerchange", () => {
    const before = controller;
    controller = sw.controller;
    // The first worker's clients.claim(): the page already runs its build.
    if (!before) return;
    if (switching || fresh()) location.reload();
    else blockingNotice();
  });

  /** @param {ServiceWorkerRegistration} reg */
  const takeOver = (reg) => {
    if (!reg.waiting) return location.reload();
    switching = true;
    reg.waiting.postMessage({ type: "SKIP_WAITING" });
  };

  const announced = new WeakSet();
  announce = async (reg) => {
    const waiting = reg.waiting;
    // No controller: first install, or a page loaded around the worker.
    if (!waiting || !sw.controller || announced.has(waiting)) return;
    announced.add(waiting);
    if (opts.onUpdate) return opts.onUpdate(reg);
    if (fresh()) {
      const reply = await ask(waiting, { type: "CLIENTS" });
      if (reply?.count === 1 && fresh()) return takeOver(reg);
    }
    toast("New version available.", { label: "Reload", run: () => takeOver(reg) });
  };

  let reg;
  try {
    reg = await sw.register(swUrl());
  } catch {
    return;
  }
  registration = reg;
  lastCheck = Date.now(); // register() checks sw.js itself
  if (reg.waiting) announce(reg);
  reg.addEventListener("updatefound", () => {
    const worker = reg.installing;
    worker?.addEventListener("statechange", () => {
      if (worker.state === "installed") announce(reg);
    });
  });
  // Resumes a group fill the browser cut short.
  sw.controller?.postMessage({ type: "FILL" });

  const check = () => { checkForUpdate().catch(() => {}); };
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") check(); });
  window.addEventListener("online", check);
  window.addEventListener("pageshow", (e) => { if (/** @type {PageTransitionEvent} */ (e).persisted) check(); });
}

/**
 * @typedef {object} ForceEnv
 * @property {CacheStorage} [caches]
 * @property {{ getRegistrations(): Promise<readonly { unregister(): Promise<boolean>, scope: string }[]> } | null} [serviceWorker]
 * @property {typeof fetch} [fetch]
 * @property {{ reload(): void, href: string }} [location]
 * @property {string} [app]
 * @property {string} [probe] a URL that must answer before anything is deleted
 */

/**
 * Starts over from the network: deletes this app's caches, unregisters its
 * worker and reloads. Returns "offline" with nothing deleted when the probe
 * fails, so the learner never loses the offline copy without a replacement.
 * @param {ForceEnv} [env]
 * @returns {Promise<"reloading" | "offline">}
 */
export async function forceUpdate(env = {}) {
  const c = env.caches ?? globalThis.caches;
  const sw = env.serviceWorker !== undefined ? env.serviceWorker : (globalThis.navigator?.serviceWorker ?? null);
  const doFetch = env.fetch ?? globalThis.fetch;
  const loc = env.location ?? globalThis.location;
  const prefix = cachePrefix(env.app ?? appId());
  const probe = env.probe ?? new URL("lk-precache.json", new URL(swUrlSafe(), loc.href)).href;
  try {
    const res = await doFetch(probe, { cache: "no-store" });
    if (!res.ok) return "offline";
  } catch {
    return "offline";
  }
  if (c) {
    const names = await c.keys();
    await Promise.all(names.filter((n) => n.startsWith(prefix)).map((n) => c.delete(n)));
  }
  if (sw) {
    const scope = new URL("./", new URL(swUrlSafe(), loc.href)).href;
    const regs = await sw.getRegistrations();
    // Other apps on the origin have their own registrations; leave them.
    await Promise.all(regs.filter((r) => r.scope === scope).map((r) => r.unregister()));
  }
  loc.reload();
  return "reloading";
}

function swUrlSafe() {
  return typeof document === "undefined" ? "./sw.js" : swUrl();
}
