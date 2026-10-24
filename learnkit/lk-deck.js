// <lk-deck>: each child <section> is a slide.
//
// Keys when the deck has focus: Right, Down, PageDown and Space go forward;
// Left, Up and PageUp go back; Home and End jump; "o" toggles the overview.
// Swipes go forward and back. Elements with data-step="n" (n from 1) appear
// one step at a time before the deck moves on. The URL hash "#slide-n"
// (1-based; "#<id>-slide-n" for a deck with an id) opens slide n.
// The visible section gets a bubbling "lk-slide-enter" event, so explorables
// inside it can start or resize.

import { Base, define, h, reducedMotion } from "./lk-util.js";

const SWIPE_PX = 50;

class LkDeck extends Base {
  /** @type {HTMLElement[]} */ slides = [];
  index = 0;
  step = 0;

  connectedCallback() {
    if (this.dataset.ready) return;
    this.dataset.ready = "1";
    this.slides = /** @type {HTMLElement[]} */ ([...this.children].filter((c) => c.tagName === "SECTION"));
    if (!this.slides.length) return;
    this.tabIndex = 0;
    this.setAttribute("role", "region");
    this.setAttribute("aria-roledescription", "slide deck");
    if (!this.hasAttribute("aria-label") && !this.hasAttribute("aria-labelledby")) this.setAttribute("aria-label", this.getAttribute("title") || "Slides");
    this.slides.forEach((s, i) => {
      s.setAttribute("role", "group");
      s.setAttribute("aria-roledescription", "slide");
      s.setAttribute("aria-label", `${i + 1} of ${this.slides.length}`);
    });

    this.bar = h("div", { class: "lk-deck-bar", role: "progressbar", "aria-label": "Deck progress", "aria-valuemin": "1", "aria-valuemax": String(this.slides.length) }, h("div"));
    this.prevBtn = h("button", { type: "button", class: "lk-btn lk-ghost", "aria-label": "Previous", text: "←" });
    this.nextBtn = h("button", { type: "button", class: "lk-btn", "aria-label": "Next", text: "→" });
    this.overBtn = h("button", { type: "button", class: "lk-btn lk-ghost", "aria-pressed": "false", text: "Overview" });
    this.count = h("span", { class: "lk-deck-count", "aria-live": "polite" });
    this.overview = h("ul", { class: "lk-overview", hidden: true, "aria-label": "All slides" });
    this.slides.forEach((s, i) => {
      const title = s.querySelector("h1,h2,h3,h4")?.textContent?.trim() || `Slide ${i + 1}`;
      const b = h("button", { type: "button" }, h("span", { class: "lk-n", text: String(i + 1) }), title);
      b.addEventListener("click", () => { this.toggleOverview(false); this.go(i, 0); this.focus(); });
      this.overview.append(h("li", {}, b));
    });
    this.prepend(this.bar);
    this.append(this.overview, h("div", { class: "lk-deck-nav" }, this.prevBtn, this.nextBtn, this.overBtn, this.count));

    this.prevBtn.addEventListener("click", () => this.prev());
    this.nextBtn.addEventListener("click", () => this.next());
    this.overBtn.addEventListener("click", () => this.toggleOverview());
    this.addEventListener("keydown", (e) => this.#onKey(e));
    this.#swipe();
    window.addEventListener("hashchange", () => this.#fromHash());
    if (!this.#fromHash()) this.go(0, 0, false);
  }

  get hashPrefix() {
    return this.id ? `${this.id}-slide-` : "slide-";
  }

  /** Number of steps in a slide: its highest data-step. */
  stepsOf(i) {
    return Math.max(0, ...[...this.slides[i].querySelectorAll("[data-step]")].map((el) => Number(el.getAttribute("data-step")) || 0));
  }

  /**
   * @param {number} i slide index
   * @param {number} step fragments shown, 0 for none
   * @param {boolean} [writeHash]
   */
  go(i, step, writeHash = true) {
    const n = this.slides.length;
    i = Math.max(0, Math.min(n - 1, i));
    const moved = i !== this.index || !this.dataset.shown;
    const dir = i >= this.index ? 1 : -1;
    this.index = i;
    this.step = Math.max(0, Math.min(this.stepsOf(i), step));
    this.dataset.shown = "1";
    this.slides.forEach((s, k) => {
      s.hidden = k !== i;
      s.setAttribute("aria-hidden", String(k !== i));
    });
    for (const el of this.slides[i].querySelectorAll("[data-step]")) {
      const hidden = (Number(el.getAttribute("data-step")) || 0) > this.step;
      el.classList.toggle("lk-hidden", hidden);
      el.toggleAttribute("aria-hidden", hidden);
    }
    const slide = this.slides[i];
    if (moved) {
      if (!reducedMotion()) {
        slide.style.setProperty("--lk-from", `${12 * dir}px`);
        slide.classList.remove("lk-enter");
        void slide.offsetWidth; // restart the animation
        slide.classList.add("lk-enter");
      }
      slide.dispatchEvent(new CustomEvent("lk-slide-enter", { bubbles: true, detail: { index: i } }));
    }
    /** @type {HTMLElement} */ (this.bar.firstChild).style.width = `${((i + 1) / n) * 100}%`;
    this.bar.setAttribute("aria-valuenow", String(i + 1));
    this.count.textContent = `${i + 1} / ${n}`;
    const lost = [this.prevBtn, this.nextBtn].includes(/** @type {any} */ (document.activeElement));
    this.prevBtn.disabled = i === 0 && this.step === 0;
    this.nextBtn.disabled = i === n - 1 && this.step >= this.stepsOf(i);
    // A disabled button drops focus to the body; keep it in the deck.
    if (lost && /** @type {HTMLButtonElement} */ (document.activeElement)?.disabled) this.focus();
    this.overview.querySelectorAll("button").forEach((b, k) => b.setAttribute("aria-current", String(k === i)));
    if (writeHash && moved) history.replaceState(null, "", `#${this.hashPrefix}${i + 1}`);
  }

  next() {
    if (this.step < this.stepsOf(this.index)) this.go(this.index, this.step + 1);
    else if (this.index < this.slides.length - 1) this.go(this.index + 1, 0);
  }

  prev() {
    if (this.step > 0) this.go(this.index, this.step - 1);
    else if (this.index > 0) this.go(this.index - 1, this.stepsOf(this.index - 1));
  }

  /** @param {boolean} [show] */
  toggleOverview(show = this.overview.hidden) {
    this.overview.hidden = !show;
    this.overBtn.setAttribute("aria-pressed", String(show));
    this.slides[this.index].hidden = show;
    if (show) /** @type {HTMLElement | null} */ (this.overview.querySelector('[aria-current="true"]'))?.focus();
    else if (this.overview.contains(document.activeElement) || document.activeElement === document.body) this.focus();
  }

  #fromHash() {
    const m = location.hash.slice(1).match(new RegExp(`^${this.hashPrefix}(\\d+)$`));
    if (!m) return false;
    this.go(Number(m[1]) - 1, 0, false);
    return true;
  }

  /** @param {KeyboardEvent} e */
  #onKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const t = /** @type {HTMLElement} */ (e.target);
    // Keys typed into a control inside a slide belong to that control.
    if (t !== this && t.closest("input, textarea, select, [contenteditable], lk-stepper, lk-quiz, lk-predict")) return;
    const k = e.key;
    // Space and Enter on a button or link press it; the deck must not act twice.
    if ((k === " " || k === "Enter") && t !== this && t.closest("button, a, summary")) return;
    if (k === "o" || (k === "Escape" && !this.overview.hidden)) this.toggleOverview();
    else if (!this.overview.hidden) return;
    else if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(k)) this.next();
    else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(k)) this.prev();
    else if (k === "Home") this.go(0, 0);
    else if (k === "End") this.go(this.slides.length - 1, this.stepsOf(this.slides.length - 1));
    else return;
    e.preventDefault();
  }

  #swipe() {
    /** @type {{ x: number, y: number } | null} */
    let start = null;
    this.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "mouse") return;
      start = { x: e.clientX, y: e.clientY };
    });
    this.addEventListener("pointerup", (e) => {
      if (!start) return;
      const dx = e.clientX - start.x, dy = e.clientY - start.y;
      start = null;
      if (Math.abs(dx) < SWIPE_PX || Math.abs(dx) < Math.abs(dy)) return;
      if (dx < 0) this.next(); else this.prev();
    });
    this.addEventListener("pointercancel", () => { start = null; });
  }
}

define("lk-deck", LkDeck);
