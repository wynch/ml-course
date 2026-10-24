// <lk-stepper steps="20" interval="700">: play, pause, step and scrub through
// an animation or an algorithm.
//
// Set one of, from page script:
//   el.render = (t, el) => {...}            draws frame t (0 .. steps-1)
//   el.init = () => state; el.step = (state, t) => next state;
//   el.draw = (state, t, el) => {...}       for step-by-step algorithms
// In step mode, states are computed forward once and kept, so stepping back
// is exact. el.stage is the element to draw into. Autoplay ("autoplay"
// attribute) is skipped when the learner asked for reduced motion.

import { Base, define, h, reducedMotion } from "./lk-util.js";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

class LkStepper extends Base {
  /** @type {((t: number, el: LkStepper) => void) | null} */ #render = null;
  /** @type {(() => any) | null} */ #init = null;
  /** @type {((state: any, t: number) => any) | null} */ #step = null;
  /** @type {((state: any, t: number, el: LkStepper) => void) | null} */ #draw = null;
  // Setting any hook redraws, so page script may set them at any time.
  get render() { return this.#render; }
  set render(fn) { this.#render = fn; this.#redraw(); }
  get init() { return this.#init; }
  set init(fn) { this.#init = fn; this.states = []; this.#redraw(); }
  get step() { return this.#step; }
  set step(fn) { this.#step = fn; this.states = []; this.#redraw(); }
  get draw() { return this.#draw; }
  set draw(fn) { this.#draw = fn; this.#redraw(); }
  #redraw() { if (this.slider) this.seek(this.t); }
  t = 0;
  speed = 1;
  /** @type {any[]} */ states = [];
  /** @type {ReturnType<typeof setTimeout> | null} */ timer = null;

  get steps() {
    return Math.max(1, Number(this.getAttribute("steps")) || 10);
  }

  connectedCallback() {
    if (this.dataset.ready) return;
    this.dataset.ready = "1";
    // Hooks set before the element upgraded are plain properties; move them
    // behind the setters.
    for (const k of ["render", "init", "step", "draw"]) {
      if (Object.hasOwn(this, k)) {
        const v = /** @type {any} */ (this)[k];
        delete /** @type {any} */ (this)[k];
        /** @type {any} */ (this)[k] = v;
      }
    }
    this.setAttribute("role", "group");
    if (!this.hasAttribute("aria-label")) this.setAttribute("aria-label", "Animation");
    this.stage = this.querySelector(".lk-stage") || h("div", { class: "lk-stage" });
    if (!this.stage.parentNode) this.prepend(this.stage);
    const mk = (/** @type {string} */ label, /** @type {string} */ text) => h("button", { type: "button", class: "lk-btn lk-ghost", "aria-label": label, text });
    this.backBtn = mk("Step back", "◀︎");
    this.playBtn = h("button", { type: "button", class: "lk-btn", "aria-label": "Play", text: "Play" });
    this.fwdBtn = mk("Step forward", "▶︎");
    this.slider = /** @type {HTMLInputElement} */ (h("input", { type: "range", min: "0", max: String(this.steps - 1), value: "0", step: "1", "aria-label": "Time" }));
    this.speedSel = /** @type {HTMLSelectElement} */ (h("select", { "aria-label": "Speed" }));
    for (const s of SPEEDS) this.speedSel.append(h("option", { value: String(s), text: `${s}×`, selected: s === 1 }));
    this.label = h("span", { class: "lk-t", "aria-live": "off" });
    this.append(h("div", { class: "lk-controls" }, this.backBtn, this.playBtn, this.fwdBtn, this.slider, this.speedSel, this.label));

    this.backBtn.addEventListener("click", () => { this.pause(); this.seek(this.t - 1); });
    this.fwdBtn.addEventListener("click", () => { this.pause(); this.seek(this.t + 1); });
    this.playBtn.addEventListener("click", () => (this.timer ? this.pause() : this.play()));
    this.slider.addEventListener("input", () => { this.pause(); this.seek(Number(this.slider.value)); });
    this.speedSel.addEventListener("change", () => { this.speed = Number(this.speedSel.value); });
    this.seek(0);
    if (this.hasAttribute("autoplay") && !reducedMotion()) this.play();
  }

  disconnectedCallback() {
    this.pause();
  }

  /** Drops kept states, for when the inputs of a step function change. */
  reset() {
    this.states = [];
    this.seek(Math.min(this.t, this.steps - 1));
  }

  /** @param {number} t */
  seek(t) {
    const max = this.steps - 1;
    this.t = Math.max(0, Math.min(max, Math.round(t)));
    this.slider.max = String(max);
    this.slider.value = String(this.t);
    this.slider.setAttribute("aria-valuetext", `Step ${this.t + 1} of ${max + 1}`);
    this.label.textContent = `${this.t + 1} / ${max + 1}`;
    this.backBtn.disabled = this.t === 0;
    this.fwdBtn.disabled = this.t === max;
    if (this.render) this.render(this.t, this);
    else if (this.step && this.draw) {
      if (!this.states.length) this.states.push(this.init ? this.init() : null);
      while (this.states.length <= this.t) this.states.push(this.step(this.states[this.states.length - 1], this.states.length));
      this.draw(this.states[this.t], this.t, this);
    }
    this.dispatchEvent(new CustomEvent("lk-step", { detail: { t: this.t } }));
  }

  play() {
    if (this.t >= this.steps - 1) this.seek(0);
    this.playBtn.textContent = "Pause";
    this.playBtn.setAttribute("aria-label", "Pause");
    const tick = () => {
      if (this.t >= this.steps - 1) return this.pause();
      this.seek(this.t + 1);
      this.timer = setTimeout(tick, this.interval / this.speed);
    };
    this.timer = setTimeout(tick, this.interval / this.speed);
  }

  pause() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.playBtn) {
      this.playBtn.textContent = "Play";
      this.playBtn.setAttribute("aria-label", "Play");
    }
  }

  get interval() {
    return Number(this.getAttribute("interval")) || 700;
  }
}

define("lk-stepper", LkStepper);
