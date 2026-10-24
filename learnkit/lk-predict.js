// <lk-predict id="..."> and <lk-calibration>.
//
// The learner types a predicted number, then Reveal shows the true value. The
// true value comes from the element's `truth` property (a function returning
// a number) or else from a child [slot="answer"] with data-value="<number>";
// that child is shown on reveal and may explain the result. Each guess is
// stored under the element's id, so <lk-calibration> on any page of the app
// can show how far off the learner tends to be.

import { Base, define, h, fmt } from "./lk-util.js";
import { createStore, registerSchema } from "./lk-store.js";
import { parseNumber } from "./lk-quiz.js";

/**
 * @typedef {{ guess: number, truth: number, log10: number | null, at: string, label?: string }} Prediction
 * log10 = log10(guess / truth), signed; 0 is exact, 1 is ten times too high.
 */

/**
 * Stored predictions hold numbers only (log10 is null when it had no
 * meaning: JSON turns NaN into null), so an imported file cannot smuggle
 * markup into the page.
 * @param {unknown} d
 */
function isPredictions(d) {
  if (!d || typeof d !== "object" || Array.isArray(d)) return false;
  return Object.values(d).every((p) => !!p && typeof p === "object"
    && Number.isFinite(p.guess) && Number.isFinite(p.truth)
    && (p.log10 === null || Number.isFinite(p.log10))
    && typeof p.at === "string" && (p.label === undefined || typeof p.label === "string"));
}

registerSchema("predict", { version: 1, validate: isPredictions });

/** @param {{ storage?: Storage | null, app?: string }} [opts] */
export function predictStore(opts = {}) {
  return createStore("predict", {
    initial: () => /** @type {Record<string, Prediction>} */ ({}),
    validate: isPredictions,
    ...opts,
  });
}

/**
 * The gap between a guess and the truth, as a signed order of magnitude.
 * Null when either is not a positive number (a ratio means nothing then).
 * @param {number} guess
 * @param {number} truth
 */
export function gap(guess, truth) {
  if (!(guess > 0 && truth > 0)) return null;
  return Math.log10(guess / truth);
}

/**
 * Summary over all stored predictions.
 * @param {Record<string, Prediction>} all
 */
export function calibration(all) {
  const list = Object.values(all).filter((p) => Number.isFinite(p.log10));
  if (!list.length) return { n: 0, meanAbs: 0, bias: 0, within2x: 0 };
  const meanAbs = list.reduce((s, p) => s + Math.abs(/** @type {number} */ (p.log10)), 0) / list.length;
  const bias = list.reduce((s, p) => s + /** @type {number} */ (p.log10), 0) / list.length;
  const within2x = list.filter((p) => Math.abs(/** @type {number} */ (p.log10)) <= Math.log10(2)).length;
  return { n: list.length, meanAbs, bias, within2x };
}

/** @param {number} lg */
function describe(lg) {
  const factor = 10 ** Math.abs(lg);
  if (factor < 1.05) return "spot on";
  return `${fmt(factor)}× too ${lg > 0 ? "high" : "low"}`;
}

class LkPredict extends Base {
  /** @type {(() => number) | null} */ truth = null;

  connectedCallback() {
    if (this.dataset.ready) return;
    this.dataset.ready = "1";
    if (!this.id) console.warn("lk-predict needs an id to store the guess");
    const answer = /** @type {HTMLElement | null} */ (this.querySelector('[slot="answer"]'));
    const reveal = h("div", { class: "lk-reveal", hidden: true, role: "status", "aria-live": "polite" });
    if (answer) {
      answer.hidden = true;
    }
    const input = /** @type {HTMLInputElement} */ (h("input", { class: "lk-input", type: "text", inputmode: "decimal", autocomplete: "off", "aria-label": this.getAttribute("label") || "Your prediction" }));
    const btn = /** @type {HTMLButtonElement} */ (h("button", { type: "button", class: "lk-btn", text: "Reveal", disabled: true }));
    const again = h("button", { type: "button", class: "lk-btn lk-ghost", text: "Try again", hidden: true });
    const row = h("div", { class: "lk-row" }, input, btn, again);
    if (answer) this.insertBefore(row, answer); else this.append(row);
    this.append(reveal);
    if (answer) reveal.append(answer);

    const result = h("p");
    reveal.prepend(result);
    const store = predictStore();

    const value = () => {
      if (this.truth) return Number(this.truth());
      return parseNumber(answer?.dataset.value);
    };
    /** @param {Prediction} p */
    const show = (p) => {
      reveal.hidden = false;
      if (answer) answer.hidden = false;
      input.value = String(p.guess);
      input.disabled = true;
      btn.hidden = true;
      again.hidden = false;
      result.replaceChildren("You said ", h("strong", { text: fmt(p.guess) }), ". The true value is ",
        h("strong", { text: fmt(p.truth) }), ": ",
        h("span", { class: "lk-gap", text: Number.isFinite(p.log10) ? describe(p.log10) : "no ratio for non-positive values" }), ".");
    };

    input.addEventListener("input", () => { btn.disabled = !Number.isFinite(parseNumber(input.value)); });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !btn.disabled) btn.click(); });
    btn.addEventListener("click", () => {
      const guess = parseNumber(input.value);
      const truth = value();
      if (!Number.isFinite(guess) || !Number.isFinite(truth)) return;
      /** @type {Prediction} */
      const p = { guess, truth, log10: gap(guess, truth), at: new Date().toISOString(), label: this.getAttribute("label") ?? undefined };
      if (this.id) store.update((all) => { all[this.id] = p; });
      show(p);
      this.dispatchEvent(new CustomEvent("lk-predict", { bubbles: true, detail: p }));
    });
    again.addEventListener("click", () => {
      input.disabled = false;
      input.value = "";
      btn.hidden = false;
      btn.disabled = true;
      again.hidden = true;
      reveal.hidden = true;
      input.focus();
    });
    const saved = this.id ? store.get()[this.id] : undefined;
    if (saved) show(saved);
  }
}

/** Shows the learner's calibration over every stored prediction. */
class LkCalibration extends Base {
  connectedCallback() {
    this.render();
    document.addEventListener("lk-predict", () => this.render());
  }

  render() {
    const all = predictStore().get();
    const c = calibration(all);
    if (!c.n) {
      this.replaceChildren(h("p", { class: "lk-muted", text: "No predictions yet." }));
      return;
    }
    const lean = Math.abs(c.bias) < 0.05 ? "no lean either way" : `a lean ${c.bias > 0 ? "high" : "low"} of ${fmt(10 ** Math.abs(c.bias))}×`;
    this.replaceChildren(h("p", { role: "status", text:
      `${c.n} prediction${c.n > 1 ? "s" : ""}: typically off by ${fmt(10 ** c.meanAbs)}×, ${lean}; ${c.within2x} within a factor of 2.` }));
  }
}

define("lk-predict", LkPredict);
define("lk-calibration", LkCalibration);
