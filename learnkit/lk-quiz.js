// <lk-quiz src="quiz.json"> and <lk-review>.
//
// Quiz file: {"id": "...", "title": "...", "questions": [Question, ...]}.
// id is required and must stay fixed: review cards are keyed by it.
// Question: {id, kind: "single"|"multi"|"numeric", text, options?, answer,
// tol?, unit?, hint?, explain?, topic?}. answer is an option index (single),
// a list of indexes (multi), or a number (numeric, right within ±tol).
// text, options, hint and explain are HTML written by the course author.

import { Base, define, h, todayKey, inAppScope, buildPaths } from "./lk-util.js";
import { createStore, registerSchema } from "./lk-store.js";
import { leitnerStore, record, dueCards, migratePathCards } from "./lk-leitner.js";

/**
 * @typedef {object} Question
 * @property {string} id
 * @property {"single" | "multi" | "numeric"} kind
 * @property {string} text
 * @property {string[]} [options]
 * @property {number | number[]} answer
 * @property {number} [tol]
 * @property {string} [unit]
 * @property {string} [hint]
 * @property {string} [explain]
 * @property {string} [topic]
 */

/**
 * Parses a typed number; accepts spaces, commas and underscores as digit
 * separators and a trailing exponent ("2.6e5").
 * @param {unknown} v
 */
export function parseNumber(v) {
  if (typeof v === "number") return v;
  const s = String(v ?? "").trim().replace(/[\s,_]/g, "");
  return s === "" ? NaN : Number(s);
}

/**
 * @param {Question} q
 * @param {unknown} ans
 */
export function gradeQuestion(q, ans) {
  if (ans === undefined || ans === null) return false;
  if (q.kind === "multi") {
    if (!Array.isArray(ans) || !Array.isArray(q.answer)) return false;
    const a = [...new Set(ans)].sort();
    const b = [...new Set(q.answer)].sort();
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  if (q.kind === "numeric") {
    const v = parseNumber(ans);
    return Number.isFinite(v) && Math.abs(v - /** @type {number} */ (q.answer)) <= (q.tol ?? 0);
  }
  return ans === q.answer;
}

/**
 * @param {Question[]} questions
 * @param {Record<string, unknown>} answers by question id
 */
export function scoreAll(questions, answers) {
  let score = 0, answered = 0;
  /** @type {string[]} */
  const missed = [];
  for (const q of questions) {
    const a = answers[q.id];
    if (a === undefined || a === null) continue;
    answered++;
    if (gradeQuestion(q, a)) score++;
    else missed.push(q.topic ?? q.id);
  }
  return { score, answered, total: questions.length, missed };
}

/**
 * Checks a quiz file's shape. Returns a list of faults, empty when sound.
 * @param {any} quiz
 */
export function checkQuiz(quiz) {
  const faults = [];
  if (!quiz || typeof quiz !== "object") return ["not a JSON object"];
  if (typeof quiz.id !== "string" || !quiz.id.trim()) faults.push('missing quiz "id" (a fixed string; review cards are keyed by it)');
  if (!Array.isArray(quiz.questions)) return [...faults, "no questions array"];
  const ids = new Set();
  for (const q of quiz.questions) {
    const where = `question ${q?.id ?? "?"}`;
    if (typeof q?.id !== "string" || ids.has(q.id)) faults.push(`${where}: missing or repeated id`);
    ids.add(q?.id);
    if (q.kind === "single" && !(Number.isInteger(q.answer) && q.options?.[q.answer] !== undefined)) faults.push(`${where}: answer is not an option index`);
    else if (q.kind === "multi" && !(Array.isArray(q.answer) && q.answer.every((/** @type {number} */ i) => q.options?.[i] !== undefined))) faults.push(`${where}: answer is not a list of option indexes`);
    else if (q.kind === "numeric" && !Number.isFinite(q.answer)) faults.push(`${where}: answer is not a number`);
    else if (!["single", "multi", "numeric"].includes(q.kind)) faults.push(`${where}: unknown kind ${q.kind}`);
  }
  return faults;
}

// Stored answers: option indexes, lists of them, or typed text.
registerSchema("quiz:", {
  version: 1,
  validate: (d) => {
    const a = /** @type {any} */ (d)?.answers;
    return !!a && typeof a === "object" && !Array.isArray(a) && Object.values(a).every((v) =>
      Number.isInteger(v) || typeof v === "string" || (Array.isArray(v) && v.every(Number.isInteger)));
  },
});

/** @type {Map<string, Promise<any>>} */
const quizCache = new Map();

/**
 * Fetches and checks a quiz file, once per URL.
 * @param {string} src as written in the page
 */
export function loadQuiz(src) {
  const url = new URL(src, document.baseURI).href;
  if (!quizCache.has(url)) {
    quizCache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }).then((quiz) => {
      const faults = checkQuiz(quiz);
      if (faults.length) throw new Error(`${src}: ${faults.join("; ")}`);
      return quiz;
    }));
  }
  return /** @type {Promise<any>} */ (quizCache.get(url));
}

/**
 * The quiz's URL path, kept in its cards so <lk-review> can load the file.
 * @param {string} src
 */
function srcKey(src) {
  return new URL(src, document.baseURI).pathname;
}

let uid = 0;

/**
 * Renders one question with its own Check button.
 * @param {Question} q
 * @param {number} n 1-based number shown to the learner
 * @param {(ans: unknown, right: boolean) => void} onCheck
 * @param {unknown} [prev] a stored answer to show as already checked
 */
function renderQuestion(q, n, onCheck, prev) {
  const id = `lk-q${++uid}`;
  const box = h("fieldset", { class: "lk-q", "aria-describedby": `${id}-fb` });
  box.append(h("legend", { class: "lk-q-text", html: `<span class="lk-q-n">${n}.</span> ${q.text}` }));
  /** @type {() => unknown} */
  let read;
  if (q.kind === "numeric") {
    const input = /** @type {HTMLInputElement} */ (h("input", { type: "text", inputmode: "decimal", id: `${id}-in`, class: "lk-input", "aria-label": "Your answer", autocomplete: "off" }));
    box.append(h("div", { class: "lk-num" }, input, q.unit ? h("span", { class: "lk-unit", text: q.unit }) : ""));
    read = () => (input.value.trim() === "" ? null : input.value);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); check(); } });
    if (prev != null) input.value = String(prev);
  } else {
    const type = q.kind === "multi" ? "checkbox" : "radio";
    const list = h("div", { class: "lk-options" });
    (q.options ?? []).forEach((opt, i) => {
      const inp = h("input", { type, name: id, value: String(i) });
      if (prev != null && (Array.isArray(prev) ? prev.includes(i) : prev === i)) /** @type {HTMLInputElement} */ (inp).checked = true;
      const lab = h("label", { class: "lk-option" }, inp);
      lab.insertAdjacentHTML("beforeend", ` <span>${opt}</span>`);
      list.append(lab);
    });
    if (q.kind === "multi") box.append(h("p", { class: "lk-muted lk-small", text: "Select all that apply." }));
    box.append(list);
    read = () => {
      const on = [...list.querySelectorAll("input:checked")].map((x) => Number(/** @type {HTMLInputElement} */ (x).value));
      if (q.kind === "multi") return on.length ? on : null;
      return on.length ? on[0] : null;
    };
  }
  const btn = h("button", { type: "button", class: "lk-btn", text: "Check" });
  const row = h("div", { class: "lk-row" }, btn);
  if (q.hint) {
    const hint = h("details", { class: "lk-hint" }, h("summary", { text: "Hint" }));
    hint.insertAdjacentHTML("beforeend", `<div>${q.hint}</div>`);
    row.append(hint);
  }
  box.append(row);
  const fb = h("div", { class: "lk-feedback", id: `${id}-fb`, role: "status", "aria-live": "polite" });
  box.append(fb);

  function show(/** @type {boolean} */ right) {
    box.classList.toggle("lk-right", right);
    box.classList.toggle("lk-wrong", !right);
    const truth = q.kind === "numeric"
      ? `${q.answer}${q.unit ? " " + q.unit : ""}${q.tol ? ` (±${q.tol})` : ""}`
      : q.kind === "multi"
        ? /** @type {number[]} */ (q.answer).map((i) => String.fromCharCode(65 + i)).join(", ")
        : String.fromCharCode(65 + /** @type {number} */ (q.answer));
    // explain is author HTML; the rest is built as text.
    fb.replaceChildren(h("strong", { text: right ? "Right." : "Not quite." }), right ? " " : ` Answer: ${truth}. `);
    if (q.explain) fb.append(h("span", { html: q.explain }));
  }
  function check() {
    const ans = read();
    if (ans === null) {
      fb.textContent = "Answer first.";
      return;
    }
    const right = gradeQuestion(q, ans);
    show(right);
    onCheck(ans, right);
  }
  btn.addEventListener("click", check);
  if (prev != null) show(gradeQuestion(q, prev));
  return box;
}

/** A quiz: every question on one page, a summary once all are answered. */
class LkQuiz extends Base {
  connectedCallback() {
    if (this.dataset.ready) return;
    this.dataset.ready = "1";
    const src = this.getAttribute("src");
    if (!src) return;
    this.setAttribute("aria-busy", "true");
    loadQuiz(src).then((quiz) => this.#render(quiz, src)).catch((err) => {
      this.textContent = `Quiz unavailable: ${err.message}`;
    }).finally(() => this.removeAttribute("aria-busy"));
  }

  /** @param {any} quiz @param {string} src */
  #render(quiz, src) {
    const store = createStore(`quiz:${quiz.id}`, { initial: () => ({ answers: {} }) });
    const boxes = leitnerStore();
    migratePathCards(boxes, srcKey(src), quiz.id);
    // Version 1.0 named the answers of a quiz without an id after its path.
    const old = createStore(`quiz:${srcKey(src)}`, { initial: () => ({ answers: {} }) });
    const oldAnswers = old.get().answers ?? {};
    if (Object.keys(oldAnswers).length) {
      if (!Object.keys(store.get().answers ?? {}).length) store.set({ answers: oldAnswers });
      old.drop();
    }
    /** @type {Record<string, unknown>} */
    const answers = store.get().answers ?? {};
    const summary = h("div", { class: "lk-summary", role: "status", "aria-live": "polite" });
    const reset = h("button", { type: "button", class: "lk-btn lk-ghost", text: "Start over" });
    const update = () => {
      const s = scoreAll(quiz.questions, answers);
      if (s.answered < s.total) {
        summary.textContent = `${s.answered} of ${s.total} answered.`;
        return;
      }
      summary.replaceChildren(h("strong", { text: `Score: ${s.score} / ${s.total}.` }), " ",
        s.missed.length ? `Missed: ${s.missed.join(", ")}. Missed questions come back in review.` : "All right.");
      this.dispatchEvent(new CustomEvent("lk-quiz-done", { bubbles: true, detail: s }));
    };
    this.replaceChildren(quiz.title ? h("h3", { class: "lk-quiz-title", text: quiz.title }) : "");
    quiz.questions.forEach((/** @type {Question} */ q, i) => {
      this.append(renderQuestion(q, i + 1, (ans, right) => {
        answers[q.id] = ans;
        store.set({ answers });
        record(boxes, { src: srcKey(src), quiz: quiz.id, q: q.id }, right, todayKey());
        this.dispatchEvent(new CustomEvent("lk-quiz-answer", { bubbles: true, detail: { id: q.id, right } }));
        update();
      }, answers[q.id]));
    });
    reset.addEventListener("click", () => {
      store.set({ answers: {} });
      this.dataset.ready = "";
      this.connectedCallback();
    });
    this.append(h("div", { class: "lk-row" }, summary, reset));
    update();
  }
}

/** Serves the questions due today from every quiz the learner missed. */
class LkReview extends Base {
  /** The learner has answered here since the last refresh. */
  started = false;
  runs = 0;

  connectedCallback() {
    if (this.dataset.ready) return;
    this.dataset.ready = "1";
    this.refresh();
    // A quiz on the same page may box a question. Show it at once, unless
    // the learner is working here: then offer it instead of wiping the work.
    document.addEventListener("lk-quiz-answer", () => {
      if (!this.started && !this.contains(document.activeElement)) return void this.refresh();
      if (this.querySelector(".lk-review-more")) return;
      const more = h("button", { type: "button", class: "lk-btn lk-ghost lk-review-more", text: "Show newly missed questions" });
      more.addEventListener("click", () => this.refresh());
      this.append(more);
    });
  }

  async refresh() {
    const run = ++this.runs;
    this.started = false;
    const boxes = leitnerStore();
    const today = todayKey();
    const due = dueCards(boxes.get(), today);
    if (!due.length) {
      this.replaceChildren(h("p", { class: "lk-muted", text: "Nothing to review today." }));
      return;
    }
    const head = h("p", { class: "lk-review-count", role: "status", "aria-live": "polite", text: `${due.length} question${due.length > 1 ? "s" : ""} due today.` });
    this.replaceChildren(head);
    const allowed = await buildPaths();
    if (run !== this.runs) return;
    let n = 0;
    for (const card of due) {
      /** @type {Question | undefined} */
      let q;
      /** @type {string} */
      let quizId = card.quiz ?? "";
      // Stored paths may come from an imported file: only files of this
      // build (any file of the app before a release build).
      if (!inAppScope(card.src) || (allowed && !allowed.has(card.src))) continue;
      try {
        const quiz = await loadQuiz(card.src);
        if (run !== this.runs) return; // a newer refresh took over
        if (!card.quiz) migratePathCards(boxes, card.src, quiz.id);
        quizId = quiz.id;
        q = quiz.questions.find((/** @type {Question} */ x) => x.id === card.q);
      } catch {
        q = undefined;
      }
      if (!q) continue; // question removed from its quiz or file unreachable offline
      const ref = { src: card.src, quiz: quizId, q: card.q };
      const box = renderQuestion(/** @type {Question} */ (q), ++n, (_ans, right) => {
        this.started = true;
        const after = record(boxes, ref, right, today);
        box.append(h("p", { class: "lk-muted lk-small", text: after ? `Box ${after.b} of 3.` : "" }));
      });
      box.append(h("p", { class: "lk-muted lk-small", text: `From ${card.src.split("/").pop()}, box ${card.b}.` }));
      this.append(box);
    }
  }
}

define("lk-quiz", LkQuiz);
define("lk-review", LkReview);
