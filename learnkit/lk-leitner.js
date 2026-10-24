// Leitner review with 3 boxes, shared by every quiz of the app.
//
// Box 1 is due again the same day, box 2 one day after its last answer, box 3
// four days after. A missed question enters box 1. A right answer to a due
// card moves it up one box; a right answer before it is due changes nothing,
// so replaying a quiz the same day cannot climb the boxes. A wrong answer
// sends the card back to box 1. Questions never missed stay out of the boxes.
//
// Card id: "<quiz id>#<question id>", so a quiz file can move without losing
// its cards. Each card also keeps the quiz's path (src) for <lk-review> to
// load it. Version 1.0 keyed cards by path; migratePathCards() moves them.
// Dates are local YYYY-MM-DD strings passed in by the caller, so tests set
// the clock.

import { createStore, registerSchema } from "./lk-store.js";
import { hasSneakySeparator } from "./lk-util.js";

/** Days after the last answer before a card is due, by box (index 0 unused). */
export const WAIT_DAYS = [0, 0, 1, 4];
export const MAX_BOX = 3;

/**
 * @typedef {{ b: 1 | 2 | 3, d: string, src: string, q: string, quiz?: string }} Card
 * quiz is missing only on path-keyed cards not yet migrated.
 * @typedef {{ src: string, quiz: string, q: string }} Ref
 * @typedef {{ items: Record<string, Card> }} Deck
 */

/**
 * A card as stored. src must be a root-relative path ("/course/q.json") with
 * no encoded separators: an imported file must not point review at another
 * origin, a data: URL, or another app's folder.
 * @param {unknown} v
 */
function isCard(v) {
  const c = /** @type {any} */ (v);
  return !!c && typeof c === "object" && [1, 2, 3].includes(c.b) && typeof c.d === "string"
    && typeof c.src === "string" && /^\/(?![\/\\])/.test(c.src) && !hasSneakySeparator(c.src) && typeof c.q === "string"
    && (c.quiz === undefined || typeof c.quiz === "string");
}

/** @param {unknown} d */
function isDeck(d) {
  const items = /** @type {any} */ (d)?.items;
  return !!items && typeof items === "object" && !Array.isArray(items) && Object.values(items).every(isCard);
}

registerSchema("leitner", { version: 1, validate: isDeck });

/**
 * @param {{ storage?: Storage | null, app?: string }} [opts]
 */
export function leitnerStore(opts = {}) {
  return createStore("leitner", {
    version: 1,
    initial: () => /** @type {Deck} */ ({ items: {} }),
    validate: isDeck,
    ...opts,
  });
}

/** @param {string} key */
function dayNum(key) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(key);
  return m ? Math.round(Date.UTC(+m[1], +m[2] - 1, +m[3]) / 86400000) : null;
}

/**
 * Due on `today`. A card with a bad date, or dated after today (the clock went
 * back), is due.
 * @param {{ b: number, d: string }} card
 * @param {string} today
 */
export function isDue(card, today) {
  const last = dayNum(card.d);
  const now = dayNum(today);
  if (last === null || now === null || last > now) return true;
  return now - last >= WAIT_DAYS[card.b];
}

/**
 * The card after one answer, or null when an unboxed question was answered
 * right (it stays out).
 * @param {Card | null} cur
 * @param {boolean} right
 * @param {string} today
 * @param {Ref} ref
 * @returns {Card | null}
 */
export function nextCard(cur, right, today, ref) {
  if (!right) return { b: 1, d: today, src: ref.src, quiz: ref.quiz, q: ref.q };
  if (!cur) return null;
  if (!isDue(cur, today)) return { ...cur, src: ref.src, quiz: ref.quiz };
  return { ...cur, src: ref.src, quiz: ref.quiz, b: /** @type {1|2|3} */ (Math.min(cur.b + 1, MAX_BOX)), d: today };
}

/** @param {string} quiz the quiz's id @param {string} q */
export const cardId = (quiz, q) => `${quiz}#${q}`;

/**
 * Records one answer and returns the card, or null when not boxed.
 * @param {ReturnType<typeof leitnerStore>} store
 * @param {Ref} ref
 * @param {boolean} right
 * @param {string} today
 */
export function record(store, ref, right, today) {
  const id = cardId(ref.quiz, ref.q);
  let out = /** @type {Card | null} */ (null);
  store.update((deck) => {
    out = nextCard(deck.items[id] ?? null, right, today, ref);
    if (out) deck.items[id] = out;
  });
  return out;
}

/**
 * Due cards, box 1 first, then the oldest answer, then id.
 * @param {Deck} deck
 * @param {string} today
 * @returns {Card[]}
 */
export function dueCards(deck, today) {
  return Object.entries(deck.items)
    .filter(([, c]) => isDue(c, today))
    .sort(([ia, a], [ib, b]) => a.b - b.b || (a.d < b.d ? -1 : a.d > b.d ? 1 : 0) || (ia < ib ? -1 : 1))
    .map(([, c]) => c);
}

/**
 * Moves cards keyed "<path>#<question>" (version 1.0) to "<quiz id>#<question>".
 * Runs whenever a quiz loads; once moved, nothing matches, so it writes only
 * the first time. When both keys exist, the id-keyed card wins.
 * @param {ReturnType<typeof leitnerStore>} store
 * @param {string} src the quiz's path, as stored in the old cards
 * @param {string} quiz the quiz's id
 * @returns {number} cards moved
 */
export function migratePathCards(store, src, quiz) {
  const deck = store.get();
  const old = Object.entries(deck.items).filter(([k, c]) => !c.quiz && c.src === src && k === `${src}#${c.q}`);
  if (!old.length) return 0;
  for (const [k, c] of old) {
    delete deck.items[k];
    const id = cardId(quiz, c.q);
    if (!deck.items[id]) deck.items[id] = { ...c, quiz };
  }
  store.set(deck);
  return old.length;
}
