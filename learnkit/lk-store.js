// Namespaced localStorage with a backup copy, schema versions, and export and
// import of all progress.
//
// Key: "<app>:<name>". Value: {"v": <schema version>, "data": <any JSON>}.
// Before each write the previous value, if readable, is copied to
// "<app>:<name>.bak"; a read that finds the main value unreadable falls back
// to it. When storage throws (private mode, quota), values live in memory for
// the rest of the page's life.

import { appId } from "./lk-util.js";

export const EXPORT_FORMAT = "learnkit-progress";
export const EXPORT_FORMAT_VERSION = 1;
const BAK = ".bak";

/** @type {Map<string, string>} values whose write to storage failed */
const mem = new Map();

/** @returns {Storage | null} */
function defaultStorage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/**
 * @param {Storage | null} storage
 * @param {string} key
 * @returns {string | null}
 */
function readRaw(storage, key) {
  if (mem.has(key)) return /** @type {string} */ (mem.get(key));
  try {
    return storage ? storage.getItem(key) : null;
  } catch {
    return null;
  }
}

/**
 * @param {Storage | null} storage
 * @param {string} key
 * @param {string} value
 */
function writeRaw(storage, key, value) {
  try {
    if (!storage) throw new Error("no storage");
    storage.setItem(key, value);
    if (storage.getItem(key) !== value) throw new Error("write did not read back");
    mem.delete(key);
  } catch {
    mem.set(key, value);
  }
}

/**
 * @typedef {{ v: number, data: unknown }} Envelope
 */

/**
 * @param {unknown} value a JSON string or an already parsed object
 * @returns {Envelope | null}
 */
export function parseEnvelope(value) {
  let obj = value;
  if (typeof value === "string") {
    try {
      obj = JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
  const env = /** @type {Record<string, unknown>} */ (obj);
  if (!Number.isInteger(env.v) || /** @type {number} */ (env.v) < 1 || !("data" in env)) return null;
  return { v: /** @type {number} */ (env.v), data: env.data };
}

/**
 * Writes an envelope, first copying the current value to the backup key when
 * it is readable. An unreadable value never overwrites a good backup. Refuses,
 * returning false, when the stored value or its backup has a newer schema
 * than `maxVersion`: an old tab must not destroy what a newer build wrote.
 * @param {Storage | null} storage
 * @param {string} key
 * @param {Envelope} env
 * @param {number} [maxVersion]
 */
function writeWithBackup(storage, key, env, maxVersion = Infinity) {
  const prev = readRaw(storage, key);
  const prevEnv = prev === null ? null : parseEnvelope(prev);
  const bakEnv = parseEnvelope(readRaw(storage, key + BAK));
  if ((prevEnv && prevEnv.v > maxVersion) || (bakEnv && bakEnv.v > maxVersion)) return false;
  if (prevEnv) writeRaw(storage, key + BAK, /** @type {string} */ (prev));
  writeRaw(storage, key, JSON.stringify(env));
  return true;
}

/**
 * @typedef {{ version: number, validate: (data: unknown) => boolean }} Schema
 */

/** @type {Map<string, Schema>} by store name; a name ending in ":" covers every name it starts */
const schemas = new Map();

/**
 * Declares the shape of a store's data, so importProgress() can refuse a
 * crafted file. Kit modules register theirs when loaded.
 * @param {string} name a store name, or a prefix ending in ":"
 * @param {Schema} schema
 */
export function registerSchema(name, schema) {
  schemas.set(name, schema);
}

/** @param {string} name */
function schemaFor(name) {
  if (schemas.has(name)) return schemas.get(name);
  for (const [k, v] of schemas) if (k.endsWith(":") && name.startsWith(k)) return v;
  return undefined;
}

/**
 * @template T
 * @typedef {object} StoreOptions
 * @property {number} [version] schema version, an integer from 1
 * @property {() => T} [initial] the value when nothing readable is stored
 * @property {(data: any, from: number) => T} [migrate] upgrades data written by an older schema
 * @property {(data: unknown) => boolean} [validate] rejects data of the right version but wrong shape
 * @property {Storage | null} [storage]
 * @property {string} [app] defaults to the page's lk-app meta
 */

/**
 * @template T
 * @typedef {object} Store
 * @property {string} key
 * @property {() => T} get
 * @property {(data: T) => boolean} set false when a newer schema is stored
 * @property {(fn: (data: T) => T | void) => T} update
 * @property {() => T | null} backup the backup copy's data, if readable
 * @property {() => boolean} restore puts the backup back as the main value
 * @property {() => void} drop removes the value and its backup
 */

/**
 * @template T
 * @param {string} name
 * @param {StoreOptions<T>} [opts]
 * @returns {Store<T>}
 */
export function createStore(name, opts = {}) {
  const version = opts.version ?? 1;
  const initial = opts.initial ?? (() => /** @type {T} */ (/** @type {unknown} */ ({})));
  const validate = opts.validate ?? schemaFor(name)?.validate ?? (() => true);
  const storage = opts.storage === undefined ? defaultStorage() : opts.storage;
  const key = `${opts.app ?? appId()}:${name}`;

  /**
   * @param {string | null} raw
   * @returns {{ data: T, migrated: boolean } | null}
   */
  function decode(raw) {
    if (raw === null) return null;
    const env = parseEnvelope(raw);
    // A newer schema than this code knows is unreadable here, and kept.
    if (!env || env.v > version) return null;
    let data = env.data;
    let migrated = false;
    if (env.v < version) {
      if (!opts.migrate) return null;
      try {
        data = opts.migrate(data, env.v);
      } catch {
        return null;
      }
      migrated = true;
    }
    return validate(data) ? { data: /** @type {T} */ (data), migrated } : null;
  }

  /** @param {T} data */
  function set(data) {
    return writeWithBackup(storage, key, { v: version, data }, version);
  }

  function get() {
    const main = decode(readRaw(storage, key));
    if (main) {
      if (main.migrated) set(main.data);
      return main.data;
    }
    const bak = decode(readRaw(storage, key + BAK));
    if (bak) return bak.data;
    return initial();
  }

  return {
    key,
    get,
    set,
    update(fn) {
      const cur = get();
      const next = fn(cur);
      const out = next === undefined ? cur : next;
      set(out);
      return out;
    },
    backup() {
      return decode(readRaw(storage, key + BAK))?.data ?? null;
    },
    drop() {
      for (const k of [key, key + BAK]) {
        mem.delete(k);
        try {
          storage?.removeItem(k);
        } catch {
          /* unwritable storage: nothing to remove */
        }
      }
    },
    restore() {
      const raw = readRaw(storage, key + BAK);
      if (!decode(raw)) return false;
      writeRaw(storage, key, /** @type {string} */ (raw));
      return true;
    },
  };
}

/**
 * @param {Storage | null} storage
 * @returns {string[]}
 */
function storageKeys(storage) {
  const out = new Set(mem.keys());
  try {
    if (storage) for (let i = 0; i < storage.length; i++) {
      const k = storage.key(i);
      if (k !== null) out.add(k);
    }
  } catch {
    /* unreadable storage: memory keys only */
  }
  return [...out];
}

/**
 * All of this app's progress as one JSON-ready object. Backups are left out.
 * @param {{ app?: string, storage?: Storage | null, now?: Date }} [opts]
 */
export function exportProgress(opts = {}) {
  const app = opts.app ?? appId();
  const storage = opts.storage === undefined ? defaultStorage() : opts.storage;
  /** @type {Record<string, Envelope>} */
  const entries = {};
  for (const k of storageKeys(storage).sort()) {
    if (!k.startsWith(app + ":") || k.endsWith(BAK)) continue;
    const env = parseEnvelope(readRaw(storage, k));
    if (env) entries[k] = env;
  }
  return {
    format: EXPORT_FORMAT,
    formatVersion: EXPORT_FORMAT_VERSION,
    app,
    exported: (opts.now ?? new Date()).toISOString(),
    entries,
  };
}

/**
 * Validates an export, then writes every entry, each with its backup. Nothing
 * is written when any part is invalid. Throws an Error that names the fault.
 * @param {unknown} input the parsed object or the file's text
 * @param {{ app?: string, storage?: Storage | null }} [opts]
 * @returns {number} the number of entries written
 */
export function importProgress(input, opts = {}) {
  const app = opts.app ?? appId();
  const storage = opts.storage === undefined ? defaultStorage() : opts.storage;
  let obj = input;
  if (typeof input === "string") {
    try {
      obj = JSON.parse(input);
    } catch {
      throw new Error("The file is not JSON.");
    }
  }
  const o = /** @type {Record<string, any>} */ (obj);
  if (!o || typeof o !== "object" || o.format !== EXPORT_FORMAT) throw new Error("The file is not a learnkit progress export.");
  if (o.formatVersion !== EXPORT_FORMAT_VERSION) throw new Error(`Unknown export format version ${o.formatVersion}.`);
  if (o.app !== app) throw new Error(`The file belongs to app "${o.app}", not "${app}".`);
  if (!o.entries || typeof o.entries !== "object" || Array.isArray(o.entries)) throw new Error("The file has no entries.");
  /** @type {[string, Envelope, number][]} */
  const checked = [];
  for (const [k, v] of Object.entries(o.entries)) {
    if (!k.startsWith(app + ":") || k.endsWith(BAK)) throw new Error(`Entry "${k}" is outside this app.`);
    const env = parseEnvelope(v);
    if (!env) throw new Error(`Entry "${k}" is malformed.`);
    const schema = schemaFor(k.slice(app.length + 1));
    if (schema && (env.v > schema.version || (env.v === schema.version && !schema.validate(env.data)))) {
      throw new Error(`Entry "${k}" does not have the expected shape.`);
    }
    const max = schema?.version ?? Infinity;
    const stored = [parseEnvelope(readRaw(storage, k)), parseEnvelope(readRaw(storage, k + BAK))];
    if (stored.some((e) => e && e.v > max)) throw new Error(`Entry "${k}" would replace data from a newer version of the app.`);
    checked.push([k, env, max]);
  }
  for (const [k, env, max] of checked) writeWithBackup(storage, k, env, max);
  return checked.length;
}

/**
 * Saves the export as a file through a download link. Browser only.
 * @param {{ app?: string }} [opts]
 */
export function downloadProgress(opts = {}) {
  const data = exportProgress(opts);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${data.app}-progress-${data.exported.slice(0, 10)}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Reads a chosen file and imports it. Rejects with the validation message.
 * @param {File} file
 * @param {{ app?: string }} [opts]
 * @returns {Promise<number>}
 */
export async function importProgressFile(file, opts = {}) {
  return importProgress(await file.text(), opts);
}

/** Test hook: forgets values held in memory after failed writes. */
export function _resetMemory() {
  mem.clear();
}
