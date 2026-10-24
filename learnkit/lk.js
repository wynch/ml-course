// Entry point: <script type="module" src="learnkit/lk.js"></script>.
// Defines every element, adds lk.css when the page has not linked it, and
// starts the update flow (a no-op before a release build).

import "./lk-quiz.js";
import "./lk-deck.js";
import "./lk-predict.js";
import "./lk-stepper.js";
import { mountUpdates } from "./lk-update.js";

export * as plot from "./lk-plot.js";
export { createStore, exportProgress, importProgress, downloadProgress, importProgressFile } from "./lk-store.js";
export { forceUpdate, checkForUpdate, toast } from "./lk-update.js";
export { appId, buildVersion } from "./lk-util.js";

if (typeof document !== "undefined") {
  const css = new URL("./lk.css", import.meta.url).href;
  if (![...document.querySelectorAll('link[rel="stylesheet"]')].some((l) => /** @type {HTMLLinkElement} */ (l).href === css)) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = css;
    document.head.append(link);
  }
  mountUpdates();
}
