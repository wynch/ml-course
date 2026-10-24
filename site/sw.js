// Every URL here is relative to this script, so the reader caches correctly
// whether it is served from a domain root or from /ml-course/reader/.
const CACHE_PREFIX = "ml-course-reader-v";
const CACHE = `${CACHE_PREFIX}16`;
const CORE = ["./", "./index.html", "./styles.css", "./app.js", "./course-content.js", "./manifest.webmanifest", "./offline-assets.json", "./favicon.svg", "./favicon.ico", "./apple-touch-icon.png"];
const HOME = new URL("./", self.location).href;
// The shell and the module list must match the latest publish whenever the
// network answers; the cache only stands in offline.
const FRESH = new Set(["./", "./index.html", "./app.js", "./course-content.js", "./styles.css", "./manifest.webmanifest", "./offline-assets.json"].map((path) => new URL(path, self.location).href));
const REVALIDATE = /\.(?:html|js|css|json)$/;

// Files go down a few at a time, in the order offline-assets.json lists them:
// the shell, then explorables, quizzes, research and moves, then lessons, then the
// rest. A phone that runs out of room or cuts the install short loses the tail
// of the list, never the visualizations.
const BATCH = 6;

// offline-assets.json is either a flat list (older builds) or groups in
// priority order.
function assetList(manifest) {
  if (Array.isArray(manifest)) return manifest;
  return (manifest.groups || []).flatMap((group) => group.urls);
}

async function warm(urls) {
  const cache = await caches.open(CACHE);
  const queue = [...new Set(urls)];
  for (let start = 0; start < queue.length; start += BATCH) {
    let full = false;
    await Promise.all(queue.slice(start, start + BATCH).map(async (url) => {
      try {
        const response = await fetch(url, { cache: "reload" });
        if (response.ok) await cache.put(url, response);
      } catch (error) {
        // One optional asset must not prevent the rest of the reader installing.
        if (error && error.name === "QuotaExceededError") full = true;
      }
    }));
    // Out of room: what is saved stays saved; the rest stays online-only.
    if (full) return;
  }
}

async function fetchAndStore(request, init) {
  // A navigate-mode Request cannot be rebuilt with a RequestInit, so it is
  // refetched by URL.
  const response = await fetch(request.mode === "navigate" ? request.url : request, init);
  // A navigation must not be answered with a followed redirect.
  if (request.mode === "navigate" && response.redirected) return Response.redirect(response.url, 302);
  if (response.ok) {
    const cache = await caches.open(CACHE);
    await cache.put(request.url, response.clone());
  }
  return response;
}

async function networkFirst(request) {
  try {
    return await fetchAndStore(request, { cache: "no-cache" });
  } catch {
    const cached = await caches.match(request.url);
    if (cached) return cached;
    if (request.mode === "navigate") return (await caches.match(HOME)) || Response.error();
    return Response.error();
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    await warm(CORE);
    try {
      const response = await fetch("./offline-assets.json", { cache: "no-cache" });
      if (response.ok) await warm(assetList(await response.json()));
    } finally {
      await self.skipWaiting();
    }
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    // The reader shares its origin (wynch.github.io) with other sites, so only
    // this reader's own older caches are deleted.
    const names = await caches.keys();
    await Promise.all(names
      .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const key = url.origin + url.pathname;
  if (request.mode === "navigate" || FRESH.has(key)) {
    event.respondWith(networkFirst(request));
    return;
  }
  event.respondWith((async () => {
    const cached = await caches.match(request);
    if (cached) {
      if (key.startsWith(HOME) && REVALIDATE.test(url.pathname)) {
        event.waitUntil(fetchAndStore(request, { cache: "no-cache" }).catch(() => {}));
      }
      return cached;
    }
    try {
      return await fetchAndStore(request);
    } catch {
      return Response.error();
    }
  })());
});
