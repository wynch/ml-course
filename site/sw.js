// Every URL here is relative to this script, so the reader caches correctly
// whether it is served from a domain root or from /ml-course/reader/.
const CACHE = "ml-course-reader-v5";
const CORE = ["./", "./styles.css", "./app.js", "./course-content.js", "./manifest.webmanifest", "./offline-assets.json"];
const HOME = new URL("./", self.location).href;

async function warm(urls) {
  const cache = await caches.open(CACHE);
  await Promise.all([...new Set(urls)].map(async (url) => {
    try {
      const response = await fetch(url, { cache: "reload" });
      if (response.ok) await cache.put(url, response);
    } catch {
      // One optional asset must not prevent the rest of the reader installing.
    }
  }));
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    await warm(CORE);
    try {
      const response = await fetch("./offline-assets.json");
      if (response.ok) await warm(await response.json());
    } finally {
      await self.skipWaiting();
    }
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((name) => name !== CACHE).map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cached = await caches.match(event.request);
    if (cached) return cached;
    try {
      const response = await fetch(event.request);
      if (response.ok) {
        const cache = await caches.open(CACHE);
        await cache.put(event.request, response.clone());
      }
      return response;
    } catch {
      if (event.request.mode === "navigate") return (await caches.match(HOME)) || Response.error();
      return Response.error();
    }
  })());
});
