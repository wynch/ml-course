(() => {
  "use strict";

  const DATA = window.ML_COURSE;
  const pages = [...DATA.guides, ...DATA.modules];
  const TRACKS = [
    { name: "Origins", blurb: "An optional prologue: the classic math still running inside every model here." },
    { name: "Foundations", blurb: "Gradients, networks, and the tokenizer that feeds them." },
    { name: "Transformers & LLMs", blurb: "Build the block, evaluate it, adapt it, and run it fast." },
    { name: "Breadth", blurb: "Carry the same ideas into pixels, denoising, and action." },
  ];
  const ASSET_ROOTS = ["explorables/", "quizzes/"];
  const storageKey = "ml-course-progress-v1";
  const emptyProgress = { completed: [], answers: {}, lastVisited: "01-autograd", updatedAt: new Date(0).toISOString() };
  let progress = loadProgress();
  let installPrompt = null;

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function slugify(value) {
    return value.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-");
  }

  function normalizePath(path) {
    const parts = [];
    path.split("/").forEach((part) => {
      if (!part || part === ".") return;
      if (part === "..") parts.pop();
      else parts.push(part);
    });
    return parts.join("/");
  }

  function resolveLocalPath(documentPath, href) {
    const clean = href.split("#")[0].split("?")[0];
    if (clean.startsWith("/")) return normalizePath(clean);
    const base = documentPath.split("/").slice(0, -1).join("/");
    return normalizePath(`${base}/${clean}`);
  }

  function pageForPath(path) {
    return pages.find((page) => page.path === path || page.path === `${path.replace(/\/$/, "")}/README.md`);
  }

  function inline(value, page) {
    let text = escapeHtml(value);
    const tokens = [];
    const hold = (html) => {
      const key = `\u0000${tokens.length}\u0000`;
      tokens.push(html);
      return key;
    };

    text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, alt, src) => {
      const source = /^(https?:|data:|\/)/.test(src) ? src : `/content/${resolveLocalPath(page.path, src)}`;
      return hold(`<img src="${escapeHtml(source)}" alt="${escapeHtml(alt)}" loading="lazy">`);
    });
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, rawHref) => {
      const href = rawHref.replace(/^&lt;|&gt;$/g, "");
      if (/^(https?:|mailto:)/.test(href)) {
        return hold(`<a href="${href}" target="_blank" rel="noreferrer">${label}</a>`);
      }
      if (href.startsWith("#")) {
        return hold(`<a href="${href}" data-section="${escapeHtml(href.slice(1))}">${label}</a>`);
      }
      const [pathPart, anchor = ""] = href.split("#");
      const resolved = resolveLocalPath(page.path, pathPart);
      const targetPage = pageForPath(resolved);
      if (targetPage) {
        return hold(`<a href="#/${targetPage.id}" data-route="${targetPage.id}" data-anchor="${escapeHtml(anchor)}">${label}</a>`);
      }
      const target = ASSET_ROOTS.some((root) => resolved.startsWith(root)) ? `/${resolved}` : `/content/${resolved}`;
      return hold(`<a href="${escapeHtml(target)}">${label}</a>`);
    });
    text = text.replace(/&lt;(https?:\/\/[^&\s]+)&gt;/g, (_, href) =>
      hold(`<a href="${href}" target="_blank" rel="noreferrer">${href}</a>`));
    text = text.replace(/`([^`]+)`/g, (_, code) => hold(`<code>${code}</code>`));
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    text = text.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
    text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    tokens.forEach((token, index) => {
      text = text.replaceAll(`\u0000${index}\u0000`, token);
    });
    return text;
  }

  function isBlockStart(line, next = "") {
    return /^(```|#{1,6}\s|>\s?|[-*_]{3,}\s*$|[-*+]\s+|\d+\.\s+)/.test(line)
      || (/^\|?.+\|.+/.test(line) && /^\|?\s*:?-+/.test(next));
  }

  function splitCells(line) {
    return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  }

  function renderMarkdown(markdown, page) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index += 1; continue; }

      const fence = line.match(/^```(.*)$/);
      if (fence) {
        const language = fence[1].trim();
        const code = [];
        index += 1;
        while (index < lines.length && !lines[index].startsWith("```")) code.push(lines[index++]);
        index += 1;
        if (language === "mermaid") {
          out.push(`<div class="mermaid-source"><span>Flow diagram · source preserved offline</span><pre><code>${escapeHtml(code.join("\n"))}</code></pre></div>`);
        } else {
          out.push(`<pre><code class="language-${escapeHtml(language)}">${escapeHtml(code.join("\n"))}</code></pre>`);
        }
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        const content = inline(heading[2], page);
        out.push(`<h${level} id="${slugify(heading[2])}">${content}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^[-*_]{3,}\s*$/.test(line)) {
        out.push("<hr>");
        index += 1;
        continue;
      }

      if (line.startsWith(">")) {
        const quote = [];
        while (index < lines.length && lines[index].startsWith(">")) {
          quote.push(lines[index++].replace(/^>\s?/, ""));
        }
        out.push(`<blockquote><p>${inline(quote.join(" "), page)}</p></blockquote>`);
        continue;
      }

      if (/^\|?.+\|.+/.test(line) && index + 1 < lines.length && /^\|?\s*:?-+/.test(lines[index + 1])) {
        const headers = splitCells(line);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) rows.push(splitCells(lines[index++]));
        out.push(`<div class="table-scroll"><table><thead><tr>${headers.map((cell) => `<th>${inline(cell, page)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inline(cell, page)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }

      const list = line.match(/^(\s*)([-*+]|\d+\.)\s+(.+)$/);
      if (list) {
        const ordered = /\d+\./.test(list[2]);
        const tag = ordered ? "ol" : "ul";
        const items = [];
        while (index < lines.length) {
          const item = lines[index].match(/^(\s*)([-*+]|\d+\.)\s+(.+)$/);
          if (!item || /\d+\./.test(item[2]) !== ordered) break;
          let content = item[3];
          const task = content.match(/^\[([ xX])\]\s+(.+)$/);
          if (task) {
            const checked = task[1].trim() ? " checked" : "";
            items.push(`<li class="task-item"><input type="checkbox" disabled${checked}> ${inline(task[2], page)}</li>`);
          } else {
            items.push(`<li>${inline(content, page)}</li>`);
          }
          index += 1;
        }
        out.push(`<${tag}>${items.join("")}</${tag}>`);
        continue;
      }

      if (/^<!--/.test(line)) {
        while (index < lines.length && !lines[index++].includes("-->")) {}
        continue;
      }

      const paragraph = [line.trim()];
      index += 1;
      while (
        index < lines.length &&
        lines[index].trim() &&
        !isBlockStart(lines[index], lines[index + 1] || "")
      ) paragraph.push(lines[index++].trim());
      out.push(`<p>${inline(paragraph.join(" "), page)}</p>`);
    }
    return out.join("\n");
  }

  function loadProgress() {
    try {
      return { ...emptyProgress, ...JSON.parse(localStorage.getItem(storageKey) || "{}") };
    } catch {
      localStorage.removeItem(storageKey);
      return { ...emptyProgress };
    }
  }

  function saveProgress() {
    progress.updatedAt = new Date().toISOString();
    localStorage.setItem(storageKey, JSON.stringify(progress));
  }

  function route() {
    const id = location.hash.replace(/^#\/?/, "").split("?")[0];
    return pages.some((page) => page.id === id) ? id : "home";
  }

  function choose(id) {
    if (id !== "home") {
      progress.lastVisited = id;
      saveProgress();
    }
    location.hash = id === "home" ? "#/" : `#/${id}`;
  }

  function formatMinutes(minutes = 0) {
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return hours ? `${hours}h${rest ? ` ${rest}m` : ""}` : `${rest} min`;
  }

  function moduleNumber(id, compact = false) {
    if (id.startsWith("05a")) return compact ? "½" : "05½";
    const match = id.match(/^\d+[a-z]?/);
    return match ? match[0] : id.slice(0, 2);
  }

  function sidebar(active) {
    const tracks = TRACKS.map((track) => track.name);
    return `<aside class="sidebar">
      <button class="brand" type="button" data-route="home">
        <span class="brand-shape" aria-hidden="true"><i></i><i></i><i></i></span>
        <span>Build ML<br><small>from scratch</small></span>
      </button>
      <nav aria-label="Course modules">${tracks.map((track) => `
        <section class="nav-track"><h2>${track}</h2>
          ${DATA.modules.filter((page) => page.track === track).map((page) => `
            <button type="button" class="${active === page.id ? "active" : ""} accent-${page.accent}" data-route="${page.id}" ${active === page.id ? 'aria-current="page"' : ""}>
              <span class="nav-index">${moduleNumber(page.id, true)}</span>
              <span>${escapeHtml(page.shortTitle)}</span>
              <span class="completion-dot ${progress.completed.includes(page.id) ? "done" : ""}">${progress.completed.includes(page.id) ? "✓" : ""}</span>
            </button>`).join("")}
        </section>`).join("")}</nav>
      <div class="sidebar-foot"><button data-route="offline">Offline packs</button><button data-route="resources">Resources</button></div>
    </aside>`;
  }

  function topbar() {
    const ready = "serviceWorker" in navigator;
    return `<header class="topbar">
      <button class="nav-toggle" type="button" aria-label="Toggle course navigation"><span></span><span></span><span></span></button>
      <div class="search-wrap">
        <label class="sr-only" for="course-search">Search the course</label>
        <span class="search-mark" aria-hidden="true">⌕</span>
        <input id="course-search" placeholder="Search concepts, commands, exercises…" autocomplete="off">
        <div class="search-results" hidden></div>
      </div>
      <div class="offline-status ${navigator.onLine ? "" : "is-offline"}"><i></i><span>${navigator.onLine ? (ready ? "Offline ready" : "Reader mode") : "Working offline"}</span></div>
      ${installPrompt ? '<button class="install-button" type="button">Install</button>' : ""}
      <details class="progress-menu"><summary aria-label="Progress options">${progress.completed.length}/${DATA.modules.length}</summary>
        <div><strong>Local progress</strong><p>Move progress between devices without an account.</p>
          <button type="button" id="export-progress">Export JSON</button>
          <button type="button" id="import-progress">Import JSON</button>
          <input id="progress-file" type="file" accept="application/json" hidden>
        </div>
      </details>
    </header>`;
  }

  function home() {
    const completed = progress.completed.length;
    const completion = Math.round(completed / DATA.modules.length * 100);
    return `<main class="home-page">
      <section class="home-hero">
        <div class="eyebrow">A graphical, build-it-twice machine-learning course</div>
        <h1>First, make it work.<br><em>Then see why.</em></h1>
        <p>From one scalar gradient to a local tool-using agent. Build the machinery in Python and Zig, then meet the production ecosystem—with every important shape, trade-off, and failure mode made visible.</p>
        <div class="hero-actions">
          <button class="primary-button" data-route="${progress.lastVisited || "01-autograd"}">${completed ? "Continue learning" : "Start with autograd"} <span>→</span></button>
          <button class="secondary-button" data-route="course">Read the course guide</button>
        </div>
        <div class="course-facts"><span><b>${DATA.modules.length}</b> hands-on labs</span><span><b>${DATA.modules.length}</b> browser explorables</span><span><b>2</b> implementation languages</span><span><b>0</b> runtime network calls</span></div>
      </section>
      <section class="progress-strip"><div><span>Your path</span><strong>${completed ? `${completed} of ${DATA.modules.length} labs complete` : "Ready when you are"}</strong></div><div class="progress-track"><i style="width:${completion}%"></i></div><b>${completion}%</b></section>
      ${TRACKS.map(({ name, blurb }, trackIndex) => `<section class="module-track track-${trackIndex}">
        <div class="track-heading"><span>0${trackIndex}</span><div><h2>${name}</h2><p>${blurb}</p></div></div>
        <div class="module-grid">${DATA.modules.filter((page) => page.track === name).map((page) => {
          const done = progress.completed.includes(page.id);
          return `<button class="module-card accent-${page.accent}" data-route="${page.id}">
            <span class="module-number">${moduleNumber(page.id)}</span>
            <span class="module-card-title">${escapeHtml(page.shortTitle)}</span>
            <span class="module-meta">${formatMinutes(page.minutes)} · lab + explorable</span>
            <span class="card-status ${done ? "done" : ""}">${done ? "Completed ✓" : "Open lab →"}</span>
          </button>`;
        }).join("")}</div></section>`).join("")}
      <section class="offline-promise"><div class="offline-glyph" aria-hidden="true">↯</div><div><span class="eyebrow">Designed for the train, plane, and bad Wi-Fi</span><h2>The reading layer fits in your pocket.</h2><p>Every lesson, figure, source file, and interactive explainer is packaged locally. Optional core and full lab packs add the runtimes, datasets, and model weights you choose.</p></div><button class="secondary-button" data-route="offline">Set up offline use</button></section>
    </main>`;
  }

  function quiz(page) {
    const questions = DATA.quizzes[page.id];
    if (!questions) return "";
    const answers = progress.answers[page.id] || {};
    const score = questions.filter((q) => answers[q.id] === q.correct).length;
    return `<section class="knowledge-check">
      <div class="knowledge-heading"><div><span class="eyebrow">Retrieval practice</span><h2>Check the mental model</h2></div><strong>${score}/${questions.length}</strong></div>
      ${questions.map((question, questionIndex) => {
        const selected = answers[question.id];
        const answered = selected !== undefined;
        return `<fieldset><legend><span>${questionIndex + 1}</span>${escapeHtml(question.prompt)}</legend>
          <div class="choice-grid">${question.choices.map((choice, choiceIndex) => `<button type="button" data-question="${question.id}" data-choice="${choiceIndex}" class="${selected === choiceIndex ? "selected" : ""} ${answered && choiceIndex === question.correct ? "correct" : ""}"><span>${String.fromCharCode(65 + choiceIndex)}</span>${escapeHtml(choice)}</button>`).join("")}</div>
          ${answered ? `<p class="${selected === question.correct ? "answer-correct" : "answer-wrong"}"><strong>${selected === question.correct ? "Exactly." : "Not quite."}</strong> ${escapeHtml(question.explanation)}</p>` : ""}
        </fieldset>`;
      }).join("")}
    </section>`;
  }

  function lesson(page) {
    const index = DATA.modules.findIndex((candidate) => candidate.id === page.id);
    const previous = index > 0 ? DATA.modules[index - 1] : null;
    const next = index >= 0 && index < DATA.modules.length - 1 ? DATA.modules[index + 1] : null;
    const done = progress.completed.includes(page.id);
    return `<main class="lesson-page">
      <header class="lesson-mast accent-${page.accent || "teal"}"><div><span class="eyebrow">${page.kind === "module" ? `${page.track} · ${formatMinutes(page.minutes)}` : "Course guide"}</span><h1>${escapeHtml(page.title)}</h1></div>${page.explorable ? `<a class="explorable-button" href="${page.explorable}">Open interactive explainer <span>↗</span></a>` : ""}</header>
      <article class="markdown-body">${renderMarkdown(page.markdown, page)}</article>
      ${page.kind === "module" ? `${quiz(page)}
        <section class="lesson-finish"><div><span class="eyebrow">Progress stays on this device</span><h2>${done ? "Lab marked complete." : "Finished this lab?"}</h2></div><button class="complete-button ${done ? "completed" : ""}" data-complete="${page.id}">${done ? "✓ Completed" : "Mark complete"}</button></section>
        <nav class="lesson-pagination">${previous ? `<button data-route="${previous.id}"><span>← Previous</span><strong>${escapeHtml(previous.shortTitle)}</strong></button>` : "<span></span>"}${next ? `<button data-route="${next.id}"><span>Next →</span><strong>${escapeHtml(next.shortTitle)}</strong></button>` : ""}</nav>` : ""}
    </main>`;
  }

  function footer() {
    return `<footer class="site-footer"><span>Build ML From Scratch</span><nav>${DATA.guides.slice(1, 5).map((page) => `<button data-route="${page.id}">${escapeHtml(page.shortTitle)}</button>`).join("")}</nav><span>Local-first · no account required</span></footer>`;
  }

  function search(query, target) {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) { target.hidden = true; target.innerHTML = ""; return; }
    const results = pages.map((page) => {
      const haystack = `${page.title}\n${page.markdown}`.toLowerCase();
      const matches = terms.reduce((sum, term) => sum + haystack.split(term).length - 1, 0);
      const first = Math.max(0, haystack.indexOf(terms[0]));
      const excerpt = page.markdown.slice(Math.max(0, first - 55), first + 170).replace(/[#*`_[\]()>|]/g, " ").replace(/\s+/g, " ").trim();
      return { page, matches, excerpt };
    }).filter((result) => result.matches > 0).sort((a, b) => b.matches - a.matches).slice(0, 8);
    target.hidden = false;
    target.innerHTML = `<div class="search-results-label">${results.length ? `${results.length} best matches` : "No matches"}</div>${results.map(({ page, excerpt }) => `<button type="button" data-route="${page.id}"><strong>${escapeHtml(page.title)}</strong><span>${escapeHtml(excerpt || "Open lesson")}</span></button>`).join("")}`;
    bindRoutes(target);
  }

  function bindRoutes(root = document) {
    root.querySelectorAll("[data-route]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        const id = button.dataset.route;
        choose(id);
        const anchor = button.dataset.anchor;
        if (anchor) setTimeout(() => document.getElementById(anchor)?.scrollIntoView(), 40);
      });
    });
    root.querySelectorAll("[data-section]").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        document.getElementById(link.dataset.section)?.scrollIntoView({ behavior: "smooth" });
      });
    });
  }

  function bind() {
    bindRoutes();
    const sidebarElement = document.querySelector(".sidebar");
    const toggle = document.querySelector(".nav-toggle");
    toggle?.addEventListener("click", () => sidebarElement?.classList.toggle("is-open"));
    const input = document.getElementById("course-search");
    const results = document.querySelector(".search-results");
    input?.addEventListener("input", () => search(input.value, results));

    document.querySelectorAll("[data-question]").forEach((button) => {
      button.addEventListener("click", () => {
        const current = route();
        progress.answers[current] ||= {};
        progress.answers[current][button.dataset.question] = Number(button.dataset.choice);
        saveProgress();
        render();
      });
    });
    document.querySelector("[data-complete]")?.addEventListener("click", (event) => {
      const id = event.currentTarget.dataset.complete;
      progress.completed = progress.completed.includes(id)
        ? progress.completed.filter((value) => value !== id)
        : [...progress.completed, id];
      saveProgress();
      render();
    });

    document.getElementById("export-progress")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(progress, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "ml-course-progress.json";
      anchor.click();
      URL.revokeObjectURL(url);
    });
    document.getElementById("import-progress")?.addEventListener("click", () => document.getElementById("progress-file")?.click());
    document.getElementById("progress-file")?.addEventListener("change", async (event) => {
      try {
        const parsed = JSON.parse(await event.target.files[0].text());
        if (!Array.isArray(parsed.completed) || typeof parsed.answers !== "object") throw new Error();
        progress = { ...emptyProgress, ...parsed, updatedAt: new Date().toISOString() };
        saveProgress();
        render();
      } catch {
        alert("That file is not a valid ML course progress export.");
      }
    });
    document.querySelector(".install-button")?.addEventListener("click", async () => {
      await installPrompt?.prompt();
      installPrompt = null;
      render();
    });
  }

  function render() {
    const active = route();
    const page = pages.find((candidate) => candidate.id === active);
    document.getElementById("app").innerHTML = `<div class="course-shell">${sidebar(active)}<div class="course-main">${topbar()}${page ? lesson(page) : home()}${footer()}</div></div>`;
    bind();
  }

  window.addEventListener("hashchange", () => { window.scrollTo(0, 0); render(); });
  window.addEventListener("online", render);
  window.addEventListener("offline", render);
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    installPrompt = event;
    render();
  });

  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
  render();
})();
