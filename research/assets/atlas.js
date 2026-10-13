/* Hidden Patch · shared runtime
   Every page: <body data-chapter="slug" data-part="N">, <header class="topbar" id="topbar"></header>,
   then load this file before the page's own script. Exposes window.Atlas. */
(function(){
"use strict";

const PARTS = [
  {n:1, name:"Predict",  blurb:"Learning by guessing what is hidden"},
  {n:2, name:"Theory",   blurb:"What training really does, in closed form"},
  {n:3, name:"Machines", blurb:"Reasoning, world models, small and fast"},
  {n:4, name:"Life",     blurb:"Emergence, open-endedness, search"},
  {n:5, name:"Map",      blurb:"People, fields, and what to try next"},
];

/* The table of contents. Order = suggested reading path. */
const CHAPTERS = [
  {slug:"predict",     part:1, title:"Why does guessing teach anything?",                 short:"Prediction as a teacher"},
  {slug:"jepa",        part:1, title:"Why predict in latent space?",                      short:"JEPA, piece by piece"},
  {slug:"collapse",    part:1, title:"Why doesn't everything collapse to one point?",     short:"Collapse and its cures"},
  {slug:"bias",        part:2, title:"Why does JEPA ignore the noise?",                   short:"Implicit bias and depth"},
  {slug:"gaussian",    part:2, title:"Why an isotropic Gaussian?",                        short:"LeJEPA, SIGReg, free energy"},
  {slug:"generalize",  part:2, title:"Why do networks generalize at all?",                short:"Double descent, grokking, simplicity"},
  {slug:"fractured",   part:2, title:"Why can the same answer hide a broken inside?",     short:"FER, convergence, the path matters"},
  {slug:"interp",      part:2, title:"Why can we read anything inside a network?",        short:"Superposition, SAEs, circuits"},
  {slug:"reasoning",   part:3, title:"Why does search plus a verifier keep winning?",     short:"LLMs, RL, test-time compute, ARC"},
  {slug:"architectures",part:3,title:"Why do transformers fail at counting?",             short:"Architecture, memory, continual learning"},
  {slug:"worlds",      part:3, title:"Why do agents need a world model?",                 short:"Planning in imagination"},
  {slug:"edge",        part:3, title:"Why could a JEPA fit in your pocket?",              short:"Quantization, latency, small models"},
  {slug:"alife",       part:4, title:"Why does life grow from simple rules?",             short:"Cellular automata, Lenia, replicators"},
  {slug:"open",        part:4, title:"Why can't greatness be planned?",                   short:"Novelty, open-endedness, AI scientists"},
  {slug:"landscape",   part:5, title:"Who is building what, and why do they disagree?",   short:"People, lineages, fault lines"},
  {slug:"fields",      part:5, title:"Why do 23 fields belong on one map?",               short:"The AI and artificial life map"},
  {slug:"lab",         part:5, title:"What can we test on a laptop tonight?",             short:"Experiments and notebooks"},
  {slug:"glossary",    part:5, title:"Glossary",                                          short:"Every term, one line each"},
];

const store = {
  get(k, d){ try{ const v = localStorage.getItem("hiddenpatch:"+k); return v==null ? d : JSON.parse(v); }catch(e){ return d; } },
  set(k, v){ try{ localStorage.setItem("hiddenpatch:"+k, JSON.stringify(v)); }catch(e){} },
};

const reduced = (()=>{ try{ return matchMedia("(prefers-reduced-motion: reduce)").matches; }catch(e){ return false; } })();

function el(tag, attrs, ...kids){
  const n = document.createElement(tag);
  if(attrs) for(const k in attrs){
    const v = attrs[k];
    if(k==="class") n.className = v;
    else if(k==="html") n.innerHTML = v;
    else if(k.startsWith("on") && typeof v==="function") n.addEventListener(k.slice(2), v);
    else if(v!==false && v!=null) n.setAttribute(k, v===true ? "" : v);
  }
  for(const c of kids.flat()) if(c!=null) n.append(c.nodeType ? c : document.createTextNode(c));
  return n;
}

const LOGO = `<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><defs><pattern id="hz" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="1.4" height="4" fill="currentColor"/></pattern></defs>
<rect x="2" y="2" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><rect x="18" y="2" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
<rect x="2" y="18" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><rect x="18" y="18" width="12" height="12" rx="2" fill="url(#hz)" stroke="var(--why)" stroke-width="2" style="color:var(--why)"/></svg>`;

function chapterIndex(slug){ return CHAPTERS.findIndex(c=>c.slug===slug); }

function buildTopbar(){
  const bar = document.getElementById("topbar");
  if(!bar) return;
  const slug = document.body.dataset.chapter || "";
  const i = chapterIndex(slug);
  bar.className = "topbar";
  bar.innerHTML = "";
  bar.append(el("a",{class:"brand",href:"index.html",html:LOGO+"<span><b>Hidden</b> Patch</span>"}));
  const crumb = el("span",{class:"crumb"});
  if(i>=0){ const c = CHAPTERS[i]; crumb.innerHTML = `<i>Part ${c.part} · ${PARTS[c.part-1].name}</i> · ${String(i+1).padStart(2,"0")} ${c.short}`; }
  bar.append(crumb);
  const themeBtn = el("button",{class:"tb-btn",type:"button","aria-label":"Switch light or dark theme",title:"Theme"},"Theme");
  themeBtn.addEventListener("click",()=>{
    const root = document.documentElement;
    const cur = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark":"light");
    root.dataset.theme = cur==="dark" ? "light" : "dark";
    store.set("theme", root.dataset.theme);
    window.dispatchEvent(new Event("atlas:theme"));
  });
  const menuBtn = el("button",{class:"tb-btn",type:"button","aria-haspopup":"dialog"},"Chapters");
  menuBtn.addEventListener("click",openDrawer);
  const courseLink = el("a",{class:"tb-btn",href:"../index.html",title:"Back to the ML course"},"Course");
  bar.append(courseLink, themeBtn, menuBtn);
  const pb = el("span",{class:"progressbar"}); bar.append(pb);
  addEventListener("scroll",()=>{
    const h = document.documentElement; const max = h.scrollHeight - h.clientHeight;
    pb.style.width = (max>0 ? (h.scrollTop/max*100) : 0) + "%";
  },{passive:true});
}

function openDrawer(){
  let d = document.getElementById("atlas-drawer");
  if(!d){
    const slug = document.body.dataset.chapter || "";
    const visited = store.get("visited",{}); const quiz = store.get("quiz",{});
    const sheet = el("div",{class:"sheet",role:"dialog","aria-modal":"true","aria-label":"Chapters"});
    const close = el("button",{class:"tb-btn close",type:"button"},"Close");
    close.addEventListener("click",()=>{ d.hidden = true; });
    sheet.append(close, el("a",{class:"brand",href:"index.html",html:LOGO+"<span><b>Hidden</b> Patch</span>"}));
    PARTS.forEach(p=>{
      sheet.append(el("h3",{html:`<span class="sw" style="background:var(--p${p.n})"></span>Part ${p.n} · ${p.name}`}));
      CHAPTERS.forEach((c,i)=>{ if(c.part!==p.n) return;
        const q = quiz[c.slug];
        const a = el("a",{class:"ch",href:c.slug+".html","aria-current":c.slug===slug?"page":false},
          el("span",{class:"n"},String(i+1).padStart(2,"0")), el("span",null,c.title),
          el("span",{class:"done"}, q ? `${q.right}/${q.total}` : (visited[c.slug] ? "seen" : "")));
        sheet.append(a);
      });
    });
    d = el("div",{class:"drawer",id:"atlas-drawer"}, sheet);
    d.addEventListener("click",e=>{ if(e.target===d) d.hidden = true; });
    addEventListener("keydown",e=>{ if(e.key==="Escape") d.hidden = true; });
    document.body.append(d);
  }
  d.hidden = false;
  const first = d.querySelector('[aria-current="page"]') || d.querySelector("a.ch");
  if(first) first.focus();
}

function buildPager(){
  const slot = document.getElementById("pager");
  if(!slot) return;
  const i = chapterIndex(document.body.dataset.chapter);
  if(i<0) return;
  slot.className = "pager";
  const prev = CHAPTERS[i-1], next = CHAPTERS[i+1];
  slot.append(prev ? el("a",{href:prev.slug+".html",class:"prev"},el("span",{class:"k"},"← Previous"),el("b",null,prev.title))
                   : el("a",{href:"index.html",class:"prev"},el("span",{class:"k"},"← Map"),el("b",null,"Back to the map")));
  slot.append(next ? el("a",{href:next.slug+".html",class:"next"},el("span",{class:"k"},"Next →"),el("b",null,next.title))
                   : el("a",{href:"index.html",class:"next"},el("span",{class:"k"},"Done →"),el("b",null,"Back to the map")));
}

function buildFooter(){
  const f = document.getElementById("site-footer");
  if(!f) return;
  f.className = "site";
  f.innerHTML = `<div class="wrap"><span><b>Hidden Patch</b> · built from the JEPA explorations project, October 2026.</span>
  <span>Companion to <a href="../index.html">Build ML From Scratch</a> · claims marked <span class="pill bad">check</span> were seen only in search results.</span></div>`;
}

/* ---------- quizzes ----------
   <section class="quiz" id="quiz"></section> and Atlas.quiz("quiz", [{q, options:[...], answer: idx, why}]) */
function quiz(id, items){
  const root = typeof id==="string" ? document.getElementById(id) : id;
  if(!root) return;
  const slug = document.body.dataset.chapter || "page";
  root.classList.add("quiz");
  root.innerHTML = "";
  const head = el("div",null);
  head.append(el("h2",{html:`Check yourself <span class="w">why</span>, not what`}));
  const score = el("p",{class:"score"}); head.append(score);
  root.append(head);
  let right = 0, done = 0;
  const upd = ()=>{ score.textContent = `${done}/${items.length} answered · ${right} right`;
    if(done===items.length){ const q = store.get("quiz",{}); q[slug] = {right,total:items.length}; store.set("quiz",q); } };
  items.forEach((it,k)=>{
    const card = el("div",{class:"qcard"});
    card.append(el("span",{class:"qn"},`Q${k+1}`), el("p",{class:"qt",html:it.q}));
    const opts = el("div",{class:"opts"});
    const ans = el("p",{class:"why-ans",html:it.why}); ans.hidden = true;
    it.options.forEach((o,j)=>{
      const b = el("button",{class:"opt",type:"button",html:o});
      b.addEventListener("click",()=>{
        opts.querySelectorAll("button").forEach((x,xi)=>{ x.disabled = true; if(xi===it.answer) x.classList.add("right"); });
        if(j!==it.answer) b.classList.add("wrong"); else right++;
        done++; ans.hidden = false; upd();
      });
      opts.append(b);
    });
    card.append(opts, ans); root.append(card);
  });
  upd();
}

/* ---------- canvas helpers ---------- */
/* Atlas.canvas(canvasEl, aspect, draw) → {ctx, w, h, redraw}. aspect = h/w, a number or a function of width (px).
   Handles DPR + resize + theme change. */
function canvas(c, aspect, draw){
  const ctx = c.getContext("2d");
  const api = {ctx, w:0, h:0, redraw(){ fit(); draw && draw(api); }};
  function fit(){
    const r = c.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio||1, 2.5);
    const w = Math.max(10, r.width), h = w*(typeof aspect==="function" ? aspect(w) : aspect);
    c.style.height = h+"px";
    if(c.width !== Math.round(w*dpr) || c.height !== Math.round(h*dpr)){ c.width = Math.round(w*dpr); c.height = Math.round(h*dpr); }
    ctx.setTransform(dpr,0,0,dpr,0,0);
    api.w = w; api.h = h;
  }
  fit();
  let raf = 0;
  const ro = new ResizeObserver(()=>{ cancelAnimationFrame(raf); raf = requestAnimationFrame(api.redraw); });
  ro.observe(c);
  addEventListener("atlas:theme", api.redraw);
  try{ matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>setTimeout(api.redraw,30)); }catch(e){}
  new MutationObserver(()=>setTimeout(api.redraw,30)).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  return api;
}

/* read colour tokens at draw time so canvases follow the theme */
function colors(){
  const s = getComputedStyle(document.body);
  const g = n => s.getPropertyValue(n).trim();
  return {bg:g("--bg"), panel:g("--panel"), panel2:g("--panel-2"), ink:g("--ink"), muted:g("--muted"), faint:g("--faint"),
    line:g("--line"), lineStrong:g("--line-strong"), why:g("--why"), part:g("--part"), hatch:g("--hatch"),
    p1:g("--p1"), p2:g("--p2"), p3:g("--p3"), p4:g("--p4"), p5:g("--p5"), good:g("--good"), bad:g("--bad"),
    mono:g("--mono"), ui:g("--ui")};
}

/* hatch fill for "masked / hidden" regions */
function hatch(ctx, x, y, w, h, color, gap=6){
  ctx.save(); ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath();
  for(let t=-h; t<w; t+=gap){ ctx.moveTo(x+t, y+h); ctx.lineTo(x+t+h, y); }
  ctx.stroke(); ctx.restore();
}

/* animation loop that pauses off-screen and when the tab is hidden; fn(dt, t) returns false to stop */
function loop(target, fn){
  let on = false, vis = true, last = 0, raf = 0, t = 0;
  const tick = now => {
    if(!on || !vis || document.hidden){ raf = 0; return; }
    const dt = Math.min(.05, last ? (now-last)/1000 : 0); last = now; t += dt;
    if(fn(dt,t) === false){ on = false; raf = 0; return; }
    raf = requestAnimationFrame(tick);
  };
  const io = new IntersectionObserver(es=>{ vis = es[0].isIntersecting; if(vis && on && !raf){ last = 0; raf = requestAnimationFrame(tick); } });
  io.observe(target);
  document.addEventListener("visibilitychange",()=>{ if(!document.hidden && on && !raf){ last=0; raf = requestAnimationFrame(tick); } });
  return {
    start(){ if(on) return; on = true; last = 0; if(!raf) raf = requestAnimationFrame(tick); },
    stop(){ on = false; },
    get running(){ return on; },
  };
}

/* seeded RNG + gaussian */
function rng(seed=1){
  let s = seed>>>0 || 1;
  const r = ()=>{ s ^= s<<13; s ^= s>>>17; s ^= s<<5; return ((s>>>0)/4294967296); };
  r.normal = ()=>{ let u=0,v=0; while(!u) u=r(); while(!v) v=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  return r;
}

/* bind a range input to an <output> and a callback: Atlas.slider("id", v=>..., fmt) */
function slider(id, on, fmt=v=>v){
  const inp = document.getElementById(id); if(!inp) return null;
  const out = document.querySelector(`output[for="${id}"]`);
  const upd = ()=>{ const v = parseFloat(inp.value); if(out) out.textContent = fmt(v); on && on(v); };
  inp.addEventListener("input", upd); upd();
  return inp;
}

/* light reveal on scroll; content is visible without JS */
function reveal(){
  if(reduced || !("IntersectionObserver" in window)) return;
  const items = [...document.querySelectorAll(".beat, .lab, .remember")];
  const io = new IntersectionObserver(es=>es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.remove("pre"); io.unobserve(e.target); } }),{rootMargin:"0px 0px -8% 0px"});
  items.forEach(n=>{ const r = n.getBoundingClientRect(); if(r.top > innerHeight){ n.classList.add("rv","pre"); io.observe(n); } });
}

function markVisited(){
  const slug = document.body.dataset.chapter; if(!slug) return;
  const v = store.get("visited",{}); v[slug] = Date.now(); store.set("visited", v);
}

function init(){
  const t = store.get("theme", null); if(t) document.documentElement.dataset.theme = t;
  buildTopbar(); buildPager(); buildFooter(); markVisited(); reveal();
}

window.Atlas = {PARTS, CHAPTERS, store, reduced, el, quiz, canvas, colors, hatch, loop, rng, slider, chapterIndex};
if(document.readyState==="loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
