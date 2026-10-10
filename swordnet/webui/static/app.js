// SwordNet web UI. No build step, no framework.
const $ = (s) => document.querySelector(s);
// Who is labeling: sent with every request so the queue and the logs know.
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
let WHO = store.get("swordnet-labeler") || "";
function askName() {
  const n = (prompt("Your name (shown to other labelers and in the logs):", WHO) || "").trim();
  if (n) { WHO = n; store.set("swordnet-labeler", n); }
  $("#who").textContent = WHO ? `👤 ${WHO}` : "Set your name";
}
const api = async (path, opts = {}) => {
  opts.headers = { ...(opts.headers || {}), "X-Labeler": WHO || "anonymous" };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const t = r.headers.get("content-type") || "";
  return t.includes("json") ? r.json() : r.text();
};
const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let CLASSES = ["sword", "stick"];

// --- tabs --------------------------------------------------------------------
document.querySelectorAll("nav button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  ({ upload: loadDatasets, images: loadImages, annotate: loadQueue, reports: loadRuns, cluster: loadCluster, logs: loadLogs })[name]();
}

async function refreshStats() {
  const q = await api("/api/queue");
  const others = q.labelers.filter((n) => n !== WHO);
  $("#stats").textContent = `${q.ingest} to label · ${q.labeled} labeled` +
    (others.length ? ` · also labeling: ${others.join(", ")}` : "");
}

// --- upload ------------------------------------------------------------------
async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  fd.append("dataset", $("#up-dataset").value || "default");
  fd.append("every", $("#up-every").value || "0.5");
  $("#up-msg").textContent = `Uploading ${files.length} file(s)…`;
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    const v = r.videos.map((x) => `${x.video}: ${x.frames} frames`).join(", ");
    const d = r.duplicates.length ? ` Skipped ${r.duplicates.length} already in the dataset: ${r.duplicates.join(", ")}.` : "";
    const bad = r.skipped.length ? ` Could not read: ${r.skipped.join(", ")}.` : "";
    $("#up-msg").textContent = `Added ${r.added} image(s).${v ? " " + v : ""}${d}${bad}`;
  } catch (e) { $("#up-msg").textContent = e.message; }
  loadDatasets();
}
$("#up-files").onchange = (e) => uploadFiles(e.target.files);
const drop = $("#drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles(e.dataTransfer.files); };

$("#sc-go").onclick = async () => {
  const urls = $("#sc-urls").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!urls.length) return;
  $("#sc-msg").textContent = "Scraping…";
  try {
    const r = await api("/api/scrape", json("POST", { urls, dataset: $("#up-dataset").value || "default" }));
    $("#sc-msg").textContent = `Added ${r.added} image(s).`;
  } catch (e) { $("#sc-msg").textContent = e.message; }
  loadDatasets();
};

async function loadDatasets() {
  const ds = await api("/api/datasets");
  $("#ds-table tbody").innerHTML = ds.map((d) => `<tr><td>${esc(d.name)}</td><td>${d.n_ingest || 0}</td>
    <td>${d.n_labeled || 0}</td><td><input type="checkbox" data-ds="${esc(d.name)}" ${d.autolabel ? "checked" : ""}></td></tr>`).join("")
    || `<tr><td colspan="4" class="hint">No datasets yet. Upload something.</td></tr>`;
  document.querySelectorAll("[data-ds]").forEach((cb) => cb.onchange = () =>
    api(`/api/datasets/${encodeURIComponent(cb.dataset.ds)}/autolabel`, json("POST", { enabled: cb.checked })));
  refreshStats();
}

// --- annotate ----------------------------------------------------------------
const canvas = $("#an-canvas"), ctx = canvas.getContext("2d");
// Queue mode ("ingest"): the server hands out one claimed image at a time; Skip asks for another.
// Browse mode ("labeled"): a plain list with Prev/Next, for fixing finished labels.
// `sel` is the highlighted box (for Q/E/Del). `picked` is true only when the labeler clicked a
// box on purpose: only then do the class keys relabel it. A box you just drew is highlighted but
// not picked, so pressing 2 to draw a stick next cannot silently turn your sword into a stick.
const A = { queue: [], pos: 0, img: null, image: null, labels: [], sel: -1, picked: false, cls: 0, drag: null, dirty: false, skipped: [] };
const queueMode = () => $("#an-filter").value === "ingest";

function classButtons() {
  $("#an-classes").innerHTML = CLASSES.map((c, i) =>
    `<button class="cls ${i === A.cls ? "active" : ""}" data-cls="${i}" style="color:var(--${c}, inherit)">${i + 1} ${esc(c)}</button>`).join(" ");
  document.querySelectorAll("[data-cls]").forEach((b) => b.onclick = () => setClass(+b.dataset.cls));
}
function setClass(i) {
  A.cls = i;
  if (A.sel >= 0 && A.picked && A.labels[A.sel].cls !== i) {
    A.labels[A.sel].cls = i; A.labels[A.sel].source = touched(A.labels[A.sel].source); A.dirty = true;
  }
  classButtons(); draw();
}
// Editing an auto-label makes it a human label (the editor shows it as reviewed).
const touched = (src) => src.startsWith("auto") ? "manual" : src;

async function loadQueue() {
  if (!WHO) askName();
  ["#an-prev", "#an-next"].forEach((b) => $(b).style.display = queueMode() ? "none" : "");
  $("#an-skip").style.display = queueMode() ? "" : "none";
  if (queueMode()) {
    if (A.image && A.image.status === "ingest" && A.image.claimed_by === WHO) return openImage(A.image);
    const r = await api(`/api/queue/next?skip=${A.skipped.join(",")}`, { method: "POST" });
    A.queue = r.image ? [r.image] : []; A.pos = 0;
  } else {
    A.queue = await api("/api/images?status=labeled");
    A.pos = Math.min(A.pos, Math.max(0, A.queue.length - 1));
  }
  refreshStats();
  await openImage(A.queue[A.pos]);
}
$("#an-filter").onchange = () => { A.pos = 0; A.image = null; loadQueue(); };

async function openImage(item) {
  $("#an-empty").style.display = item ? "none" : "block";
  $("#an-empty").textContent = queueMode() ? "Queue is empty: nothing left to label (or everything is claimed)." : "Nothing labeled yet.";
  canvas.style.display = item ? "block" : "none";
  $("#an-pos").textContent = queueMode() ? "" : item ? `${A.pos + 1} / ${A.queue.length}` : "0 / 0";
  if (!item) { A.image = null; return; }
  const { image, labels } = await api(`/api/images/${item.id}/labels`);
  A.image = image; A.labels = labels.map((l) => ({ cls: l.cls, points: l.points, source: l.source, score: l.score }));
  A.sel = -1; A.picked = false; A.dirty = false;
  const img = new Image();
  img.onload = () => { A.img = img; canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(); };
  img.src = `/api/images/${item.id}/file?v=${image.status}`;
  const auto = A.labels.some((l) => l.source.startsWith("auto"));
  $("#an-info").textContent = `${image.filename} · ${image.origin} · dataset ${image.collection}` +
    (auto ? " · showing auto-label proposals: fix or delete them, then save" : "");
}

function draw() {
  if (!A.img) return;
  ctx.drawImage(A.img, 0, 0);
  const lw = Math.max(2, canvas.width / 400);
  A.labels.forEach((l, i) => {
    const color = getComputedStyle(document.documentElement).getPropertyValue(`--${CLASSES[l.cls]}`).trim() || "#f0f";
    ctx.beginPath();
    l.points.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.setLineDash(l.source.startsWith("auto") ? [lw * 4, lw * 3] : []);
    ctx.lineWidth = i === A.sel ? lw * 2 : lw;
    ctx.strokeStyle = i === A.sel ? "#f5c400" : color;
    ctx.stroke();
    ctx.setLineDash([]);
    const [x, y] = l.points[0];
    ctx.font = `${lw * 7}px system-ui`;
    ctx.fillStyle = color;
    const tag = CLASSES[l.cls] + (l.score != null && l.source.startsWith("auto") ? ` ${l.score.toFixed(2)}` : "");
    ctx.fillText(tag, x + lw, y - lw * 2);
    if (l.source.includes("sam")) { ctx.beginPath(); ctx.arc(x, y, lw * 2, 0, 7); ctx.fill(); }
  });
  if (A.drag && A.drag.box) {
    const { x0, y0, x1, y1 } = A.drag;
    ctx.setLineDash([]); ctx.lineWidth = lw; ctx.strokeStyle = "#fff";
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function toImg(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
}
function inside([x, y], pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
canvas.onmousedown = (e) => { const [x, y] = toImg(e); A.drag = { x0: x, y0: y, x1: x, y1: y, box: false }; };
canvas.onmousemove = (e) => {
  if (!A.drag) return;
  [A.drag.x1, A.drag.y1] = toImg(e);
  A.drag.box = Math.abs(A.drag.x1 - A.drag.x0) > 4 && Math.abs(A.drag.y1 - A.drag.y0) > 4;
  draw();
};
canvas.onmouseup = (e) => {
  const d = A.drag; A.drag = null;
  if (!d) return;
  if (d.box) {
    const [x0, x1] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)], [y0, y1] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
    A.labels.push({ cls: A.cls, points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], source: "manual", score: null });
    A.sel = A.labels.length - 1; A.picked = false; A.dirty = true;
  } else {
    // click: select the smallest box under the cursor
    const hits = A.labels.map((l, i) => [i, l]).filter(([, l]) => inside([d.x0, d.y0], l.points));
    const area = (p) => Math.abs(p.reduce((s, [x, y], i) => { const [u, v] = p[(i + 1) % 4]; return s + x * v - u * y; }, 0)) / 2;
    hits.sort((a, b) => area(a[1].points) - area(b[1].points));
    A.sel = hits.length ? hits[0][0] : -1;
    A.picked = A.sel >= 0;
    if (A.picked) { A.cls = A.labels[A.sel].cls; classButtons(); }
  }
  draw();
};

function rotate(deg) {
  if (A.sel < 0) return;
  const l = A.labels[A.sel], t = deg * Math.PI / 180;
  const cx = l.points.reduce((s, p) => s + p[0], 0) / 4, cy = l.points.reduce((s, p) => s + p[1], 0) / 4;
  l.points = l.points.map(([x, y]) => [cx + (x - cx) * Math.cos(t) - (y - cy) * Math.sin(t),
                                       cy + (x - cx) * Math.sin(t) + (y - cy) * Math.cos(t)]);
  l.source = touched(l.source); A.dirty = true; draw();
}

async function save() {
  if (!A.image) return;
  await api(`/api/images/${A.image.id}/labels`, json("PUT", { labels: A.labels }));
  A.dirty = false;
  $("#an-info").textContent = `Saved ${A.labels.length} box(es) on ${A.image.filename}`;
}
async function done() {
  if (!A.image) return;
  // Accepting an image confirms its boxes, including auto-label proposals the labeler kept.
  A.labels = A.labels.map((l) => ({ ...l, source: touched(l.source) }));
  await save();
  await api(`/api/images/${A.image.id}/processed`, { method: "POST" });
  A.image = null;
  if (queueMode()) return loadQueue();
  A.pos = Math.min(A.pos + 1, Math.max(0, A.queue.length - 1));
  refreshStats(); openImage(A.queue[A.pos]);
}
async function go(delta) {
  if (A.dirty && !confirm("Discard unsaved changes on this image?")) return;
  A.pos = Math.max(0, Math.min(A.queue.length - 1, A.pos + delta));
  openImage(A.queue[A.pos]);
}
$("#an-skip").onclick = async () => {
  if (!A.image) return;
  if (A.dirty && !confirm("Discard unsaved changes on this image?")) return;
  A.skipped.push(A.image.id);
  await api(`/api/images/${A.image.id}/release`, { method: "POST" });
  A.image = null; loadQueue();
};
$("#an-prev").onclick = () => go(-1);
$("#an-next").onclick = () => go(1);
$("#an-save").onclick = save;
$("#an-done").onclick = done;
$("#an-reject").onclick = async () => {
  if (!A.image || !confirm(`Reject and delete ${A.image.filename}?`)) return;
  await api(`/api/images/${A.image.id}`, { method: "DELETE" });
  A.image = null;
  if (queueMode()) return loadQueue();
  A.queue.splice(A.pos, 1);
  A.pos = Math.min(A.pos, Math.max(0, A.queue.length - 1));
  refreshStats(); openImage(A.queue[A.pos]);
};

document.addEventListener("keydown", (e) => {
  if (!$("#tab-annotate").classList.contains("active") || e.target.matches("input, textarea, select")) return;
  const k = e.key;
  if (k >= "1" && k <= String(CLASSES.length)) setClass(+k - 1);
  else if (k === "Delete" || k === "Backspace") { if (A.sel >= 0) { A.labels.splice(A.sel, 1); A.sel = -1; A.picked = false; A.dirty = true; draw(); } }
  else if (k === "Escape") { A.sel = -1; A.picked = false; draw(); }
  else if (k === "q" || k === "Q") rotate(-5);
  else if (k === "e" || k === "E") rotate(5);
  else if (k === "s" || k === "S") save();
  else if (k === "Enter") done();
  else if (k === "ArrowRight" && !queueMode()) go(1);
  else if (k === "ArrowLeft" && !queueMode()) go(-1);
  else return;
  e.preventDefault();
});

// --- reports -----------------------------------------------------------------
const fmt = (v, d = 3) => v == null ? "–" : typeof v === "number" ? v.toFixed(d) : v;
async function loadRuns() {
  const runs = await api("/api/runs");
  $("#runs-table tbody").innerHTML = runs.map((r) => `<tr class="clickable" data-run="${r.id}">
    <td>${r.id}</td><td>${esc(r.model)}</td><td>${r.epochs}</td><td>${r.n_train ?? "–"}</td>
    <td>${fmt(r.metrics?.best_mAP50_95)}</td><td>${fmt(r.metrics?.fps, 1)}</td>
    <td class="status-${r.status}">${r.status}${r.deployed ? ' <span class="badge">deployed</span>' : ""}</td></tr>`).join("")
    || `<tr><td colspan="7" class="hint">No runs yet. Start one with <code>bridge cycle</code>.</td></tr>`;
  document.querySelectorAll("[data-run]").forEach((tr) => tr.onclick = () => showReport(+tr.dataset.run, tr));
  const files = await api("/api/metrics");
  $("#metrics-list").innerHTML = files.map((f) => `<a data-mf="${esc(f)}">${esc(f)}</a>`).join("") || "None yet.";
  document.querySelectorAll("[data-mf]").forEach((a) => a.onclick = async () => {
    const m = await api(`/api/metrics/${encodeURIComponent(a.dataset.mf)}`);
    if (m.history) m.history = `(${m.history.length} epochs; see results.csv)`;
    $("#metrics-view").textContent = JSON.stringify(m, null, 2);
  });
}
async function showReport(id, tr) {
  document.querySelectorAll("[data-run]").forEach((t) => t.classList.toggle("sel", t === tr));
  try {
    const md = await api(`/api/runs/${id}/report`);
    $("#report").innerHTML = markdown(md) +
      `<h2>Training curves</h2><img src="/api/runs/${id}/plot/results.png" alt="results.png from ultralytics" onerror="this.remove()">`;
  } catch { $("#report").innerHTML = `<p class="hint">Run ${id} has no report yet.</p>`; }
}
// Just enough Markdown for our reports: headings, tables, bullets, bold, code, italics.
function markdown(md) {
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>").replace(/_(.+?)_/g, "<i>$1</i>");
  const out = []; let table = null, list = false;
  const flush = () => { if (table) { out.push("<table>" + table.join("") + "</table>"); table = null; } if (list) { out.push("</ul>"); list = false; } };
  for (const line of md.split("\n")) {
    if (/^\|.*\|$/.test(line)) {
      if (/^\|[-| ]+\|$/.test(line)) continue;
      const cells = line.slice(1, -1).split("|").map((c) => inline(c.trim()));
      const tag = table ? "td" : "th";
      table = table || [];
      table.push("<tr>" + cells.map((c) => `<${tag}>${c}</${tag}>`).join("") + "</tr>");
      continue;
    }
    if (/^\s*[-*] /.test(line)) { if (table) flush(); if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(line.replace(/^\s*[-*] /, ""))}</li>`); continue; }
    flush();
    const h = line.match(/^(#{1,3}) (.*)/);
    if (h) out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  flush();
  return out.join("\n");
}

// --- images ------------------------------------------------------------------
const IM = { state: "" };
document.querySelectorAll("#im-filters [data-state]").forEach((b) => b.onclick = () => {
  IM.state = b.dataset.state;
  document.querySelectorAll("#im-filters [data-state]").forEach((x) => x.classList.toggle("active", x === b));
  loadImages();
});
$("#im-dataset").onchange = () => loadImages();
async function loadImages() {
  const [counts, ds] = await Promise.all([api("/api/images/states"), api("/api/datasets")]);
  for (const k of ["pending", "annotating", "labeled"]) $(`#im-n-${k}`).textContent = counts[k];
  $("#im-n-all").textContent = counts.pending + counts.annotating + counts.labeled;
  const cur = $("#im-dataset").value;
  $("#im-dataset").innerHTML = `<option value="">All datasets</option>` +
    ds.map((d) => `<option ${d.name === cur ? "selected" : ""}>${esc(d.name)}</option>`).join("");
  const params = new URLSearchParams({ limit: 300, newest_first: true });
  if (IM.state) params.set("state", IM.state);
  if (cur) params.set("dataset", cur);
  const imgs = await api(`/api/images?${params}`);
  const now = Date.now() / 1000;
  const when = (t) => t ? ago(now - t) : "";
  $("#im-grid").innerHTML = imgs.map((i) => {
    const line = {
      pending: `waiting${i.n_auto ? ` · ${i.n_auto} auto box(es)` : ""}`,
      annotating: `${esc(i.claimed_by)} · claim ends in ${Math.max(0, Math.round((i.claim_expires - now) / 60))} min`,
      labeled: `${esc(i.labeled_by || "?")} · ${when(i.labeled_at)} · ${i.n_labels} box(es)`,
    }[i.state];
    const dot = { pending: "", annotating: "busy", labeled: "ok" }[i.state];
    const label = { pending: "Pending label", annotating: "Annotating", labeled: "Labeled" }[i.state];
    return `<div class="thumb ${i.state}" title="${esc(i.filename)} · ${esc(i.origin)}">
      <img loading="lazy" src="/api/images/${i.id}/file?v=${i.status}" alt="">
      <div class="meta"><div class="state"><span class="dot ${dot}"></span>${label}</div>
      <div class="sub">${line}</div><div class="sub">${esc(i.collection)} · added ${when(i.ingested_at)}</div></div></div>`;
  }).join("") || `<p class="hint">No images${IM.state ? " in this state" : ""}.</p>`;
  $("#im-foot").textContent = imgs.length === 300 ? "Showing the newest 300." : "";
}
setInterval(() => { if ($("#tab-images").classList.contains("active")) loadImages(); }, 10000);

// --- cluster -----------------------------------------------------------------
const ago = (s) => s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
async function loadCluster() {
  const c = await api("/api/cluster");
  const st = c.status;
  const tile = (v, k) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  const llmUp = st && st.seats.some((s) => s.is_agent && s.llm);
  $("#cl-tiles").innerHTML = [
    tile(st ? `${st.seats.filter((s) => s.pod && s.pod.ready).length}/${st.seats.length}` : "…", "seats ready"),
    tile(c.active_runs.length, "runs training now"),
    tile(`${c.done_runs}${c.failed_runs ? ` <small class="lvl-error">/ ${c.failed_runs} failed</small>` : ""}`, "runs finished"),
    tile(c.deployed ? `run ${c.deployed}` : "none", "deployed model"),
    tile(`${c.queue.ingest} / ${c.queue.labeled}`, "to label / labeled"),
    tile(c.queue.labelers.length ? esc(c.queue.labelers.join(", ")) : "nobody", "labeling now"),
    tile(st ? (llmUp ? `<span class="dot ok"></span>up` : `<span class="dot bad"></span>down`) : "…", "Qwen3 (planner)"),
    tile(st ? (st.gptoss ? `<span class="dot ok"></span>set` : `<span class="dot"></span>not set`) : "…", "gpt-oss (reports)"),
  ].join("");

  if (st) {
    $("#cl-seats").innerHTML = st.seats.map((s) => {
      const pod = s.pod || {};
      const health = s.error ? "bad" : pod.ready ? (s.runs || []).some((r) => r.status === "running") || (s.procs || []).length ? "busy" : "ok" : "bad";
      const label = { ok: "idle", busy: "working", bad: s.error ? "unreachable" : pod.phase || "unknown" }[health];
      const rows = [];
      if (s.error) rows.push(`<div class="row lvl-error">${esc(s.error)}</div>`);
      rows.push(`<div class="row"><span>pod</span><span>${esc(pod.phase || "?")}${pod.restarts ? `, ${pod.restarts} restarts` : ""}</span></div>`);
      if (s.load) rows.push(`<div class="row"><span>node load (1/5/15 min)</span><span>${s.load.map((x) => x.toFixed(1)).join(" / ")}</span></div>`);
      rows.push(`<div class="row"><span>CPU limit</span><span>${s.cpus ?? "?"} cores${st.train_threads[s.seat] ? ` (trains with ${st.train_threads[s.seat]})` : ""}</span></div>`);
      rows.push(`<div class="row"><span>training deps</span><span>${s.deps ? "ready" : "not installed"}</span></div>`);
      if (s.is_agent) rows.push(`<div class="row"><span>Qwen3 server</span><span><span class="dot ${s.llm ? "ok" : "bad"}"></span>${s.llm ? "answering" : "down"}</span></div>`);
      for (const r of s.runs || []) {
        const pct = r.epochs ? Math.min(100, 100 * r.epochs_done / r.epochs) : 0;
        rows.push(`<div class="row"><span>${esc(r.run)} ${esc(r.model || "")}${r.imgsz ? ` @${r.imgsz}` : ""}</span>
          <span class="status-${r.status}">${esc(r.status)}${r.status === "running" ? ` ${r.epochs_done}/${r.epochs}` : ""}</span></div>` +
          (r.status === "running" ? `<div class="bar"><i style="width:${pct}%"></i></div>` : ""));
      }
      for (const j of s.jobs || []) if (!j.done && j.age_s < 7200)
        rows.push(`<div class="row"><span>job ${esc(j.job)}</span><span>running, started ${ago(j.age_s)}</span></div>`);
      return `<div class="card seat"><h3>${esc(s.seat)}<span><span class="dot ${health}"></span><small>${label}</small></span></h3>
        <div class="roles">${s.roles.map(esc).join(" · ")}</div>${rows.join("")}</div>`;
    }).join("");
  }
  $("#cl-events tbody").innerHTML = c.events.slice().reverse().map((e) => `<tr class="lvl-${e.level}">
    <td>${new Date(e.ts * 1000).toLocaleTimeString()}</td><td>${esc(e.source)}</td><td>${esc(e.msg)}</td></tr>`).join("")
    || `<tr><td class="hint">No bridge or agent activity yet.</td></tr>`;
  $("#cl-foot").textContent = st
    ? `Pods checked ${ago(Date.now() / 1000 - st.ts)} (took ${st.took_s} s); refreshes every ${c.refresh_s} s.` + (c.error ? ` Last refresh failed: ${c.error}` : "")
    : c.error ? `Cannot reach the cluster: ${c.error}` : "Asking the pods…";
}
setInterval(() => { if ($("#tab-cluster").classList.contains("active")) loadCluster(); }, 5000);

// --- logs --------------------------------------------------------------------
const L = { last: 0, timer: null, key: "" };
async function loadLogs() {
  const params = new URLSearchParams();
  for (const [k, sel] of [["source", "#lg-source"], ["level", "#lg-level"], ["run_id", "#lg-run"], ["q", "#lg-q"]])
    if ($(sel).value) params.set(k, $(sel).value);
  const key = params.toString();
  if (key !== L.key) { L.key = key; L.last = 0; $("#lg-table tbody").innerHTML = ""; }
  if (L.last) params.set("after", L.last);
  const r = await api(`/api/logs?${params}`);
  const cur = $("#lg-source").value;
  $("#lg-source").innerHTML = `<option value="">All sources</option>` +
    r.sources.map((s) => `<option ${s === cur ? "selected" : ""}>${esc(s)}</option>`).join("");
  const rows = r.entries.map((e) => `<tr class="lvl-${e.level}"><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td>
    <td>${e.level}</td><td>${esc(e.source)}</td><td>${e.run_id ?? ""}</td><td>${esc(e.msg)}</td></tr>`).join("");
  $("#lg-table tbody").insertAdjacentHTML("beforeend", rows);
  if (r.entries.length) L.last = r.entries[r.entries.length - 1].ts;
  if (rows && $("#lg-follow").checked) window.scrollTo(0, document.body.scrollHeight);
}
["#lg-source", "#lg-level", "#lg-run", "#lg-q"].forEach((s) => $(s).onchange = loadLogs);
setInterval(() => { if ($("#tab-logs").classList.contains("active") && $("#lg-follow").checked) loadLogs(); }, 3000);

// --- boot --------------------------------------------------------------------
$("#who").onclick = askName;
$("#who").textContent = WHO ? `👤 ${WHO}` : "Set your name";
// Leaving the page hands an unfinished claimed image back to the queue.
window.addEventListener("pagehide", () => {
  if (A.image && A.image.status === "ingest")
    navigator.sendBeacon(`/api/images/${A.image.id}/release?who=${encodeURIComponent(WHO)}`);
});
setInterval(refreshStats, 15000);
api("/api/config").then((c) => { CLASSES = c.classes; classButtons(); });
loadDatasets();
