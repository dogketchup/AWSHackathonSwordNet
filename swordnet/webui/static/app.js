// SwordNet web UI. No build step, no framework.
const $ = (s) => document.querySelector(s);
const api = async (path, opts = {}) => {
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
  ({ upload: loadDatasets, annotate: loadQueue, reports: loadRuns })[name]();
}

async function refreshStats() {
  const ds = await api("/api/datasets");
  const ing = ds.reduce((a, d) => a + (d.n_ingest || 0), 0), lab = ds.reduce((a, d) => a + (d.n_labeled || 0), 0);
  $("#stats").textContent = `${ing} to label · ${lab} labeled`;
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
    $("#up-msg").textContent = `Added ${r.added} image(s).${v ? " " + v : ""}`;
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
const A = { queue: [], pos: 0, img: null, image: null, labels: [], sel: -1, cls: 0, drag: null, dirty: false };

function classButtons() {
  $("#an-classes").innerHTML = CLASSES.map((c, i) =>
    `<button class="cls ${i === A.cls ? "active" : ""}" data-cls="${i}" style="color:var(--${c}, inherit)">${i + 1} ${esc(c)}</button>`).join(" ");
  document.querySelectorAll("[data-cls]").forEach((b) => b.onclick = () => setClass(+b.dataset.cls));
}
function setClass(i) {
  A.cls = i;
  if (A.sel >= 0) { A.labels[A.sel].cls = i; A.labels[A.sel].source = touched(A.labels[A.sel].source); A.dirty = true; }
  classButtons(); draw();
}
// Editing an auto-label makes it a human label (the editor shows it as reviewed).
const touched = (src) => src.startsWith("auto") ? "manual" : src;

async function loadQueue() {
  A.queue = await api(`/api/images?status=${$("#an-filter").value}`);
  A.pos = Math.min(A.pos, Math.max(0, A.queue.length - 1));
  refreshStats();
  await openImage();
}
$("#an-filter").onchange = () => { A.pos = 0; loadQueue(); };

async function openImage() {
  const item = A.queue[A.pos];
  $("#an-empty").style.display = item ? "none" : "block";
  canvas.style.display = item ? "block" : "none";
  $("#an-pos").textContent = item ? `${A.pos + 1} / ${A.queue.length}` : "0 / 0";
  if (!item) return;
  const { image, labels } = await api(`/api/images/${item.id}/labels`);
  A.image = image; A.labels = labels.map((l) => ({ cls: l.cls, points: l.points, source: l.source, score: l.score }));
  A.sel = -1; A.dirty = false;
  const img = new Image();
  img.onload = () => { A.img = img; canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(); };
  img.src = `/api/images/${item.id}/file?v=${image.status}`;
  $("#an-info").textContent = `${image.filename} · ${image.origin} · dataset ${image.dataset}`;
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
    A.sel = A.labels.length - 1; A.dirty = true;
  } else {
    // click: select the smallest box under the cursor
    const hits = A.labels.map((l, i) => [i, l]).filter(([, l]) => inside([d.x0, d.y0], l.points));
    const area = (p) => Math.abs(p.reduce((s, [x, y], i) => { const [u, v] = p[(i + 1) % 4]; return s + x * v - u * y; }, 0)) / 2;
    hits.sort((a, b) => area(a[1].points) - area(b[1].points));
    A.sel = hits.length ? hits[0][0] : -1;
    if (A.sel >= 0) { A.cls = A.labels[A.sel].cls; classButtons(); }
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
  await save();
  await api(`/api/images/${A.image.id}/processed`, { method: "POST" });
  if ($("#an-filter").value === "ingest") A.queue.splice(A.pos, 1); else A.pos++;
  A.pos = Math.min(A.pos, Math.max(0, A.queue.length - 1));
  refreshStats();
  openImage();
}
async function go(delta) {
  if (A.dirty && !confirm("Discard unsaved changes on this image?")) return;
  A.pos = Math.max(0, Math.min(A.queue.length - 1, A.pos + delta));
  openImage();
}
$("#an-prev").onclick = () => go(-1);
$("#an-next").onclick = () => go(1);
$("#an-save").onclick = save;
$("#an-done").onclick = done;
$("#an-reject").onclick = async () => {
  if (!A.image || !confirm(`Reject and delete ${A.image.filename}?`)) return;
  await api(`/api/images/${A.image.id}`, { method: "DELETE" });
  A.queue.splice(A.pos, 1);
  A.pos = Math.min(A.pos, Math.max(0, A.queue.length - 1));
  refreshStats(); openImage();
};

document.addEventListener("keydown", (e) => {
  if (!$("#tab-annotate").classList.contains("active") || e.target.matches("input, textarea, select")) return;
  const k = e.key;
  if (k >= "1" && k <= String(CLASSES.length)) setClass(+k - 1);
  else if (k === "Delete" || k === "Backspace") { if (A.sel >= 0) { A.labels.splice(A.sel, 1); A.sel = -1; A.dirty = true; draw(); } }
  else if (k === "q" || k === "Q") rotate(-5);
  else if (k === "e" || k === "E") rotate(5);
  else if (k === "s" || k === "S") save();
  else if (k === "Enter") done();
  else if (k === "ArrowRight") go(1);
  else if (k === "ArrowLeft") go(-1);
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
    <td class="status-${r.status}">${r.status}</td></tr>`).join("")
    || `<tr><td colspan="7" class="hint">No runs yet. Start one with <code>bridge cycle</code>.</td></tr>`;
  document.querySelectorAll("[data-run]").forEach((tr) => tr.onclick = () => showReport(+tr.dataset.run, tr));
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

// --- boot --------------------------------------------------------------------
api("/api/config").then((c) => { CLASSES = c.classes; classButtons(); });
loadDatasets();
