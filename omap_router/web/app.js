/* omap-router single page UI. Coordinates sent to the API are original-image pixels (x, y). */
"use strict";

const LAYERS = ["labels", "forbidden", "outside", "purple", "hatch", "barriers", "cost"];
const CLASSES = ["WHITE", "YELLOW", "BEIGE", "GREEN", "OLIVE", "GREY", "BLACK", "PURPLE", "BROWN", "BLUE"];
const CLASS_COLORS = {
  WHITE: "#fff", YELLOW: "#f7be50", BEIGE: "#ebcdaa", GREEN: "#6ec864", OLIVE: "#a0a028",
  GREY: "#828282", BLACK: "#000", PURPLE: "#c828c8", BROWN: "#aa6e32", BLUE: "#00aae6",
};

const state = {
  project: null,
  clickMode: null, // sample | addControl | circle | measure | crop
  pending: [],
  overlays: {},
  visible: { forbidden: true, routes: true, controls: true },
  opacity: 0.6,
  layerVersion: Date.now(),
};

// ---------------------------------------------------------------- helpers

const $ = (sel) => document.querySelector(sel);
const ll = (x, y) => L.latLng(-y, x);
const xy = (latlng) => [latlng.lng, -latlng.lat];

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + kind;
}

async function api(method, path, body, isForm = false) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  const res = await fetch("/api" + path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

async function busy(label, fn) {
  const buttons = document.querySelectorAll("#sidebar button");
  buttons.forEach((b) => (b.disabled = true));
  setStatus(label + "…", "busy");
  try {
    const out = await fn();
    showMessages();
    return out;
  } catch (e) {
    setStatus(label + " failed: " + e.message, "error");
    throw e;
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

function showMessages(extra) {
  const p = state.project;
  if (!p) return;
  const msgs = (p.messages || []).slice();
  if (extra) msgs.unshift(extra);
  setStatus(msgs.length ? msgs.join("\n") : "Ready.", msgs.length ? "" : "");
}

// ---------------------------------------------------------------- map

const map = L.map("map", { crs: L.CRS.Simple, minZoom: -5, maxZoom: 5, zoomSnap: 0.25, attributionControl: false });
map.setView([0, 0], 0);
const ignore = { pmIgnore: true };
const controlLayer = L.layerGroup([], ignore).addTo(map);
const routeLayer = L.layerGroup([], ignore).addTo(map);
const sampleLayer = L.layerGroup([], ignore).addTo(map);
const toolLayer = L.layerGroup([], ignore).addTo(map);
const editGroup = L.featureGroup().addTo(map);
let imageLayer = null;

map.pm.addControls({
  position: "topleft",
  drawMarker: false, drawCircleMarker: false, drawPolyline: false, drawCircle: false, drawText: false,
  drawRectangle: true, drawPolygon: true, editMode: true, dragMode: true, cutPolygon: false,
  removalMode: true, rotateMode: false,
});
map.pm.setGlobalOptions({ layerGroup: editGroup });

function imageBounds() {
  const p = state.project;
  return L.latLngBounds(ll(0, 0), ll(p.width, p.height));
}

// ---------------------------------------------------------------- project

async function refreshProjectList() {
  const list = await api("GET", "/projects");
  const sel = $("#project-list");
  sel.innerHTML = '<option value="">— open project —</option>';
  for (const p of list) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = `${p.id} (${p.width}×${p.height})`;
    sel.appendChild(o);
  }
  if (state.project) sel.value = state.project.id;
}

async function openProject(id) {
  state.project = await api("GET", "/projects/" + id);
  const p = state.project;
  if (imageLayer) map.removeLayer(imageLayer);
  imageLayer = L.imageOverlay(`/api/projects/${p.id}/image`, imageBounds(), { pmIgnore: true }).addTo(map);
  imageLayer.bringToBack();
  map.fitBounds(imageBounds());
  state.layerVersion = Date.now();
  rebuildOverlays();
  renderAll();
  history.replaceState(null, "", "#" + p.id);
  $("#project-list").value = p.id;
  showMessages(p.controls.length ? "" : "Run “Re-analyze”, then “Detect controls”.");
}

function renderAll() {
  renderProjectInfo();
  renderSamples();
  renderControls();
  renderEdits();
  renderScale();
  renderLegs();
}

function renderProjectInfo() {
  const p = state.project;
  const crop = p.crop ? `crop ${p.crop.join(", ")}` : "no crop";
  $("#project-info").textContent = `${p.width}×${p.height}px, ${crop}`;
  toolLayer.clearLayers();
  if (p.crop) {
    const [x0, y0, x1, y1] = p.crop;
    L.rectangle([ll(x0, y0), ll(x1, y1)], { color: "#333", weight: 1, dashArray: "6 4", fill: false, interactive: false, pmIgnore: true }).addTo(toolLayer);
  }
}

// ---------------------------------------------------------------- overlays

function buildLayerToggles() {
  const box = $("#layer-toggles");
  for (const name of [...LAYERS, "routes", "controls"]) {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!state.visible[name];
    cb.onchange = () => { state.visible[name] = cb.checked; applyVisibility(); };
    lab.append(cb, " " + name);
    box.appendChild(lab);
  }
  $("#opacity").oninput = (e) => {
    state.opacity = +e.target.value;
    for (const o of Object.values(state.overlays)) o.setOpacity(state.opacity);
  };
}

function rebuildOverlays() {
  for (const o of Object.values(state.overlays)) map.removeLayer(o);
  state.overlays = {};
  if (!state.project) return;
  for (const name of LAYERS) {
    const url = `/api/projects/${state.project.id}/layers/${name}.png?v=${state.layerVersion}`;
    state.overlays[name] = L.imageOverlay(url, imageBounds(), { opacity: state.opacity, pmIgnore: true, interactive: false });
  }
  applyVisibility();
}

function applyVisibility() {
  for (const [name, o] of Object.entries(state.overlays)) {
    if (state.visible[name]) o.addTo(map); else map.removeLayer(o);
  }
  if (state.visible.routes) routeLayer.addTo(map); else map.removeLayer(routeLayer);
  if (state.visible.controls) controlLayer.addTo(map); else map.removeLayer(controlLayer);
}

// ---------------------------------------------------------------- click modes

function setMode(mode, button) {
  document.querySelectorAll("button.toggle").forEach((b) => b.classList.remove("active"));
  state.pending = [];
  if (state.clickMode === mode) mode = null;
  state.clickMode = mode;
  if (mode && button) button.classList.add("active");
  map.getContainer().classList.toggle("picking", !!mode);
  const hints = {
    sample: "Click on the map to add colour samples for the selected class.",
    addControl: "Click on the map to add a control.",
    circle: "Click the centre of a control circle, then a point on its ring.",
    measure: "Click two points a known distance apart.",
    crop: "Click two opposite corners of the crop rectangle.",
  };
  if (mode) setStatus(hints[mode]); else showMessages();
}

map.on("click", async (e) => {
  if (!state.project || !state.clickMode) return;
  const [x, y] = xy(e.latlng);
  const p = state.project;
  switch (state.clickMode) {
    case "sample": {
      const cls = $("#sample-class").value;
      (p.color_samples[cls] = p.color_samples[cls] || []).push([Math.round(x), Math.round(y)]);
      state.project = await api("PUT", `/projects/${p.id}/color-samples`, p.color_samples);
      renderSamples();
      break;
    }
    case "addControl": {
      const id = Math.max(0, ...p.controls.map((c) => c.id)) + 1;
      const r = p.calibration.r0 || 15;
      const list = sortedControls();
      const fin = list.findIndex((c) => c.kind === "finish" && c.order !== null);
      const c = { id, order: 0, kind: "control", x, y, r, source: "user" };
      if (fin >= 0) list.splice(fin, 0, c); else list.push(c);
      await saveControls(list);
      break;
    }
    case "circle":
    case "measure":
    case "crop": {
      state.pending.push([x, y]);
      L.circleMarker(e.latlng, { radius: 4, color: "#f60", pmIgnore: true }).addTo(toolLayer);
      if (state.pending.length < 2) return;
      const [p1, p2] = state.pending;
      const mode = state.clickMode;
      setMode(null);
      renderProjectInfo();
      if (mode === "circle") {
        const r0 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
        state.project = await busy("Calibrating", () => api("PUT", `/projects/${p.id}/calibration`, { r0 }));
        renderScale();
        showMessages(`Control circle radius set to ${r0.toFixed(1)} px. Re-analyze and detect controls.`);
      } else if (mode === "measure") {
        const m = parseFloat(prompt("Real length of the measured line in metres:", "100"));
        if (!(m > 0)) return;
        state.project = await busy("Calibrating", () => api("PUT", `/projects/${p.id}/calibration`, { p1, p2, meters: m }));
        renderScale();
        renderLegs();
      } else {
        const crop = [Math.round(Math.min(p1[0], p2[0])), Math.round(Math.min(p1[1], p2[1])),
          Math.round(Math.max(p1[0], p2[0])), Math.round(Math.max(p1[1], p2[1]))];
        state.project = await busy("Setting crop", () => api("PUT", `/projects/${p.id}/crop`, { crop }));
        renderProjectInfo();
        showMessages("Crop set. Re-analyze to apply.");
      }
      break;
    }
  }
});

// ---------------------------------------------------------------- colours

function buildClassSelect() {
  const sel = $("#sample-class");
  for (const c of CLASSES) {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c.toLowerCase();
    sel.appendChild(o);
  }
}

function renderSamples() {
  sampleLayer.clearLayers();
  const p = state.project;
  const parts = [];
  for (const [cls, pts] of Object.entries(p.color_samples || {})) {
    parts.push(`${cls.toLowerCase()}: ${pts.length}`);
    for (const [x, y] of pts) {
      L.circleMarker(ll(x, y), { radius: 4, color: "#000", weight: 1, fillColor: CLASS_COLORS[cls] || "#f0f", fillOpacity: 1, pmIgnore: true })
        .bindTooltip(cls.toLowerCase()).addTo(sampleLayer);
    }
  }
  $("#sample-summary").textContent = parts.length ? "Samples: " + parts.join(", ") : "No samples (default colours).";
}

// ---------------------------------------------------------------- controls

function sortedControls() {
  return state.project.controls.slice().sort((a, b) => {
    const ao = a.order === null ? Infinity : a.order;
    const bo = b.order === null ? Infinity : b.order;
    return ao - bo || a.id - b.id;
  });
}

function controlLabel(c) {
  if (c.order === null) return "?";
  if (c.kind === "start") return "S";
  if (c.kind === "finish") return "F";
  return String(c.order);
}

function renumber(list) {
  let n = 0;
  for (const c of list) c.order = c.order === null ? null : n++;
  return list;
}

async function saveControls(list) {
  const p = state.project;
  state.project = await api("PUT", `/projects/${p.id}/controls`, renumber(list));
  renderControls();
  renderLegs();
}

function renderControls() {
  controlLayer.clearLayers();
  const p = state.project;
  const list = sortedControls();
  const r0 = p.calibration.r0;
  $("#r0-info").textContent = r0
    ? `Control circle radius r0 = ${r0.toFixed(1)} px (${p.calibration.r0_source}), ${p.calibration.px_per_mm.toFixed(2)} px/mm`
    : "Control circle size unknown.";
  const tbody = $("#control-table tbody");
  tbody.innerHTML = "";
  list.forEach((c, i) => {
    const used = c.order !== null;
    const color = { start: "#dc2626", finish: "#2563eb", control: "#c026d3" }[c.kind];
    const circle = L.circle(ll(c.x, c.y), { radius: c.r || r0 || 15, color, weight: 2, fill: false, dashArray: used ? null : "4 4", interactive: false, pmIgnore: true }).addTo(controlLayer);
    const icon = L.divIcon({ className: "", html: `<div class="ctrl-icon ${c.kind} ${used ? "" : "unused"}" style="width:22px;height:22px">${controlLabel(c)}</div>`, iconSize: [22, 22] });
    const m = L.marker(ll(c.x, c.y), { icon, draggable: true, pmIgnore: true, title: `${c.kind} #${c.id}${c.code ? ", code " + c.code : ""}` }).addTo(controlLayer);
    m.on("drag", (e) => circle.setLatLng(e.latlng));
    m.on("dragend", async (e) => {
      const [x, y] = xy(e.target.getLatLng());
      c.x = x; c.y = y; c.source = "user";
      await saveControls(list);
    });
    m.on("contextmenu", async () => {
      list.splice(list.indexOf(c), 1);
      await saveControls(list);
    });

    const tr = document.createElement("tr");
    if (!used) tr.className = "unused";
    const tdN = document.createElement("td");
    tdN.textContent = controlLabel(c) + (c.code ? ` (${c.code})` : "");
    const tdU = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = used;
    cb.title = "part of the course";
    cb.onchange = () => { c.order = cb.checked ? 0 : null; saveControls(sortedByListPosition(list)); };
    tdU.appendChild(cb);
    const tdK = document.createElement("td");
    const sel = document.createElement("select");
    for (const k of ["start", "control", "finish"]) {
      const o = document.createElement("option");
      o.value = k; o.textContent = k; o.selected = c.kind === k;
      sel.appendChild(o);
    }
    sel.onchange = () => { c.kind = sel.value; saveControls(list); };
    tdK.appendChild(sel);
    const tdB = document.createElement("td");
    const up = document.createElement("button");
    up.textContent = "▲";
    up.onclick = () => { if (i > 0) { [list[i - 1], list[i]] = [list[i], list[i - 1]]; saveControls(sortedByListPosition(list)); } };
    const down = document.createElement("button");
    down.textContent = "▼";
    down.onclick = () => { if (i < list.length - 1) { [list[i + 1], list[i]] = [list[i], list[i + 1]]; saveControls(sortedByListPosition(list)); } };
    const zoom = document.createElement("button");
    zoom.textContent = "⌖";
    zoom.title = "zoom to";
    zoom.onclick = () => map.setView(ll(c.x, c.y), Math.max(map.getZoom(), 1));
    tdB.append(up, down, zoom);
    tr.append(tdN, tdU, tdK, tdB);
    tbody.appendChild(tr);
  });
}

// After a reorder, list position defines order: give used controls a provisional order matching position.
function sortedByListPosition(list) {
  list.forEach((c, i) => { if (c.order !== null) c.order = i; });
  return list;
}

// ---------------------------------------------------------------- edits

function editStyle(kind) {
  return kind === "forbid"
    ? { color: "#dc2626", fillColor: "#dc2626", fillOpacity: 0.25, weight: 2 }
    : { color: "#16a34a", fillColor: "#16a34a", fillOpacity: 0.25, weight: 2 };
}

function attachEditLayer(layer, kind, id) {
  layer._omap = { kind, id };
  layer.setStyle(editStyle(kind));
  layer.bindTooltip(kind);
  layer.on("pm:edit pm:dragend", saveEdits);
}

function renderEdits() {
  editGroup.clearLayers();
  for (const e of state.project.edits || []) {
    const poly = L.polygon(e.points.map(([x, y]) => ll(x, y)));
    attachEditLayer(poly, e.kind, e.id);
    poly.addTo(editGroup);
  }
  renderEditSummary();
}

function renderEditSummary() {
  const edits = state.project.edits || [];
  const f = edits.filter((e) => e.kind === "forbid").length;
  $("#edit-summary").textContent = `${f} forbid, ${edits.length - f} allow polygon(s).`;
}

async function saveEdits() {
  const p = state.project;
  if (!p) return;
  const edits = [];
  editGroup.eachLayer((layer) => {
    if (!layer._omap || !layer.getLatLngs) return;
    let ring = layer.getLatLngs();
    while (Array.isArray(ring[0])) ring = ring[0];
    edits.push({ id: layer._omap.id, kind: layer._omap.kind, points: ring.map(xy) });
  });
  state.project = await api("PUT", `/projects/${p.id}/edits`, edits);
  renderEditSummary();
}

map.on("pm:create", (e) => {
  if (!state.project) { map.removeLayer(e.layer); return; }
  const kind = document.querySelector('input[name="edit-kind"]:checked').value;
  const id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now())).slice(0, 8);
  attachEditLayer(e.layer, kind, id);
  if (!editGroup.hasLayer(e.layer)) editGroup.addLayer(e.layer);
  saveEdits();
});
map.on("pm:remove", () => saveEdits());

// ---------------------------------------------------------------- scale

function renderScale() {
  const c = state.project.calibration;
  $("#scale").value = c.scale;
  const mpp = c.px_per_mm ? c.scale / 1000 / c.px_per_mm : null;
  $("#scale-info").textContent =
    `1:${c.scale} ${c.scale_confirmed ? "(confirmed)" : "(assumed — lengths are estimates)"}` +
    (mpp ? `, ${mpp.toFixed(3)} m/px` : "");
}

// ---------------------------------------------------------------- legs

function renderLegs() {
  routeLayer.clearLayers();
  const p = state.project;
  const byId = Object.fromEntries(p.controls.map((c) => [c.id, c]));
  const tbody = $("#leg-table tbody");
  tbody.innerHTML = "";
  let total = 0;
  const est = p.calibration.scale_confirmed ? "" : " (est.)";
  for (const leg of p.legs || []) {
    const a = byId[leg.from_id], b = byId[leg.to_id];
    const name = `${a ? controlLabel(a) : "?"}–${b ? controlLabel(b) : "?"}`;
    let line;
    if (leg.ok && leg.path.length) {
      line = L.polyline(leg.path.map(([x, y]) => ll(x, y)), { color: "#1d4ed8", weight: 3, opacity: 0.85, pmIgnore: true });
      total += leg.length_m;
    } else if (a && b) {
      line = L.polyline([ll(a.x, a.y), ll(b.x, b.y)], { color: "#dc2626", weight: 3, dashArray: "8 6", pmIgnore: true });
    }
    if (line) line.bindTooltip(name).addTo(routeLayer);
    const tr = document.createElement("tr");
    if (!leg.ok) tr.className = "bad";
    const len = leg.ok ? `${Math.round(leg.length_m)} m${est}` : "—";
    tr.innerHTML = `<td>${name}</td><td>${len}</td><td>${leg.ok ? "" : leg.message || "unreachable"}</td>`;
    tr.onclick = () => line && map.fitBounds(line.getBounds(), { maxZoom: 2 });
    tbody.appendChild(tr);
  }
  $("#route-total").textContent = (p.legs || []).length ? `Total ${Math.round(total)} m${est}` : "";
}

// ---------------------------------------------------------------- buttons

function bind() {
  $("#upload").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    const p = await busy("Uploading", () => api("POST", "/projects", fd, true));
    await refreshProjectList();
    await openProject(p.id);
    await analyze();
  };
  $("#project-list").onchange = (e) => e.target.value && openProject(e.target.value);
  $("#btn-crop").onclick = (e) => state.project && setMode("crop", e.target);
  $("#btn-crop-clear").onclick = async () => {
    if (!state.project) return;
    state.project = await api("PUT", `/projects/${state.project.id}/crop`, { crop: null });
    renderProjectInfo();
    showMessages("Crop cleared. Re-analyze to apply.");
  };
  $("#btn-sample").onclick = (e) => state.project && setMode("sample", e.target);
  $("#btn-sample-clear").onclick = async () => {
    const p = state.project;
    if (!p) return;
    delete p.color_samples[$("#sample-class").value];
    state.project = await api("PUT", `/projects/${p.id}/color-samples`, p.color_samples);
    renderSamples();
  };
  $("#btn-analyze").onclick = analyze;
  $("#btn-detect").onclick = async () => {
    if (!state.project) return;
    state.project = await busy("Detecting controls", () => api("POST", `/projects/${state.project.id}/controls/detect`));
    renderControls();
    renderLegs();
    renderScale();
    showMessages();
  };
  $("#btn-add-control").onclick = (e) => state.project && setMode("addControl", e.target);
  $("#btn-mark-circle").onclick = (e) => state.project && setMode("circle", e.target);
  $("#btn-draw").onclick = () => state.project && map.pm.enableDraw("Polygon", { snappable: false });
  $("#btn-edits-clear").onclick = async () => {
    const p = state.project;
    if (!p || !(p.edits || []).length) return;
    if (!confirm(`Remove all ${p.edits.length} drawn polygon(s)?`)) return;
    map.pm.disableDraw();
    state.project = await busy("Clearing polygons", () => api("PUT", `/projects/${p.id}/edits`, []));
    renderEdits();
    showMessages("All polygons removed. Re-route to apply.");
  };
  $("#btn-scale").onclick = async () => {
    if (!state.project) return;
    const scale = parseInt($("#scale").value, 10);
    state.project = await busy("Setting scale", () => api("PUT", `/projects/${state.project.id}/calibration`, { scale }));
    renderScale();
    renderLegs();
  };
  $("#btn-measure").onclick = (e) => state.project && setMode("measure", e.target);
  $("#btn-route").onclick = async () => {
    if (!state.project) return;
    const legs = await busy("Routing", () => api("POST", `/projects/${state.project.id}/route`));
    state.project.legs = legs;
    renderLegs();
    const bad = legs.filter((l) => !l.ok).length;
    setStatus(`${legs.length} legs computed` + (bad ? `, ${bad} unreachable.` : "."), bad ? "error" : "");
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMode(null); });
}

async function analyze() {
  if (!state.project) return;
  const res = await busy("Analyzing map", () => api("POST", `/projects/${state.project.id}/analyze`));
  state.project = res.project;
  state.layerVersion = Date.now();
  rebuildOverlays();
  renderAll();
  showMessages(state.project.controls.length ? "" : "Analysis done. Next: “Detect controls”.");
}

// ---------------------------------------------------------------- init

buildLayerToggles();
buildClassSelect();
bind();
refreshProjectList().then(() => {
  const id = location.hash.slice(1);
  if (id) openProject(id).catch(() => setStatus("Project not found.", "error"));
});
