// Time and Space: animated tracking pitch + every viewer's verified explanation of the same moment.
"use strict";

const TYPE_LABEL = {
  press_trap: "Press wins the ball", press_broken: "Press broken", missed_option: "Better pass missed",
  goal: "Goal", big_chance: "Big chance", momentum_swing: "Momentum swing",
};
const BAND_LABEL = { none: "No pressure", some: "Some pressure", high: "High pressure" };
const THIRD_LABEL = { attacking: "final third", middle: "middle third", defensive: "defensive third" };
const KEY_NOTE = "A key pressing moment of the match";
const SLOW_WINDOW = 10;      // frames around a key frame played in slow motion
const HOLD_MS = 2200;        // pause on the key frame
const OVERLAY_IN = 14, OVERLAY_OUT = 30;   // frames before / after the key frame the overlay is shown

const NOTABLE = new Set(["goal", "shot", "save", "foul", "corner", "substitution", "period_start", "period_end"]);
const TRAIL = 10;            // recent event locations drawn in Live mode
const SPRINT_MS = 25 / 3.6;  // sprint threshold: 25 km/h
const MAX_STEP_M = 2.4;      // a jump bigger than this between two samples (0.2 s) is a substitution or restart
const PASS_LABEL_MS = 2500;  // how long the latest pass stays labelled (real time)

const state = {
  data: null, match: null, matchId: null, viewer: null, moment: null, mode: "live", showAll: false,
  clip: null, clipCache: {}, t: 0, playing: true, speed: 1, slowmo: true,
  holdUntil: 0, held: new Set(), last: 0,
  live: { feed: [], i: 0, frame: 0, playing: true, speed: 30, st: null, ticker: [], trail: [], moments: [],
    dirty: true, seen: 0, follow: null, fstats: null, lastPass: null },
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => `${Math.round(100 * x)}%`;
const signed = (x) => (x > 0 ? "+" : "") + x.toFixed(4);

// ------------------------------------------------------------------ data

async function init() {
  state.data = await (await fetch("data/app.json")).json();
  state.matchId = Object.keys(state.data.matches)[0];
  state.match = state.data.matches[state.matchId];
  state.viewer = state.match.viewers[0].id;
  const m = state.match;
  $("scoreline").innerHTML = `${esc(m.home.name)}<span class="score">${esc(m.score)}</span>${esc(m.away.name)}`;
  $("model").textContent = `Recorded with ${state.data.model} · replayed offline`;
  renderViewers();
  renderTimeline();
  renderAnswers();
  renderRecap();
  bindControls();
  await selectMoment(m.key_moments[0]);
  await loadLive();
  setMode("live");
  requestAnimationFrame(loop);
}

async function getClip(id) {
  if (!state.clipCache[id]) state.clipCache[id] = await (await fetch(`data/clips/${id}.json`)).json();
  return state.clipCache[id];
}

function momentInfo(id) { return state.match.moments.find((m) => m.moment_id === id); }

function feedItem(viewer, id) { return (state.match.feeds[viewer] || []).find((i) => i.moment_id === id); }

/** The text this viewer reads for this moment, and why they got it. */
function viewerText(id, viewer) {
  const story = state.match.stories[id];
  const item = feedItem(viewer, id);
  const reasons = [];
  if (story) reasons.push(KEY_NOTE);
  if (item) reasons.push(...item.reasons);
  if (story) {
    const v = story.versions.find((x) => x.profile_id === viewer);
    return { version: v || story.neutral, neutral: story.neutral, facts: story.facts, reasons, inFeed: !!item };
  }
  if (item) return { version: item.version, neutral: item.neutral, facts: null, reasons, inFeed: true };
  const any = Object.values(state.match.feeds).flat().find((i) => i.moment_id === id);
  return { version: any.neutral, neutral: any.neutral, facts: null, reasons: [], inFeed: false, notForViewer: true };
}

// ------------------------------------------------------------------ selection

async function selectMoment(id) {
  const info = momentInfo(id);
  const clip = await getClip(info.clip);
  state.moment = id;
  state.clip = clip;
  // In a chain (press won, press broken, goal) start after the earlier key frames, and do not stop on them.
  const earlier = clip.moments.filter((m) => m.index < info.index);
  state.held = new Set(earlier.map((m) => m.moment_id));
  const prev = earlier.length ? Math.max(...earlier.map((m) => m.index)) : -Infinity;
  state.t = Math.max(0, Math.min(info.index - 15, Math.max(info.index - 45, prev + 20)));
  state.holdUntil = 0;
  state.playing = true;
  $("play").textContent = "❚❚";
  $("scrub").max = clip.xy.length - 1;
  renderTimeline();
  renderStory();
}

function selectViewer(id) {
  state.viewer = id;
  state.live.dirty = true;
  const focus = state.match.viewers.find((v) => v.id === id)?.focus_player;
  if (focus && state.live.track) setFollow(focus);
  renderViewers();
  renderTimeline();
  renderStory();
  renderRecap();
}

// ------------------------------------------------------------------ panels

function renderViewers() {
  $("viewers").innerHTML = state.match.viewers.map((v) => `
    <button class="viewer ${v.id === state.viewer ? "active" : ""}" data-id="${v.id}">
      ${esc(v.name)}<small>${esc(v.extras.join(" · ") || "English")}</small></button>`).join("");
  $("viewers").querySelectorAll(".viewer").forEach((b) => (b.onclick = () => selectViewer(b.dataset.id)));
}

function renderTimeline() {
  const keys = new Set(state.match.key_moments);
  const mine = (m) => keys.has(m.moment_id) || feedItem(state.viewer, m.moment_id);
  const shown = state.match.moments.filter((m) => state.showAll || mine(m) || m.moment_id === state.moment);
  const viewer = state.match.viewers.find((v) => v.id === state.viewer);
  $("timeline").innerHTML = shown.map((m) => {
    const cls = ["chip", keys.has(m.moment_id) ? "key" : "", feedItem(state.viewer, m.moment_id) ? "feed" : "",
      m.moment_id === state.moment ? "active" : "", mine(m) ? "" : "dim"].join(" ");
    return `<button class="${cls}" data-id="${m.moment_id}"><span class="t">${esc(m.clock)}</span>${esc(TYPE_LABEL[m.type] || m.type)}</button>`;
  }).join("") + `<label class="toggle all"><input type="checkbox" id="show-all" ${state.showAll ? "checked" : ""}>
    show all moments (not only ${esc(viewer.name)}'s)</label>`;
  $("timeline").querySelectorAll(".chip").forEach((b) => (b.onclick = () => selectMoment(b.dataset.id)));
  $("show-all").onchange = (e) => { state.showAll = e.target.checked; renderTimeline(); };
}

function badges(v) {
  const out = [];
  const fallback = v.source === "template" || (v.issues && v.issues.length);
  out.push(v.source === "template" ? `<span class="badge info">Safe template</span>`
    : `<span class="badge info">Written by ${esc(v.source.replace(/ \(.*\)/, ""))}</span>`);
  out.push(`<span class="badge ok">✓ Numbers & names verified</span>`);
  if (!fallback && v.source !== "template") out.push(`<span class="badge ok">✓ Meaning checked by Critic</span>`);
  if (v.attempts > 1) out.push(`<span class="badge warn">Rewritten after ${v.attempts - 1} rejection${v.attempts > 2 ? "s" : ""}</span>`);
  if (fallback) out.push(`<span class="badge warn">⚠ Personal version rejected · showing safe text</span>`);
  return out.join("");
}

function factsTable(facts) {
  if (!facts) return "";
  const rows = facts.facts.map((f) => `<tr><td>${esc(f.label)}</td><td>${esc(f.value)} ${esc(f.unit)}
    ${f.meaning ? `<div class="mean">${esc(f.meaning)}</div>` : ""}</td></tr>`).join("");
  return `<details><summary>Facts the agents were allowed to use (${facts.facts.length})</summary>
    <p class="mean">${esc(facts.situation)}</p><table class="facts">${rows}</table></details>`;
}

function curveBox(info, facts) {
  const k = state.clip?.moments.find((m) => m.moment_id === info.moment_id);
  const o = k?.overlay;
  if (!o || o.kind !== "pass" || !o.league) return "";
  const vals = Object.values(o.league);
  const max = Math.max(...vals.map(Math.abs)) || 1;
  const rows = ["none", "some", "high"].map((b) => {
    const v = o.league[b];
    const w = (Math.abs(v) / max) * 50;
    const left = v < 0 ? 50 - w : 50;
    return `<div class="row ${b === o.band ? "here" : ""}"><span>${BAND_LABEL[b]}</span>
      <div class="track"><div class="axis"></div><div class="bar" style="left:${left}%;width:${w}%"></div></div>
      <span class="num">${signed(v)}</span></div>`;
  }).join("");
  const rank = facts?.facts.find((f) => f.key === "league_xt_zone")?.meaning.split("; ")[1] || "";
  return `<div class="curve"><h3>League: threat added per pass in the ${THIRD_LABEL[o.third]}, by pressure</h3>${rows}
    <div class="this">This pass: <b>${signed(o.xt_added)} xT</b> under ${BAND_LABEL[o.band].toLowerCase()}
    ${rank ? `— ${esc(rank)}` : ""}</div></div>`;
}

function renderStory() {
  const info = momentInfo(state.moment);
  const tx = viewerText(state.moment, state.viewer);
  const v = tx.version;
  const chosenFor = state.match.viewers.filter((x) => feedItem(x.id, state.moment)).map((x) => x.name);
  const viewerName = state.match.viewers.find((x) => x.id === state.viewer).name;
  const why = tx.notForViewer ? `<div class="why">Not in the ${esc(viewerName)}'s feed: the relevance agent chose this
    moment for ${esc(chosenFor.join(", "))}. Showing the neutral explanation.</div>`
    : `<div class="why"><b>Why you got this:</b> ${esc(tx.reasons.join("; "))}</div>`;
  const rejected = (v.rejected || []).length ? `<details><summary>What the checkers rejected before this text (${v.rejected.length})</summary>
    <ul class="issues">${v.rejected.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>` : "";
  const neutral = v !== tx.neutral && v.profile_id !== "neutral" ? `<details><summary>Neutral explanation (before personalising)</summary>
    <p><b>${esc(tx.neutral.headline)}</b><br>${esc(tx.neutral.body)}</p></details>` : "";
  $("story").innerHTML = `
    <div class="meta">${esc(info.clock)} · ${esc(TYPE_LABEL[info.type] || info.type)}</div>
    <h2 lang="${esc(v.language)}">${esc(v.headline)}</h2>
    <div class="body" lang="${esc(v.language)}">${esc(v.body)}</div>
    ${why}
    <div class="badges">${badges(v)}</div>
    ${curveBox(info, tx.facts)}
    ${factsTable(tx.facts)}
    ${neutral}
    ${rejected}`;
  $("caption").textContent = `${info.clock}  ${TYPE_LABEL[info.type] || info.type}`;
  renderLegend(info);
}

function renderLegend(info) {
  const pass = ["press_trap", "press_broken", "missed_option"].includes(info.type);
  $("legend").innerHTML = pass ? `
    <span><i class="l-ring"></i>pressure on the passer (5 m)</span>
    <span><i class="l-opt"></i>passing options, coloured by xPass</span>
    <span><i class="l-chosen"></i>chosen pass</span>
    <span><i class="l-best"></i>best option</span>` : "";
}

function renderRecap() {
  const r = state.match.recaps[state.viewer];
  $("tab-recap").innerHTML = r ? `<div class="recap"><h3 lang="${esc(r.language)}">${esc(r.headline)}</h3>
    <p lang="${esc(r.language)}">${esc(r.body)}</p><div class="badges">${badges(r)}</div></div>` : "";
}

function renderAnswers() {
  $("tab-answers").innerHTML = state.data.answers.map((a) => `<div class="qa">
    <div class="q">Q: ${esc(a.headline)}</div><div class="a">${esc(a.body)}</div>
    <div class="badges">${badges(a)}</div></div>`).join("");
}

// ------------------------------------------------------------------ controls

function bindControls() {
  $("play").onclick = () => { state.playing = !state.playing; $("play").textContent = state.playing ? "❚❚" : "▶"; };
  $("replay").onclick = () => selectMoment(state.moment);
  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("live-play").onclick = () => {
    state.live.playing = !state.live.playing;
    $("live-play").textContent = state.live.playing ? "❚❚" : "▶";
  };
  $("live-speed").onchange = (e) => (state.live.speed = +e.target.value);
  $("follow").onchange = (e) => setFollow(e.target.value || null);
  let tipHidden = false;
  try { tipHidden = localStorage.getItem("ts-follow-tip") === "hidden"; } catch { /* storage unavailable */ }
  $("follow-tip").classList.toggle("hidden", tipHidden);
  $("follow-tip-x").onclick = () => {
    $("follow-tip").classList.add("hidden");
    try { localStorage.setItem("ts-follow-tip", "hidden"); } catch { /* storage unavailable */ }
  };
  $("follow-tip-go").onclick = () => {
    const label = $("follow-label");
    label.scrollIntoView({ behavior: "smooth", block: "center" });
    label.classList.remove("flash"); void label.offsetWidth; label.classList.add("flash");
    $("follow").focus();
  };
  $("live-scrub").oninput = (e) => liveSeek((+e.target.value / 1000) * state.match.end_frame);
  $("scrub").oninput = (e) => { state.t = +e.target.value; state.holdUntil = 0; };
  $("speed").onchange = (e) => (state.speed = +e.target.value);
  $("slowmo").onchange = (e) => (state.slowmo = e.target.checked);
  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${b.dataset.tab}`));
  }));
  window.addEventListener("resize", () => draw());
}

// ------------------------------------------------------------------ animation

function loop(now) {
  const dt = Math.min(0.1, (now - (state.last || now)) / 1000);
  state.last = now;
  if (state.mode === "live") {
    liveTick(dt);
    draw();
    requestAnimationFrame(loop);
    return;
  }
  const clip = state.clip;
  if (clip && state.playing && now >= state.holdUntil) {
    const n = clip.xy.length - 1;
    let rate = state.speed;
    const near = clip.moments.some((m) => Math.abs(state.t - m.index) < SLOW_WINDOW);
    if (state.slowmo && near) rate *= 0.3;
    const next = state.t + dt * clip.fps * rate;
    for (const m of clip.moments) {
      if (state.t < m.index && next >= m.index && !state.held.has(m.moment_id)) {
        state.held.add(m.moment_id);
        state.t = m.index;
        if (state.slowmo) state.holdUntil = now + HOLD_MS;
        if (m.moment_id !== state.moment) { state.moment = m.moment_id; renderTimeline(); renderStory(); }
        break;
      }
    }
    if (now >= state.holdUntil) state.t = Math.min(next, n);
    if (state.t >= n) { state.holdUntil = now + 1500; state.t = 0; state.held = new Set(); }
    $("scrub").value = Math.round(state.t);
  }
  draw();
  requestAnimationFrame(loop);
}

function frameXY(clip, t) {
  const i = Math.floor(t), j = Math.min(i + 1, clip.xy.length - 1), a = t - i;
  const A = clip.xy[i], B = clip.xy[j];
  const pts = [];
  for (let k = 0; k < 23; k++) {
    pts.push([(A[2 * k] * (1 - a) + B[2 * k] * a) / 10, (A[2 * k + 1] * (1 - a) + B[2 * k + 1] * a) / 10]);
  }
  return { pts, z: clip.ball_z[i] * (1 - a) + clip.ball_z[j] * a };
}

function setupCanvas() {
  const c = $("pitch");
  const w = c.clientWidth, h = Math.round(w * 74 / 111);
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`;
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const s = w / 111;
  return { ctx, w, h, s, X: (x) => (x + 55.5) * s, Y: (y) => (37 - y) * s };
}

function drawPitch(g) {
  const { ctx, w, h, s, X, Y } = g;
  for (let i = 0; i < 12; i++) {
    ctx.fillStyle = i % 2 ? "#2f7a3a" : "#2c7336";
    ctx.fillRect((i * w) / 12, 0, w / 12 + 1, h);
  }
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.5;
  const rect = (x0, y0, x1, y1) => ctx.strokeRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s);
  rect(-52.5, -34, 52.5, 34);
  ctx.beginPath(); ctx.moveTo(X(0), Y(34)); ctx.lineTo(X(0), Y(-34)); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(0), Y(0), 9.15 * s, 0, 2 * Math.PI); ctx.stroke();
  for (const d of [-1, 1]) {
    const gx = 52.5 * d;
    rect(Math.min(gx, gx - 16.5 * d), -20.16, Math.max(gx, gx - 16.5 * d), 20.16);
    rect(Math.min(gx, gx - 5.5 * d), -9.16, Math.max(gx, gx - 5.5 * d), 9.16);
    ctx.fillStyle = "rgba(255,255,255,.9)";
    ctx.beginPath(); ctx.arc(X(gx - 11 * d), Y(0), 2, 0, 2 * Math.PI); ctx.fill();
    ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(gx + 1.2 * d), Y(3.66)); ctx.lineTo(X(gx + 1.2 * d), Y(-3.66)); ctx.stroke();
    ctx.lineWidth = 1.5;
  }
  ctx.setLineDash([4, 6]); ctx.strokeStyle = "rgba(255,255,255,.22)";
  for (const x of [-17.5, 17.5]) { ctx.beginPath(); ctx.moveTo(X(x), Y(34)); ctx.lineTo(X(x), Y(-34)); ctx.stroke(); }
  ctx.setLineDash([]);
}

function label(ctx, text, x, y, color = "#fff") {
  ctx.font = "600 12px system-ui, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(13,17,23,.82)";
  ctx.fillRect(x - w / 2, y - 9, w, 18);
  ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}

function xpassColor(p) { return `hsl(${Math.round(120 * p)}, 75%, 52%)`; }

function activeOverlay(clip, t) {
  let best = null;
  for (const m of clip.moments) {
    if (t >= m.index - OVERLAY_IN && t <= m.index + OVERLAY_OUT) best = m;
  }
  if (!best) return null;
  const d = t - best.index;
  const alpha = d < 0 ? 1 + d / OVERLAY_IN : d > OVERLAY_OUT - 8 ? (OVERLAY_OUT - d) / 8 : 1;
  return { m: best, alpha: Math.max(0, Math.min(1, alpha)) };
}

function drawOverlay(g, clip, pts, ov) {
  const { ctx, s, X, Y } = g;
  const o = ov.m.overlay;
  ctx.save(); ctx.globalAlpha = ov.alpha;
  const P = (k) => [X(pts[k][0]), Y(pts[k][1])];
  if (o.kind === "pass" && o.passer >= 0) {
    const [px, py] = P(o.passer);
    ctx.fillStyle = `rgba(248,81,73,${0.12 + 0.3 * o.pressure})`;
    ctx.strokeStyle = "rgba(248,81,73,.9)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(px, py, 5 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    const opts = [...o.options].sort((a, b) => (a.chosen || a.best) - (b.chosen || b.best));
    for (const op of opts) {
      if (op.slot < 0) continue;
      const [rx, ry] = P(op.slot);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(rx, ry);
      if (op.chosen) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.setLineDash([]); }
      else if (op.best) { ctx.strokeStyle = "#f2c14e"; ctx.lineWidth = 3.5; ctx.setLineDash([9, 6]); }
      else { ctx.strokeStyle = xpassColor(op.xpass); ctx.lineWidth = 1.6; ctx.setLineDash([]); ctx.globalAlpha = 0.7 * ov.alpha; }
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = ov.alpha;
    }
    for (const op of o.options) {
      if (op.slot < 0 || !(op.chosen || op.best)) continue;
      const [rx, ry] = P(op.slot);
      const tag = op.chosen && op.best ? "Chosen = best" : op.chosen ? "Chosen" : "Best option";
      const extra = op.chosen && o.distance_m ? ` · ${Math.round(o.distance_m)} m · ${o.ball_speed_kmh} km/h` : "";
      label(ctx, `${tag}: ${op.name.split(" ").slice(-1)[0]} · xPass ${pct(op.xpass)}${extra}`, rx, ry - 18,
        op.chosen ? "#fff" : "#f2c14e");
    }
    if (o.presser >= 0) {
      const [dx, dy] = P(o.presser);
      ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(dx, dy); ctx.stroke(); ctx.setLineDash([]);
      label(ctx, `${o.presser_m} m`, (px + dx) / 2, (py + dy) / 2 - 12);
    }
    label(ctx, `${clip.objects[o.passer].name.split(" ").slice(-1)[0]} · pressure ${o.pressure}`, px, py + 5 * s + 12, "#ffb4ae");
  } else if (o.kind === "shot" && o.shooter >= 0) {
    const [sx, sy] = P(o.shooter);
    const gx = pts[o.shooter][0] > 0 ? 52.5 : -52.5;
    ctx.strokeStyle = "#f2c14e"; ctx.lineWidth = 3; ctx.setLineDash([8, 5]);
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(X(gx), Y(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(sx, sy, 2.6 * s, 0, 2 * Math.PI); ctx.stroke();
    label(ctx, `${clip.objects[o.shooter].name.split(" ").slice(-1)[0]} · xG ${o.xg}${o.ball_speed_kmh ? ` · ${o.ball_speed_kmh} km/h` : ""}`,
      sx, sy - 3.4 * s - 8, "#f2c14e");
  }
  ctx.restore();
}

function draw() {
  const clip = state.clip;
  const g = setupCanvas();
  drawPitch(g);
  if (state.mode === "live") { drawLive(g); return; }
  if (!clip) return;
  const { ctx, s, X, Y } = g;
  const { pts, z } = frameXY(clip, state.t);
  const ov = activeOverlay(clip, state.t);
  if (ov) drawOverlay(g, clip, pts, ov);
  const cols = [clip.home, clip.away];
  for (let k = 0; k < 22; k++) {
    const team = cols[clip.objects[k].team];
    const x = X(pts[k][0]), y = Y(pts[k][1]);
    ctx.fillStyle = team.color; ctx.strokeStyle = team.secondary; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 1.25 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.fillStyle = team.secondary; ctx.font = `700 ${Math.max(8, Math.round(1.2 * s))}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(clip.objects[k].number ?? "", x, y + 0.5);
  }
  const bx = X(pts[22][0]), by = Y(pts[22][1]);
  ctx.fillStyle = "rgba(0,0,0,.35)";
  ctx.beginPath(); ctx.arc(bx + z * s * 0.4, by + z * s * 0.4, 0.6 * s, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(bx, by, (0.6 + z * 0.08) * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
}

// ------------------------------------------------------------------ live mode

async function loadLive() {
  const text = await (await fetch(`data/live/${state.matchId}.jsonl`)).text();
  const L = state.live;
  L.feed = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  L.track = await (await fetch(`data/live/${state.matchId}_tracking.json`)).json();
  L.slotHistory = Array.from({ length: 22 }, () => []);
  for (const c of L.track.slots) L.slotHistory[c.slot].push(c);
  computeKinematics();
  const byTeam = {};
  for (const c of L.track.slots) (byTeam[c.player_id.split("-")[0]] ||= new Map()).set(c.player_id, c);
  $("follow").innerHTML = `<option value="">nobody</option>` + [state.match.home.club_id, state.match.away.club_id]
    .map((t) => `<optgroup label="${esc(teamName(t))}">${[...(byTeam[t] || new Map()).values()]
      .sort((a, b) => (a.number ?? 99) - (b.number ?? 99))
      .map((c) => `<option value="${esc(c.player_id)}">${c.number ?? ""} ${esc(c.name)}</option>`).join("")}</optgroup>`).join("");
  const end = state.match.end_frame;
  const keys = new Set(state.match.key_moments);
  $("live-marks").innerHTML = L.feed.filter((m) => m.kind === "moment").map((m) =>
    `<i class="${keys.has(m.moment_id) ? "key" : ""} ${m.type === "goal" ? "goal" : ""}" style="left:${(100 * m.frame) / end}%" title="${esc(m.clock)} ${esc(TYPE_LABEL[m.type] || m.type)}"></i>`).join("");
  liveReset();
}

function liveReset() {
  Object.assign(state.live, { i: 0, frame: 0, st: null, ticker: [], trail: [], moments: [], dirty: true, seen: 0,
    lastPass: null, fstats: emptyStats() });
}

function liveApply(msg) {
  const L = state.live;
  if (msg.kind === "state") L.st = msg;
  else if (msg.kind === "moment") L.moments.push(msg);
  else if (msg.kind === "event") {
    const e = msg.event;
    if (e.x !== undefined && e.y !== undefined && e.type !== "possession") {
      L.trail.push({ x: e.end_x ?? e.x, y: e.end_y ?? e.y, x0: e.x, y0: e.y, team: e.team, type: e.type, frame: msg.frame });
      if (L.trail.length > TRAIL) L.trail.shift();
    }
    if (NOTABLE.has(e.type)) { L.ticker.unshift({ clock: msg.clock, ...tickerText(e, msg.metrics) }); L.ticker.length = Math.min(L.ticker.length, 12); }
    if ((e.type === "pass" || e.type === "shot") && msg.metrics) L.lastPass = { e, m: msg.metrics, wall: performance.now(), frame: msg.frame };
    if (L.follow && e.player_id === L.follow) addStats(L.fstats, e, msg.metrics);
  }
  L.dirty = true;
}

function liveAdvance(frame) {
  const L = state.live;
  while (L.i < L.feed.length && (L.feed[L.i].frame ?? 0) <= frame) liveApply(L.feed[L.i++]);
  L.frame = frame;
}

function liveSeek(frame) {
  if (frame < state.live.frame) liveReset();
  liveAdvance(frame);
}

function liveTick(dt) {
  const L = state.live;
  if (L.playing) {
    const end = state.match.end_frame;
    liveAdvance(Math.min(end, L.frame + dt * 10 * L.speed));
    if (L.frame >= end) { L.playing = false; $("live-play").textContent = "↺"; }
    $("live-scrub").value = Math.round((1000 * L.frame) / end);
  }
  if (L.dirty) { renderLive(); L.dirty = false; }
}

const teamName = (code) => state.match.colors[code]?.name || code || "";
const playerName = (pid) => state.match.players[pid] || pid || "";

function tickerText(e, q = {}) {
  const who = playerName(e.player_id), team = teamName(e.team);
  switch (e.type) {
    case "goal": return { text: `⚽ GOAL! ${who} (${team}) · ${e.score || ""}`, cls: "goal" };
    case "shot": return { text: `Shot: ${who} (${team})${q?.ball_speed_kmh ? `, ${q.ball_speed_kmh} km/h` : ""}${e.provider_xg != null ? `, xG ${e.provider_xg.toFixed(2)}` : ""}` };
    case "save": return { text: `Save: ${who} (${team})` };
    case "foul": return { text: `Foul by ${who} (${team})` };
    case "corner": return { text: `Corner to ${team}` };
    case "substitution": return { text: `Substitution ${team}: ${who} ↔ ${playerName(e.other_player_id)}` };
    case "period_start": return { text: e.period === 1 ? "Kick-off" : "Second half under way" };
    case "period_end": return { text: e.period === 1 ? "Half-time" : "Full-time" };
    default: return { text: e.type };
  }
}

function renderLive() {
  const L = state.live, m = state.match;
  const clock = L.st?.clock || "0'";
  $("live-clock").textContent = L.frame >= m.end_frame ? "FT" : clock;
  $("caption").textContent = `${clock} · Live`;
  $("scoreline").innerHTML = `${esc(m.home.name)}<span class="score">${esc(L.st?.score || "0-0")}</span>${esc(m.away.name)}`;
  const h = m.home.club_id, a = m.away.club_id;
  const T = L.st?.teams || {};
  const rows = [
    ["Possession", "possession_pct", (v) => `${v ?? 50}%`],
    ["Shots", "shots", (v) => v ?? 0],
    ["Chance quality (xG)", "xg", (v) => (v ?? 0).toFixed(2)],
    ["Pass completion", "completion_pct", (v) => (v == null ? "–" : `${v}%`)],
    ["Threat created (xT)", "xt_created", (v) => (v ?? 0).toFixed(2)],
    ["Pressures applied", "pressures", (v) => v ?? 0],
    ["Pressure faced per pass", "pressure_faced", (v) => (v == null ? "–" : v.toFixed(2))],
    ["Momentum, last 5 min (xT)", "momentum_5min", (v) => (v ?? 0).toFixed(2)],
    ["Distance covered (km)", "distance_km", (v) => (v ?? 0).toFixed(1)],
  ];
  if (L.kin) {
    const i = sampleIndex();
    for (const [code, t] of [[h, 0], [a, 1]]) T[code] = { ...(T[code] || {}), distance_km: L.kin.team[t][i] / 1000 };
  }
  const ch = m.colors[h].color, ca = m.colors[a].color;
  $("stats").innerHTML = `<tr><th>${esc(m.colors[h].name)}</th><th></th><th>${esc(m.colors[a].name)}</th></tr>` +
    rows.map(([label, key, f]) => {
      const vh = T[h]?.[key], va = T[a]?.[key];
      const nh = Math.max(+vh || 0, 0), na = Math.max(+va || 0, 0), tot = nh + na || 1;
      return `<tr><td class="v">${esc(f(vh))}</td><td class="k">${label}
        <div class="bar"><span style="width:${(100 * nh) / tot}%;background:${ch}"></span><span style="width:${(100 * na) / tot}%;background:${ca}"></span></div></td>
        <td class="v">${esc(f(va))}</td></tr>`;
    }).join("");
  renderPlayerCard();
  const viewer = m.viewers.find((v) => v.id === state.viewer);
  $("alerts-for").textContent = `for ${viewer.name}`;
  const mine = L.moments.filter((x) => x.for.includes(state.viewer));
  $("alerts").innerHTML = mine.length ? mine.slice().reverse().map((x, i) => {
    const v = viewerText(x.moment_id, state.viewer).version;
    return `<button class="alert ${i === 0 && mine.length > L.seen ? "new" : ""}" data-id="${x.moment_id}">
      <div class="m">${esc(x.clock)} · ${esc(TYPE_LABEL[x.type] || x.type)} · ${esc(teamName(x.team))}</div>
      <div class="h" lang="${esc(v.language)}">${esc(v.headline)}</div></button>`;
  }).join("") : `<div class="empty">No alerts yet. They appear here as your moments happen.</div>`;
  L.seen = mine.length;
  $("alerts").querySelectorAll(".alert").forEach((b) => (b.onclick = () => { setMode("moments"); selectMoment(b.dataset.id); }));
  $("ticker").innerHTML = L.ticker.slice(0, 8).map((t) => `<div class="${t.cls || ""}"><span class="t">${esc(t.clock)}</span>${esc(t.text)}</div>`).join("");
}

/** Speed, distance, top speed and sprints per slot, and distance per team, from the 5 Hz tracking. */
function computeKinematics() {
  const L = state.live, T = L.track, n = T.xy.length, dt = T.step / 10;
  const kin = { speed: [], cum: [], top: [], sprints: [], team: [new Float32Array(n), new Float32Array(n)] };
  for (let k = 0; k < 22; k++) {
    const speed = new Float32Array(n), cum = new Float32Array(n), top = new Float32Array(n), spr = new Uint16Array(n);
    const changes = new Set(L.slotHistory[k].slice(1).map((c) => Math.round(c.frame / T.step)));
    for (let i = 1; i < n; i++) {
      if (changes.has(i)) continue;              // a substitute starts from zero
      const dx = (T.xy[i][2 * k] - T.xy[i - 1][2 * k]) / 10, dy = (T.xy[i][2 * k + 1] - T.xy[i - 1][2 * k + 1]) / 10;
      let d = Math.hypot(dx, dy);
      if (d > MAX_STEP_M) d = 0;
      speed[i] = 0.5 * (d / dt) + 0.5 * speed[i - 1];
      cum[i] = cum[i - 1] + d;
      top[i] = Math.max(top[i - 1], speed[i]);
      spr[i] = spr[i - 1] + (speed[i] > SPRINT_MS && speed[i - 1] <= SPRINT_MS ? 1 : 0);
    }
    kin.speed.push(speed); kin.cum.push(cum); kin.top.push(top); kin.sprints.push(spr);
  }
  for (let i = 1; i < n; i++) {
    for (let t = 0; t < 2; t++) {
      let d = 0;
      for (let k = 11 * t; k < 11 * t + 11; k++) d += Math.max(0, kin.cum[k][i] - kin.cum[k][i - 1]);
      kin.team[t][i] = kin.team[t][i - 1] + d;
    }
  }
  L.kin = kin;
}

function sampleIndex() {
  const T = state.live.track;
  return Math.min(Math.floor(state.live.frame / T.step), T.xy.length - 1);
}

function emptyStats() { return { passes: 0, completed: 0, xpass: 0, nx: 0, xt: 0, pressure: 0, shots: 0, xg: 0, goals: 0 }; }

function addStats(f, e, q = {}) {
  if (e.type === "pass") {
    f.passes++; f.completed += e.outcome === "complete" ? 1 : 0;
    if (q?.xpass != null) { f.xpass += q.xpass; f.pressure += q.pressure ?? 0; f.nx++; }
    f.xt += Math.max(0, q?.xt_added ?? 0);
  } else if (e.type === "shot") { f.shots++; f.xg += e.provider_xg ?? 0; }
  else if (e.type === "goal") f.goals++;
}

function setFollow(pid) {
  const L = state.live;
  L.follow = pid;
  $("follow").value = pid || "";
  if (pid) $("follow-tip").classList.add("hidden");
  L.fstats = emptyStats();
  if (pid) for (let j = 0; j < L.i; j++) {
    const msg = L.feed[j];
    if (msg.kind === "event" && msg.event.player_id === pid) addStats(L.fstats, msg.event, msg.metrics);
  }
  L.dirty = true;
}

function followedSlot() {
  const L = state.live;
  if (!L.follow || !L.track) return -1;
  for (let k = 0; k < 22; k++) if (slotPlayer(k, L.frame)?.player_id === L.follow) return k;
  return -1;
}

function renderPlayerCard() {
  const L = state.live, card = $("player-card");
  card.classList.toggle("hidden", !L.follow);
  if (!L.follow) return;
  const f = L.fstats, k = followedSlot(), i = sampleIndex();
  const kmh = (ms) => Math.round(ms * 3.6);
  const items = [
    ["Passes", `${f.completed}/${f.passes}`],
    ["Accuracy", f.passes ? `${Math.round((100 * f.completed) / f.passes)}%` : "–"],
    ["Avg difficulty (xPass)", f.nx ? `${Math.round((100 * f.xpass) / f.nx)}%` : "–"],
    ["Threat created (xT)", f.xt.toFixed(2)],
    ["Pressure faced", f.nx ? (f.pressure / f.nx).toFixed(2) : "–"],
    ["Shots · goals", `${f.shots} · ${f.goals}`],
    ["Distance", k >= 0 ? `${(L.kin.cum[k][i] / 1000).toFixed(1)} km` : "–"],
    ["Top speed", k >= 0 ? `${kmh(L.kin.top[k][i])} km/h` : "–"],
    ["Sprints (>25 km/h)", k >= 0 ? L.kin.sprints[k][i] : "–"],
  ];
  card.innerHTML = `<h4>Following ${esc(playerName(L.follow))} <small>${esc(teamName(L.follow.split("-")[0]))}${k < 0 ? " · not on the pitch" : ""}</small></h4>
    <dl>${items.map(([a, b]) => `<div><dt>${a}</dt><dd>${esc(b)}</dd></div>`).join("")}</dl>`;
}

/** Who plays in a slot at a frame (substitutes take over a slot). */
function slotPlayer(k, frame) {
  const hist = state.live.slotHistory[k];
  let cur = hist[0];
  for (const c of hist) { if (c.frame <= frame) cur = c; else break; }
  return cur;
}

function drawLivePlayers(g) {
  const { ctx, s, X, Y } = g;
  const L = state.live, T = L.track, cols = state.match.colors;
  const pos = L.frame / T.step, i = Math.min(Math.floor(pos), T.xy.length - 1), j = Math.min(i + 1, T.xy.length - 1);
  const a = pos - i, A = T.xy[i], B = T.xy[j];
  const P = (k) => [(A[2 * k] * (1 - a) + B[2 * k] * a) / 10, (A[2 * k + 1] * (1 - a) + B[2 * k + 1] * a) / 10];
  const teams = [cols[state.match.home.club_id], cols[state.match.away.club_id]];
  for (let k = 0; k < 22; k++) {
    const [x, y] = P(k), team = teams[k < 11 ? 0 : 1], who = slotPlayer(k, L.frame);
    ctx.fillStyle = team.color; ctx.strokeStyle = team.secondary; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(X(x), Y(y), 1.25 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.fillStyle = team.secondary; ctx.font = `700 ${Math.max(8, Math.round(1.2 * s))}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(who?.number ?? "", X(x), Y(y) + 0.5);
  }
  const si = sampleIndex(), kmh = (k) => Math.round(L.kin.speed[k][si] * 3.6);
  for (let k = 0; k < 22; k++) {                       // sprint indicators
    if (L.kin.speed[k][si] > SPRINT_MS) {
      const [x, y] = P(k);
      label(ctx, `⚡${kmh(k)}`, X(x) + 1.25 * s + 14, Y(y), "#f2c14e");
    }
  }
  const lp = L.lastPass;
  if (lp && (performance.now() - lp.wall < PASS_LABEL_MS || L.frame - lp.frame < 30)) {
    const e = lp.e, q = lp.m;
    ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(X(e.x), Y(e.y)); ctx.lineTo(X(e.end_x), Y(e.end_y)); ctx.stroke(); ctx.setLineDash([]);
    const txt = e.type === "shot" ? `Shot ${q.ball_speed_kmh ?? "?"} km/h · xG ${(e.provider_xg ?? 0).toFixed(2)}`
      : `Pass ${Math.round(q.distance_m)} m · ${q.ball_speed_kmh ?? "?"} km/h${q.difficulty ? ` · ${q.difficulty} (xPass ${pct(q.xpass)})` : ""}`;
    label(ctx, txt, (X(e.x) + X(e.end_x)) / 2, (Y(e.y) + Y(e.end_y)) / 2 - 12);
  }
  const carrier = T.carrier[i];
  if (carrier >= 0) {
    const [x, y] = P(carrier), who = slotPlayer(carrier, L.frame);
    label(ctx, `${who?.name || ""} · ${kmh(carrier)} km/h · ${(L.kin.cum[carrier][si] / 1000).toFixed(1)} km`, X(x), Y(y) - 1.25 * s - 10);
  }
  const fk = followedSlot();
  if (fk >= 0 && fk !== carrier) {
    const [x, y] = P(fk);
    label(ctx, `${playerName(L.follow)} · ${kmh(fk)} km/h · ${(L.kin.cum[fk][si] / 1000).toFixed(1)} km`, X(x), Y(y) - 1.25 * s - 10, "#f2c14e");
  }
  if (fk >= 0) {
    const [x, y] = P(fk);
    ctx.strokeStyle = "#f2c14e"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(X(x), Y(y), 2.2 * s, 0, 2 * Math.PI); ctx.stroke();
  }
  const [bx, by] = P(22);
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(X(bx), Y(by), 0.65 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
}

function drawLive(g) {
  const { ctx, s, X, Y } = g;
  const L = state.live, cols = state.match.colors;
  if (L.track) {
    drawLivePlayers(g);
    drawLiveBanner(g);
    return;
  }
  L.trail.forEach((p, k) => {
    const age = (L.trail.length - 1 - k) / TRAIL;
    ctx.globalAlpha = 0.85 * (1 - age);
    const c = cols[p.team]?.color || "#ccc";
    if (["pass", "carry", "shot", "clearance"].includes(p.type) && p.x0 !== undefined) {
      ctx.strokeStyle = c; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(p.x0), Y(p.y0)); ctx.lineTo(X(p.x), Y(p.y)); ctx.stroke();
    }
    ctx.fillStyle = c; ctx.strokeStyle = cols[p.team]?.secondary || "#fff"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(X(p.x0 ?? p.x), Y(p.y0 ?? p.y), 0.9 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  });
  ctx.globalAlpha = 1;
  const last = L.trail[L.trail.length - 1];
  if (last) {
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(X(last.x), Y(last.y), 0.75 * s, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  drawLiveBanner(g);
}

function drawLiveBanner(g) {
  const L = state.live;
  const recent = L.moments[L.moments.length - 1];
  if (recent && L.frame - recent.frame < 80 && recent.for.includes(state.viewer)) {
    label(g.ctx, `${recent.clock} ${TYPE_LABEL[recent.type] || recent.type}: see your alert →`, g.w / 2, 24, "#f2c14e");
  }
}

function setMode(mode) {
  state.mode = mode;
  const live = mode === "live";
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("live-controls").classList.toggle("hidden", !live);
  $("ticker").classList.toggle("hidden", !live);
  $("live").classList.toggle("hidden", !live);
  for (const id of ["moment-controls", "legend", "timeline", "story"]) $(id).classList.toggle("hidden", live);
  if (live) state.live.dirty = true;
  else {
    const m = state.match;
    $("scoreline").innerHTML = `${esc(m.home.name)}<span class="score">${esc(m.score)}</span>${esc(m.away.name)}`;
    renderStory();
  }
}

init();
