// MLB Edge site behavior: sortable tables, the History day picker, and the
// Accuracy charts (Chart.js, loaded via CDN on that page). Same patterns as
// NFL Edge's web/site.js.

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function makeSortable(table) {
  const tbody = table.tBodies[0];
  table.querySelectorAll("th[data-sort-key]").forEach(th => {
    th.tabIndex = 0;
    th.setAttribute("aria-sort", "none");
    th.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); th.click(); }
    });
    th.addEventListener("click", () => {
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const asc = !th.classList.contains("sorted-asc");
      table.querySelectorAll("th").forEach(h => {
        h.classList.remove("sorted-asc", "sorted-desc");
        if (h.hasAttribute("aria-sort")) h.setAttribute("aria-sort", "none");
      });
      th.classList.add(asc ? "sorted-asc" : "sorted-desc");
      th.setAttribute("aria-sort", asc ? "ascending" : "descending");

      const key = th.dataset.sortKey;
      const isNumeric = th.classList.contains("num");
      const val = r => r.querySelector(`[data-key="${key}"]`)?.dataset.value ?? "";
      rows.sort((a, b) => {
        if (isNumeric) {
          // Blank values (no stat yet) always sort last.
          const av = parseFloat(val(a)), bv = parseFloat(val(b));
          if (isNaN(av)) return 1;
          if (isNaN(bv)) return -1;
          return asc ? av - bv : bv - av;
        }
        return asc ? val(a).localeCompare(val(b)) : val(b).localeCompare(val(a));
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
}
document.querySelectorAll("table.data[data-sortable]").forEach(makeSortable);

// --- History day picker ---
function resultPill(p) {
  if (p.void) return "<span class='pill pill-void'>NO DECISION</span>";
  if (p.got_hit === null) return "<span class='faint'>Pending</span>";
  const line = `${p.hits}-for-${p.at_bats}`;
  return p.got_hit
    ? `${line} <span class='pill pill-positive'>HIT</span>`
    : `${line} <span class='pill pill-danger'>MISS</span>`;
}

function initHistoryPicker() {
  if (typeof HISTORY_DATA === "undefined") return;
  const picker = document.getElementById("day-select");
  const container = document.getElementById("day-content");
  if (!picker || !container) return;

  function render(day) {
    const d = HISTORY_DATA.days[day];
    let rows = "";
    d.picks.forEach(p => {
      const prob = p.confidence === null ? "<span class='faint'>-</span>" : `${p.confidence.toFixed(0)}%`;
      rows += `<tr>
        <td><div class="player-name">${esc(p.player_name)}</div><div class="player-meta">${esc(p.matchup)}</div></td>
        <td class="num prob" data-label="Hit chance">${prob}</td>
        <td class="num" data-label="Result"><span>${resultPill(p)}</span></td>
      </tr>`;
    });
    const s = d.summary;
    const note = d.legacy
      ? "<div class='table-footnote'>Picked by the old weighted-score model, which didn't produce a real hit chance.</div>"
      : "";
    container.innerHTML = `<div class="section-label">${esc(d.label)}: ${s.hits} of ${s.graded} got a hit${s.voided ? `, ${s.voided} no decision` : ""}</div>
      <table class="data responsive-stack">
        <thead><tr><th>Player</th><th class="num">Hit chance</th><th class="num">Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>${note}`;
  }

  HISTORY_DATA.order.forEach(day => {
    const opt = document.createElement("option");
    opt.value = day;
    opt.textContent = HISTORY_DATA.days[day].label;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => render(picker.value));
  picker.value = HISTORY_DATA.order[0];
  render(picker.value);
}
initHistoryPicker();

// --- Accuracy charts ---
function initCharts() {
  if (typeof ACCURACY_DATA === "undefined") return;
  if (typeof Chart === "undefined") {
    document.querySelectorAll(".chart-card").forEach(c => c.dataset.state = "failed");
    return;
  }
  Chart.defaults.font.family = "'Barlow', 'Helvetica Neue', Arial, sans-serif";
  Chart.defaults.animation = false;
  const pct = v => (v * 100).toFixed(0) + "%";

  function lineChart(canvasId, title, series) {
    const el = document.getElementById(canvasId);
    if (!el) return;
    el.closest(".chart-card")?.setAttribute("data-state", "ready");
    new Chart(el, {
      type: "line",
      data: {
        labels: ACCURACY_DATA.labels,
        datasets: series.map(s => ({
          label: s.label, data: s.data, borderColor: s.color, backgroundColor: s.color,
          borderDash: s.dash || [], pointRadius: 3, borderWidth: 2, tension: 0,
        })),
      },
      options: {
        plugins: {
          title: { display: true, text: title, align: "start", font: { size: 15, weight: "bold" }, color: "#ecebe7" },
          legend: { display: true, position: "top", align: "start", labels: { color: "#ecebe7", font: { size: 13 }, boxWidth: 12, boxHeight: 2 } },
          tooltip: {
            backgroundColor: "#1a1b1d", borderColor: "#45484e", borderWidth: 1, cornerRadius: 6,
            titleColor: "#ecebe7", bodyColor: "#a8a7a1",
            callbacks: { label: ctx => `${ctx.dataset.label}: ${pct(ctx.parsed.y)}` },
          },
        },
        scales: {
          y: { ticks: { color: "#a8a7a1", callback: pct }, grid: { color: "#2b2d31" }, border: { display: false } },
          x: { ticks: { color: "#a8a7a1" }, grid: { display: false }, border: { color: "#45484e" } },
        },
      },
    });
  }

  // NBA Edge passes its own chart titles in ACCURACY_DATA.titles.
  const t = ACCURACY_DATA.titles || {};
  lineChart("chart-weekly", t.weekly || "Hit Rate by Week: Picks vs. What the Model Predicted", [
    { label: t.weekly_actual || "Actual hit rate", data: ACCURACY_DATA.actual, color: "#e5793b" },
    { label: t.weekly_predicted || "Model's predicted hit rate", data: ACCURACY_DATA.predicted, color: "#a8a7a1", dash: [4, 4] },
  ]);
  lineChart("chart-cumulative", t.cumulative || "Season-to-Date Hit Rate", [
    { label: "Actual, season to date", data: ACCURACY_DATA.cumulative_actual, color: "#e5793b" },
    { label: "Predicted, season to date", data: ACCURACY_DATA.cumulative_predicted, color: "#a8a7a1", dash: [4, 4] },
  ]);
}
initCharts();

// --- Scoreboard strip (shared by every Edge site) ---
// Fills <div class="scoreboard"> under the top bar with the latest games in
// every sport, ESPN style: live games first, then what's next, then recent
// finals, each with our pick; a game opens the Schedule tab (schedule.html
// and schedule.js on the home site). Each site's build publishes games.json (ESPN's
// current slate plus our picks) next to its summary.json; the strip then asks
// ESPN for fresh scores in the browser, refreshes every minute while a game is
// live, and keeps the published file if ESPN can't be reached. If no sport has
// games, it falls back to each site's top picks from summary.json. Keep this
// block identical in home.js (ant56-arch.github.io), web/site.js (nfl-edge)
// and web/site.js (mlb-hit-predictor).
const EDGE_SITES = [
  { sport: "NFL", summary: "/nfl-edge/nfl/summary.json", games: "/nfl-edge/nfl/games.json",
    href: "/nfl-edge/nfl/index.html", schedule: "/schedule.html#nfl" },
  { sport: "CFB", summary: "/nfl-edge/cfb/summary.json", games: "/nfl-edge/cfb/games.json",
    href: "/nfl-edge/cfb/index.html", schedule: "/schedule.html#cfb" },
  { sport: "MLB", summary: "/mlb-hit-predictor/summary.json", games: "/mlb-hit-predictor/games.json",
    href: "/mlb-hit-predictor/", schedule: "/schedule.html#mlb" },
  { sport: "NBA", summary: "/mlb-hit-predictor/nba/summary.json", games: "/mlb-hit-predictor/nba/games.json",
    href: "/mlb-hit-predictor/nba/index.html", schedule: "/schedule.html#nba" },
];
const EDGE_GAMES_PER_SPORT = 16;

function edgeFetchJson(url, ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms || 8000);
  return fetch(url, { cache: "no-cache", signal: ctrl.signal })
    .then(r => (r.ok ? r.json() : null))
    .catch(() => null)
    .finally(() => clearTimeout(timer));
}

function edgeFetchSummaries() {
  if (!window.edgeSummaries) {
    window.edgeSummaries = Promise.all(EDGE_SITES.map(site => edgeFetchJson(site.summary)));
  }
  return window.edgeSummaries;
}

function edgeNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function edgeResultPill(result, labels) {
  const [yes, no] = labels || ["HIT", "MISS"];
  if (result === true) return edgeNode("span", "pill pill-positive", yes);
  if (result === false) return edgeNode("span", "pill pill-danger", no);
  return null;
}

// ESPN scoreboard event -> the same shape games.py publishes.
function edgeParseEspn(ev) {
  const comp = (ev.competitions || [])[0] || {};
  const sides = {};
  (comp.competitors || []).forEach(c => { sides[c.homeAway] = c; });
  if (!sides.home || !sides.away) return null;
  const type = ((comp.status || ev.status || {}).type) || {};
  const team = c => {
    const t = c.team || {};
    const rank = (c.curatedRank || {}).current;
    const score = c.score === undefined || c.score === "" ? null : Number(c.score);
    const record = (c.records || []).find(r => !r.type || r.type === "total");
    const probable = ((c.probables || [])[0] || {}).athlete;
    return { abbr: t.abbreviation || "", short: t.shortDisplayName || t.name || "", logo: t.logo || "",
             rank: rank >= 1 && rank <= 25 ? rank : null, score: Number.isFinite(score) ? score : null,
             winner: !!c.winner, record: record ? record.summary : null,
             probable: probable ? probable.shortName : null };
  };
  const tv = [];
  (comp.broadcasts || []).forEach(b => (b.names || []).forEach(n => { if (!tv.includes(n)) tv.push(n); }));
  return { id: String(ev.id), start: ev.date, state: type.state || "pre", detail: type.shortDetail || type.detail || "",
           tv: tv.slice(0, 2).join(", "), neutral: !!comp.neutralSite, away: team(sides.away), home: team(sides.home) };
}

async function edgeLoadGames(site) {
  const published = await edgeFetchJson(site.games);
  if (!published) return null;
  const live = published.espn ? await edgeFetchJson(published.espn, 6000) : null;
  if (live && Array.isArray(live.events)) {
    const picks = {};
    (published.games || []).forEach(g => { if (g.pick) picks[g.id] = g.pick; });
    let games = live.events.map(edgeParseEspn).filter(Boolean);
    if (published.top25_only) games = games.filter(g => g.away.rank || g.home.rank);
    games.forEach(g => { if (picks[g.id]) g.pick = picks[g.id]; });
    const week = live.week && live.week.number;
    return { label: week && site.sport !== "MLB" && site.sport !== "NBA" ? `Week ${week}` : published.label,
             games, live: true };
  }
  return { label: published.label, games: published.games || [], live: false };
}

function edgeOrderGames(games) {
  const t = g => new Date(g.start).getTime() || 0;
  const live = games.filter(g => g.state === "in").sort((a, b) => t(a) - t(b));
  const next = games.filter(g => g.state === "pre").sort((a, b) => t(a) - t(b));
  const done = games.filter(g => g.state === "post").sort((a, b) => t(b) - t(a));
  return live.concat(next, done).slice(0, EDGE_GAMES_PER_SPORT);
}

function edgeGameStatus(g) {
  if (g.state !== "pre") return g.detail || (g.state === "post" ? "Final" : "Live");
  const d = new Date(g.start);
  if (isNaN(d)) return "";
  const opts = { timeZone: "America/New_York" };
  const day = d.toLocaleDateString("en-US", { ...opts, weekday: "short" });
  const today = new Date().toLocaleDateString("en-US", { ...opts, weekday: "short" });
  const time = d.toLocaleTimeString("en-US", { ...opts, hour: "numeric", minute: "2-digit" });
  return (day === today ? "" : day + " ") + time + " ET";
}

function edgeGameCell(site, g) {
  const cell = edgeNode("a", "score-cell game-cell" + (g.state === "in" ? " is-live" : ""));
  cell.href = site.schedule;
  const top = edgeNode("span", "score-top");
  top.append(edgeNode("span", "game-status", edgeGameStatus(g)));
  if (g.tv) top.append(edgeNode("span", "game-tv", g.tv));
  cell.append(top);
  [g.away, g.home].forEach(t => {
    const row = edgeNode("span", "game-row" + (g.state === "post" && t.winner ? " is-winner" : ""));
    const name = edgeNode("span", "game-team");
    if (t.logo) {
      const img = edgeNode("img", "game-logo");
      img.src = t.logo;
      img.alt = "";
      img.loading = "lazy";
      name.append(img);
    }
    if (t.rank) name.append(edgeNode("span", "game-rank", String(t.rank)));
    name.append(edgeNode("span", null, t.abbr || t.short));
    row.append(name, edgeNode("span", "game-score", g.state !== "pre" && t.score != null ? String(t.score) : ""));
    cell.append(row);
  });
  if (g.pick) {
    const sub = edgeNode("span", "score-sub game-pick");
    sub.append(edgeNode("span", null, g.pick.text));
    const pill = edgeResultPill(g.pick.result);
    if (pill) sub.append(pill);
    cell.append(sub);
  }
  return cell;
}

// Fallback when no sport has games: each site's top picks.
function edgePickCells(track, summaries) {
  EDGE_SITES.forEach((site, i) => {
    const s = summaries[i];
    const picks = s ? (s.picks || []).slice(0, 3) : [];
    if (!picks.length) return;
    const head = edgeNode("a", "score-cell score-sport");
    head.href = site.href;
    head.append(edgeNode("span", "score-sport-name", site.sport), edgeNode("span", "score-top", s.heading || ""));
    track.append(head);
    picks.forEach((p, rank) => {
      const cell = edgeNode("a", "score-cell");
      cell.href = site.href;
      cell.append(edgeNode("span", "score-top", rank === 0 ? "Top pick" : `Pick ${rank + 1}`));
      const main = edgeNode("span", "score-main");
      main.append(edgeNode("span", "score-label", p.label), edgeNode("span", "score-value", p.value));
      const sub = edgeNode("span", "score-sub");
      sub.append(edgeNode("span", null, p.sub || ""));
      const pill = edgeResultPill(p.result, s.result_labels);
      if (pill) sub.append(pill);
      cell.append(main, sub);
      track.append(cell);
    });
  });
}

async function initScoreboard() {
  const board = document.querySelector(".scoreboard");
  if (!board) return;
  const slates = await Promise.all(EDGE_SITES.map(edgeLoadGames));
  const track = edgeNode("div", "scoreboard-track");
  EDGE_SITES.forEach((site, i) => {
    const slate = slates[i];
    const games = slate ? edgeOrderGames(slate.games) : [];
    if (!games.length) return;
    const head = edgeNode("a", "score-cell score-sport");
    head.href = site.schedule;
    head.append(edgeNode("span", "score-sport-name", site.sport), edgeNode("span", "score-top", slate.label || ""));
    track.append(head);
    games.forEach(g => track.append(edgeGameCell(site, g)));
  });
  if (!track.children.length) edgePickCells(track, await edgeFetchSummaries());
  if (!track.children.length) return;
  const scroll = board.firstChild ? board.firstChild.scrollLeft : 0;
  board.replaceChildren(track);
  track.scrollLeft = scroll;
  board.hidden = false;
  // Keep live scores moving, like ESPN's bar, while any game is in progress.
  if (slates.some(s => s && s.live && s.games.some(g => g.state === "in"))) setTimeout(initScoreboard, 60000);
}
initScoreboard();

// --- Section tabs on narrow screens ---
// When the tabs don't fit (phones), scroll the current one into view and fade
// whichever edge has more tabs hidden past it, so it's clear the row swipes.
function initSectionTabs() {
  const row = document.querySelector(".tabs-inner");
  if (!row) return;
  const active = row.querySelector("a.active");
  if (active && row.scrollWidth > row.clientWidth) {
    row.scrollLeft = active.offsetLeft - (row.clientWidth - active.offsetWidth) / 2;
  }
  const edges = () => {
    row.classList.toggle("fade-left", row.scrollLeft > 4);
    row.classList.toggle("fade-right", row.scrollLeft + row.clientWidth < row.scrollWidth - 4);
  };
  edges();
  row.addEventListener("scroll", edges, { passive: true });
  window.addEventListener("resize", edges);
}
initSectionTabs();
