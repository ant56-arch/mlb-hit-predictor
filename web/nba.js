// Day picker for game picks: NBA Edge's History tab (NBA_HISTORY) and MLB
// Edge's Games tab (GAME_HISTORY). site.js handles sorting and the Accuracy
// charts; this renders one day of game picks.

function nbaResult(g) {
  if (g.void) return "<span class='pill pill-void'>NO DECISION</span>";
  if (g.correct === null) return "<span class='faint'>Pending</span>";
  const pillHtml = g.correct
    ? "<span class='pill pill-positive'>WIN</span>"
    : "<span class='pill pill-danger'>LOSS</span>";
  return `${esc(g.score)} ${pillHtml}`;
}

// The moneyline bet: "BUF to win +135", the model's pick, VALUE at a 6+ point edge, and once graded
// the units won or lost at the book price.
function nbaMoneyline(m) {
  if (!m) return "<span class='faint'>No odds</span>";
  let res = "";
  if (m.void) res = " <span class='pill pill-void'>NO DECISION</span>";
  else if (m.won !== null && m.won !== undefined) {
    const u = `${m.units >= 0 ? "+" : ""}${m.units.toFixed(2)}u`;
    res = ` <span class='pill ${m.won ? "pill-positive" : "pill-danger"}'>${u}</span>`;
  }
  const value = m.value ? " <span class='pill pill-primary'>VALUE</span>" : "";
  return `<div class="ml"><div class="ml-pick">${esc(m.text)}${value}${res}</div>
    <div class="ml-sub">${esc(m.detail)}</div></div>`;
}

function initNbaHistory() {
  const H = typeof NBA_HISTORY !== "undefined" ? NBA_HISTORY
    : typeof GAME_HISTORY !== "undefined" ? GAME_HISTORY : null;
  if (!H) return;
  const picker = document.getElementById("day-select");
  const container = document.getElementById("day-content");
  if (!picker || !container) return;

  function render(day) {
    const d = H.days[day];
    const hasMl = d.games.some(g => g.ml);  // days before moneyline picks have none
    const rows = d.games.map(g => `<tr>
        <td><div class="player-name">${esc(g.matchup)}</div>${g.meta ? `<div class="player-meta">${esc(g.meta)}</div>` : ""}</td>
        <td data-label="Pick"><span class="matchup-team">${esc(g.pick)}</span></td>
        <td class="num prob" data-label="Win chance">${g.prob.toFixed(0)}%</td>
        ${hasMl ? `<td data-label="Moneyline bet" class="ml-cell">${nbaMoneyline(g.ml)}</td>` : ""}
        <td class="num" data-label="Result"><span>${nbaResult(g)}</span></td>
      </tr>`).join("");
    const s = d.summary;
    const ml = s.ml ? ` · Moneyline ${s.ml.wins}-${s.ml.losses}, ${esc(s.ml.units)}` : "";
    container.innerHTML = `<div class="section-label">${esc(d.label)}: ${s.wins}-${s.losses}${s.voided ? `, ${s.voided} no decision` : ""}${ml}</div>
      <table class="data responsive-stack">
        <thead><tr><th>Game</th><th>Pick</th><th class="num">Win chance</th>${hasMl ? "<th>Moneyline bet</th>" : ""}<th class="num">Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  H.order.forEach(day => {
    const opt = document.createElement("option");
    opt.value = day;
    opt.textContent = H.days[day].label;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => render(picker.value));
  picker.value = H.order[0];
  render(picker.value);
}
initNbaHistory();
