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

function initNbaHistory() {
  const H = typeof NBA_HISTORY !== "undefined" ? NBA_HISTORY
    : typeof GAME_HISTORY !== "undefined" ? GAME_HISTORY : null;
  if (!H) return;
  const picker = document.getElementById("day-select");
  const container = document.getElementById("day-content");
  if (!picker || !container) return;

  function render(day) {
    const d = H.days[day];
    const rows = d.games.map(g => `<tr>
        <td><div class="player-name">${esc(g.matchup)}</div>${g.meta ? `<div class="player-meta">${esc(g.meta)}</div>` : ""}</td>
        <td data-label="Pick"><span class="matchup-team">${esc(g.pick)}</span></td>
        <td class="num prob" data-label="Win chance">${g.prob.toFixed(0)}%</td>
        <td class="num" data-label="Result"><span>${nbaResult(g)}</span></td>
      </tr>`).join("");
    const s = d.summary;
    container.innerHTML = `<div class="section-label">${esc(d.label)}: ${s.wins}-${s.losses}${s.voided ? `, ${s.voided} no decision` : ""}</div>
      <table class="data responsive-stack">
        <thead><tr><th>Game</th><th>Pick</th><th class="num">Win chance</th><th class="num">Result</th></tr></thead>
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
