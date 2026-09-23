// NBA Edge's History day picker. site.js handles sorting and the Accuracy
// charts; this renders a day of game picks from NBA_HISTORY.

function nbaResult(g) {
  if (g.void) return "<span class='pill pill-void'>NO DECISION</span>";
  if (g.correct === null) return "<span class='faint'>Pending</span>";
  const pillHtml = g.correct
    ? "<span class='pill pill-positive'>WIN</span>"
    : "<span class='pill pill-danger'>LOSS</span>";
  return `${esc(g.score)} ${pillHtml}`;
}

function initNbaHistory() {
  if (typeof NBA_HISTORY === "undefined") return;
  const picker = document.getElementById("day-select");
  const container = document.getElementById("day-content");
  if (!picker || !container) return;

  function render(day) {
    const d = NBA_HISTORY.days[day];
    const rows = d.games.map(g => `<tr>
        <td><div class="player-name">${esc(g.matchup)}</div></td>
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

  NBA_HISTORY.order.forEach(day => {
    const opt = document.createElement("option");
    opt.value = day;
    opt.textContent = NBA_HISTORY.days[day].label;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => render(picker.value));
  picker.value = NBA_HISTORY.order[0];
  render(picker.value);
}
initNbaHistory();
