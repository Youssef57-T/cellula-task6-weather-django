(function () {
  "use strict";

  const form = document.getElementById("forecast-form");
  const dateInput = document.getElementById("id_date");
  const hourInput = document.getElementById("id_hour");
  const horizon = document.getElementById("id_horizon");
  const horizonOut = document.getElementById("horizon-out");
  const hint = document.getElementById("recorded-hint");

  // Live value of the horizon slider
  if (horizon && horizonOut) {
    horizon.addEventListener("input", () => { horizonOut.textContent = horizon.value + " h"; });
  }

  // Spinner on submit
  form.addEventListener("submit", () => {
    const btn = document.getElementById("submit-btn");
    btn.querySelector(".spinner-border").classList.remove("d-none");
    btn.querySelector(".bi").classList.add("d-none");
  });

  // Show the recorded temperature of the hour before the chosen start
  async function updateHint() {
    if (!dateInput.value) { hint.textContent = ""; return; }
    const params = new URLSearchParams({ date: dateInput.value, hour: hourInput.value });
    try {
      const res = await fetch(window.OBSERVATION_URL + "?" + params);
      if (!res.ok) { hint.textContent = ""; return; }
      const obs = await res.json();
      if (obs.temperature === null) {
        const end = new Date(obs.data_end).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
        hint.textContent = `No recording after ${end}: the model uses its own estimate.`;
        return;
      }
      const t = new Date(obs.time);
      const hh = String(t.getHours()).padStart(2, "0");
      hint.textContent = `Recorded at ${hh}:00: ${obs.temperature.toFixed(1)} °C.`;
    } catch (e) {
      hint.textContent = "";
    }
  }
  dateInput.addEventListener("change", updateHint);
  hourInput.addEventListener("change", updateHint);
  updateHint();

  // Forecast chart
  const dataEl = document.getElementById("forecast-data");
  if (!dataEl || !window.Chart) return;
  const data = JSON.parse(dataEl.textContent);
  // on phones the result sits below the form: bring it into view
  if (window.innerWidth < 992) document.getElementById("result").scrollIntoView({ block: "start" });
  const band = getComputedStyle(document.getElementById("result")).getPropertyValue("--band").trim() || "#2a78d6";

  const fmt = (iso) => {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, "0") + ":00";
    return d.getHours() === 0 ? [hh, d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })] : hh;
  };
  const hist = data.history, fc = data.forecast;
  const labels = [...hist, ...fc].map((p) => fmt(p.time));
  const pad = (n) => Array(n).fill(null);
  // the forecast line starts at the last observed hour so the two lines join
  const last = hist[hist.length - 1].temperature;
  const series = (key) => [...pad(hist.length - 1), last, ...fc.map((p) => p[key])];
  const hasActual = fc.some((p) => p.actual !== null);
  const temps = [...hist, ...fc].map((p) => p.temperature).concat(fc.map((p) => p.actual)).filter((v) => v !== null);

  // input hours: recordings, and after the data ends the model's own estimates
  const histPart = (recorded) => hist.map((p, i) => {
    const next = hist[i + 1];
    // keep the joining point in both lines so they connect
    return p.recorded === recorded || (next && next.recorded === recorded) ? p.temperature : null;
  });
  const anyEstimated = hist.some((p) => !p.recorded);
  const datasets = [
    { label: "Recorded (input)", data: [...histPart(true), ...pad(fc.length)],
      borderColor: "#8a93a6", backgroundColor: "#8a93a6", borderWidth: 2, pointRadius: 0, tension: .25 },
    { label: "95 % range", data: series("upper"), borderWidth: 0, pointRadius: 0, fill: false, tension: .25,
      backgroundColor: band + "55", borderColor: band + "55" },
    { label: "95 % range low", data: series("lower"), borderWidth: 0, pointRadius: 0,
      backgroundColor: band + "2e", fill: "-1", tension: .25 },
    { label: "Forecast", data: series("temperature"), borderColor: band, backgroundColor: band,
      borderWidth: 3, pointRadius: 2.5, tension: .25 },
  ];
  if (anyEstimated) {
    datasets.push({ label: "Model estimate (no data)", data: [...histPart(false), ...pad(fc.length)],
      borderColor: "#b4bac7", backgroundColor: "#b4bac7", borderDash: [3, 3], borderWidth: 2, pointRadius: 0,
      tension: .25 });
  }
  if (hasActual) {
    datasets.push({ label: "Recorded (actual)", data: [...pad(hist.length), ...fc.map((p) => p.actual)],
      borderColor: "#1d2433", backgroundColor: "#1d2433", borderDash: [5, 4], borderWidth: 1.6,
      pointRadius: 2, tension: .25 });
  }

  new Chart(document.getElementById("forecast-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      spanGaps: false,
      scales: {
        x: { grid: { display: false }, ticks: { autoSkip: true, maxRotation: 0, maxTicksLimit: 12, color: "#66708a" } },
        y: { title: { display: true, text: "°C", color: "#66708a" }, grid: { color: "#eceff4" }, ticks: { color: "#66708a" },
             // fit the temperatures, not the band: far from the data the 95 % range can be hundreds of degrees wide
             min: Math.floor(Math.min(...temps) - 6), max: Math.ceil(Math.max(...temps) + 6) },
      },
      plugins: {
        legend: {
          labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 14, color: "#1d2433",
                    filter: (item) => item.text !== "95 % range low" },
        },
        tooltip: {
          // at the joining hour only the recorded value is shown
          filter: (item) => item.dataset.label !== "95 % range low" && item.raw !== null &&
            (item.dataset.label.endsWith("(input)") || item.dataset.label.startsWith("Model estimate") ||
             item.dataIndex >= hist.length),
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.label === "95 % range") {
                const lo = ctx.chart.data.datasets[2].data[ctx.dataIndex];
                return ` 95 % range: ${lo.toFixed(1)} – ${ctx.raw.toFixed(1)} °C`;
              }
              return ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)} °C`;
            },
          },
        },
      },
    },
  });
})();
