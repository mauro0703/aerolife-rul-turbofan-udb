"use strict";

const byId = (id) => document.getElementById(id);
const number = (value, digits = 1) => new Intl.NumberFormat("es-SV", {
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
}).format(value);
const states = {
  "Revisión prioritaria": { css: "priority", color: "#a64540" },
  "Planificar inspección": { css: "plan", color: "#b88138" },
  "Seguimiento": { css: "follow", color: "#18776f" },
};
const defaults = { horizon: 30, buffer: 10, maximum: 20 };
const svgNamespace = "http://www.w3.org/2000/svg";
let dataset;
let visibleFleet = [];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function svg(parent, tag, attributes, text) {
  const node = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  parent.append(node);
  return node;
}

function scenario() {
  const values = {};
  for (const [name, fallback] of Object.entries(defaults)) {
    const input = byId(name);
    const value = Number.isFinite(input.valueAsNumber) ? Math.trunc(input.valueAsNumber) : fallback;
    values[name] = Math.max(Number(input.min), Math.min(Number(input.max), value));
  }
  return values;
}

function decision(lower, { horizon, buffer, maximum }) {
  let status = "Seguimiento";
  if (lower <= buffer) status = "Revisión prioritaria";
  else if (lower <= horizon + buffer) status = "Planificar inspección";
  return { status, revisit: Math.floor(Math.min(maximum, Math.max(0, lower - buffer))) };
}

function cards(target, values) {
  target.replaceChildren(...values.map(({ label, value, detail, css }) => {
    const card = element("div", undefined, `card ${css || ""}`);
    card.append(element("div", label, "card-label"), element("div", value, "card-value"));
    if (detail) card.append(element("div", detail, "card-detail"));
    return card;
  }));
}

function table(target, headings, rows) {
  const result = element("table");
  const headingRow = result.createTHead().insertRow();
  headings.forEach((heading) => {
    const cell = element("th", heading);
    cell.scope = "col";
    headingRow.append(cell);
  });
  const body = result.createTBody();
  for (const row of rows) {
    const line = body.insertRow();
    for (const value of row) {
      const cell = line.insertCell();
      if (typeof value === "object") cell.append(value);
      else cell.textContent = value;
    }
  }
  target.replaceChildren(result);
}

function axes(target, xDomain, yDomain, xTicks, yTicks, xTitle, yTitle, description) {
  target.replaceChildren();
  const width = 960;
  const height = 278;
  const left = 70;
  const right = width - 20;
  const top = 16;
  const bottom = height - 48;
  const x = (value) => left + (value - xDomain[0]) * (right - left) / (xDomain[1] - xDomain[0]);
  const y = (value) => bottom - (value - yDomain[0]) * (bottom - top) / (yDomain[1] - yDomain[0]);
  const chart = svg(target, "svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": description });
  const yPrecision = Math.max(0, Math.min(6, 1 + Math.ceil(-Math.log10((yDomain[1] - yDomain[0]) / 4))));
  for (const tick of yTicks) {
    svg(chart, "line", { x1: left, x2: right, y1: y(tick), y2: y(tick), class: "gridline" });
    svg(chart, "text", { x: left - 10, y: y(tick) + 4, "text-anchor": "end" }, number(tick, yPrecision));
  }
  svg(chart, "line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "axis" });
  svg(chart, "line", { x1: left, x2: left, y1: top, y2: bottom, class: "axis" });
  for (const [value, label] of xTicks) {
    svg(chart, "text", { x: x(value), y: bottom + 20, "text-anchor": "middle" }, label);
  }
  svg(chart, "text", { x: (left + right) / 2, y: height - 7, "text-anchor": "middle", class: "axis-title" }, xTitle);
  svg(chart, "text", { transform: `translate(17 ${(top + bottom) / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-title" }, yTitle);
  return { chart, x, y };
}

function legend(target, entries) {
  const row = element("div", undefined, "legend");
  for (const [label, color] of entries) {
    const marker = element("i");
    marker.style.backgroundColor = color;
    const item = element("span");
    item.append(marker, document.createTextNode(label));
    row.append(item);
  }
  target.append(row);
}

function fleet() {
  const settings = scenario();
  const filter = byId("status-filter").value;
  visibleFleet = dataset.official.map((row) => ({ ...row, ...decision(row.lower90, settings) }))
    .filter((row) => filter === "all" || row.status === filter)
    .sort((a, b) => a.lower90 - b.lower90 || a.rul_pred - b.rul_pred);
  cards(byId("fleet-cards"), [
    { label: "Motores en vista", value: visibleFleet.length },
    ...Object.entries(states).map(([label, style]) => ({
      label, css: style.css, value: visibleFleet.filter((row) => row.status === label).length,
    })),
  ]);
  const target = byId("fleet-chart");
  if (visibleFleet.length === 0) {
    target.replaceChildren(element("p", "No hay motores para el estado seleccionado.", "notice"));
  } else {
    const upper = Math.ceil(Math.max(...visibleFleet.map((row) => row.upper90)) / 50) * 50;
    const xTicks = visibleFleet.flatMap((row, index) => {
      const step = Math.max(1, Math.ceil(visibleFleet.length / 12));
      return index % step === 0 ? [[index + 0.5, row.engine_id]] : [];
    });
    const yTicks = Array.from({ length: upper / 50 + 1 }, (_, index) => index * 50);
    const { chart, x, y } = axes(target, [0, visibleFleet.length], [0, upper], xTicks, yTicks,
      "Motor de prueba, ordenado por límite inferior", "RUL [ciclos]", "RUL estimada y banda de incertidumbre de la flota");
    const width = Math.min(32, (x(1) - x(0)) * 0.7);
    visibleFleet.forEach((row, index) => {
      const center = x(index + 0.5);
      const bar = svg(chart, "rect", {
        x: center - width / 2, y: y(row.rul_pred), width,
        height: Math.max(1, y(0) - y(row.rul_pred)), fill: states[row.status].color, class: "selectable",
      });
      svg(bar, "title", {}, `Motor ${row.engine_id} · ${number(row.rul_pred)} ciclos · ${row.status}`);
      bar.addEventListener("click", () => {
        byId("engine").value = row.engine_id;
        selectEngine();
        showPage("engine-history");
      });
      svg(chart, "line", { x1: center, x2: center, y1: y(row.upper90), y2: y(row.lower90), stroke: "#203b50", "stroke-opacity": 0.52, "pointer-events": "none" });
    });
    legend(target, Object.entries(states).map(([name, style]) => [name, style.color]));
  }
  table(byId("fleet-table"), ["Motor", "Ciclo", "RUL estimada", "Intervalo nominal", "Estado", "Reevaluar en"],
    visibleFleet.map((row) => [row.engine_id, row.last_cycle, number(row.rul_pred),
      `${number(row.lower90)} – ${number(row.upper90)}`,
      element("span", row.status, `badge ${states[row.status].css}`), `${row.revisit} ciclos`]));
}

function line(chart, x, y, cycles, values, color, width = 2) {
  svg(chart, "polyline", { points: cycles.map((cycle, index) => `${x(cycle)},${y(values[index])}`).join(" "),
    fill: "none", stroke: color, "stroke-width": width, "stroke-linejoin": "round" });
  if (cycles.length === 1) svg(chart, "circle", { cx: x(cycles[0]), cy: y(values[0]), r: 3, fill: color });
}

function engineHistory() {
  const id = byId("engine").value;
  const engine = dataset.engines[id];
  const end = Number(byId("cycle").value);
  const index = end - 1;
  const current = decision(engine.l[index], scenario());
  byId("cycle-label").textContent = `${end} / ${engine.c.length}`;
  cards(byId("engine-cards"), [
    { label: "Motor", value: id, detail: `Ciclo ${end}` },
    { label: "RUL estimada", value: number(engine.p[index]), detail: "ciclos" },
    { label: "Intervalo nominal", value: `${number(engine.l[index], 0)} – ${number(engine.u[index], 0)}`, detail: "ciclos" },
    { label: "Reevaluar en", value: current.revisit, detail: "ciclos" },
  ]);
  byId("engine-note").textContent = end < 30
    ? `${current.status}. Los primeros 29 ciclos quedan fuera del protocolo de calibración.`
    : current.status;
  const cycles = engine.c.slice(0, end);
  const xStep = Math.max(1, Math.ceil(end / 5 / 5) * 5);
  const xTicks = [[1, "1"]];
  for (let value = xStep; value <= end; value += xStep) if (value > 1) xTicks.push([value, value]);
  const upper = Math.ceil(Math.max(...engine.u.slice(0, end)) / 50) * 50;
  const yTicks = Array.from({ length: upper / 50 + 1 }, (_, position) => position * 50);
  const rulTarget = byId("rul-chart");
  const rul = axes(rulTarget, [1, Math.max(2, end)], [0, upper], xTicks, yTicks,
    "Ciclo observado", "RUL [ciclos]", `Trayectoria de RUL del motor ${id}`);
  const band = cycles.map((cycle, j) => `${rul.x(cycle)},${rul.y(engine.u[j])}`)
    .concat(cycles.slice().reverse().map((cycle) => `${rul.x(cycle)},${rul.y(engine.l[cycle - 1])}`));
  svg(rul.chart, "polygon", { points: band.join(" "), fill: "#dcebf5" });
  line(rul.chart, rul.x, rul.y, cycles, engine.p, "#165e90");
  legend(rulTarget, [["RUL estimada", "#165e90"], ["Intervalo nominal del 90 %", "#dcebf5"]]);

  const channel = byId("sensor").value;
  const values = engine.s[channel].slice(0, end);
  const average = values.map((_, j) => {
    const window = values.slice(Math.max(0, j - 14), j + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * 0.1, Math.abs(high) * 0.00001, 0.00001);
  const yDomain = [low - padding, high + padding];
  const ticks = Array.from({ length: 5 }, (_, j) => yDomain[0] + j * (yDomain[1] - yDomain[0]) / 4);
  const sensor = dataset.channels[channel];
  const sensorTarget = byId("sensor-chart");
  const plot = axes(sensorTarget, [1, Math.max(2, end)], yDomain, xTicks, ticks,
    "Ciclo observado", `${sensor.symbol} [${sensor.unit}]`, `Historial de ${sensor.symbol} en motor ${id}`);
  line(plot.chart, plot.x, plot.y, cycles, values, "#839caf", 1.1);
  line(plot.chart, plot.x, plot.y, cycles, average, "#18776f");
  legend(sensorTarget, [["Medición", "#839caf"], ["Media causal de 15 ciclos", "#18776f"]]);
}

function validation() {
  const metrics = dataset.metrics;
  cards(byId("validation-cards"), [
    { label: "RMSE", value: number(metrics.rmse, 2), detail: "ciclos" },
    { label: "MAE", value: number(metrics.mae, 2), detail: "ciclos" },
    { label: "R²", value: number(metrics.r2, 3), detail: `${metrics.n} motores` },
    { label: "Cobertura empírica", value: `${number(metrics.interval_coverage90 * 100, 0)} %`, detail: "intervalo nominal del 90 %" },
  ]);
  byId("interval-note").textContent = `Ancho medio del intervalo ${number(metrics.mean_interval_width, 2)} ciclos. NASA score ${number(metrics.nasa_score, 2)}.`;
  const maximum = Math.ceil(Math.max(...dataset.official.map((row) => Math.max(row.rul_true, row.rul_pred))) / 30) * 30;
  const ticks = Array.from({ length: maximum / 30 + 1 }, (_, index) => index * 30);
  const { chart, x, y } = axes(byId("test-chart"), [0, maximum], [0, maximum], ticks.map((tick) => [tick, tick]), ticks,
    "RUL de referencia [ciclos]", "RUL estimada [ciclos]", "Resultados de la prueba oficial por motor");
  svg(chart, "line", { x1: x(0), x2: x(maximum), y1: y(0), y2: y(maximum), stroke: "#7893a7", "stroke-dasharray": "5 4" });
  for (const row of dataset.official) {
    const dot = svg(chart, "circle", { cx: x(row.rul_true), cy: y(row.rul_pred), r: 4, fill: "#165e90", opacity: 0.76 });
    svg(dot, "title", {}, `Motor ${row.engine_id} · Referencia ${number(row.rul_true)} · Estimación ${number(row.rul_pred)}`);
  }
  table(byId("cv-table"), ["Candidato", "RMSE medio", "Desviación entre particiones", "MAE medio"],
    dataset.comparison.map((row) => [row.candidate, number(row.cv_rmse, 2), number(row.cv_sd, 2), number(row.cv_mae, 2)]));
}

function selectEngine() {
  const cycle = byId("cycle");
  cycle.max = dataset.engines[byId("engine").value].c.length;
  cycle.value = cycle.max;
  engineHistory();
}

function showPage(page) {
  document.querySelectorAll(".page").forEach((section) => { section.hidden = section.id !== page; });
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.page === page));
  });
}

function exportFleet() {
  const rows = [["engine_id", "cycle", "rul_pred", "lower90", "upper90", "status", "revisit_cycles"]];
  visibleFleet.forEach((row) => rows.push([row.engine_id, row.last_cycle, row.rul_pred, row.lower90, row.upper90, row.status, row.revisit]));
  const blob = new Blob(["\ufeff", rows.map((row) => row.join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
  const download = element("a");
  download.href = URL.createObjectURL(blob);
  download.download = "priorizacion_fd001.csv";
  download.click();
  setTimeout(() => URL.revokeObjectURL(download.href), 1000);
}

async function initialize() {
  const bytes = Uint8Array.from(atob(window.AEROLIFE_DATA_GZIP), (character) => character.charCodeAt(0));
  const content = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  dataset = await new Response(content).json();
  for (const id of Object.keys(dataset.engines)) byId("engine").add(new Option(`Motor ${id}`, id));
  for (const [key, channel] of Object.entries(dataset.channels)) {
    byId("sensor").add(new Option(`${channel.symbol} [${channel.unit}]`, key));
  }
  byId("sensor").value = "s3";
  byId("engine").addEventListener("change", selectEngine);
  byId("cycle").addEventListener("input", engineHistory);
  byId("sensor").addEventListener("change", engineHistory);
  byId("status-filter").addEventListener("change", fleet);
  byId("export").addEventListener("click", exportFleet);
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  for (const name of Object.keys(defaults)) {
    byId(name).addEventListener("input", () => { fleet(); engineHistory(); });
    byId(name).addEventListener("change", () => { byId(name).value = scenario()[name]; });
  }
  fleet();
  selectEngine();
  validation();
  byId("dashboard").hidden = false;
  byId("load-status").hidden = true;
}

initialize().catch(() => {
  byId("load-status").textContent = "No se pudieron abrir los datos. Abra index.html desde la carpeta extraída con un navegador actualizado.";
});
