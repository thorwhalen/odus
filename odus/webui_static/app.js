// odus explorer: the page's behaviour.
//
// The URL is the state: ?r=<respondent>&f=<variable>&f=<variable>... (repeated,
// since variable names may contain commas). Each respondent is a record, so
// choosing one, prev/next included, is going somewhere: a new history entry, and
// Back returns to the previous respondent. Choosing variables refines the current
// view: the entry is replaced. The figures are made in Python (odus.explore).
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (name, params = {}) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    for (const v of [].concat(value)) search.append(key, v); // arrays repeat the key
  }
  const query = search.toString();
  const response = await fetch(`api/${name}${query ? "?" + query : ""}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
};

// --- route: the one place that reads and writes the URL ---------------------
function readRoute() {
  const params = new URLSearchParams(location.search);
  return { respondent: params.get("r"), fields: params.getAll("f").filter(Boolean) };
}
function formatRoute({ respondent, fields }) {
  const params = new URLSearchParams();
  if (respondent) params.set("r", respondent);
  for (const field of fields) params.append("f", field);
  const query = params.toString();
  return location.pathname + (query ? "?" + query : "");
}
function navigate(route, { replace = false } = {}) {
  const url = formatRoute(route);
  if (url === location.pathname + location.search) return render();
  history[replace ? "replaceState" : "pushState"](null, "", url);
  render();
}

// --- state loaded once ------------------------------------------------------
let respondents = [];
let latestRequest = 0; // only the newest request may draw (latest wins)

async function init() {
  wireControls();
  await loadData();
}

async function loadData() {
  $("plot").setAttribute("aria-busy", "true");
  setStatus("Loading the respondents…");
  try {
    const [r, v] = await Promise.all([api("respondents"), api("variables")]);
    respondents = r.respondents;
    fillRespondents();
    fillVariables(v.variables);
    const unreadable = Object.keys(v.unreadable || {});
    if (unreadable.length) console.warn("odus explorer: unreadable data", v.unreadable);
  } catch (error) {
    $("plot").removeAttribute("aria-busy");
    return showError(`Couldn't load the data: ${error.message}`, loadData);
  }
  const route = readRoute();
  if (!route.respondent && respondents.length) {
    history.replaceState(null, "", formatRoute({ ...route, respondent: respondents[0] }));
  }
  render();
}

function fillRespondents() {
  const select = $("respondent");
  select.replaceChildren(...respondents.map((k) => new Option(k, k)));
  select.disabled = false;
}

function fillVariables(variables) {
  $("variables").replaceChildren(
    ...variables.map(({ name, n_respondents }) => {
      const li = document.createElement("li");
      li.dataset.name = name.toLowerCase();
      const label = document.createElement("label");
      const box = Object.assign(document.createElement("input"), { type: "checkbox", value: name });
      const count = Object.assign(document.createElement("span"), {
        className: "count",
        textContent: `${n_respondents}`,
        title: `${n_respondents} respondents have it`,
      });
      label.append(box, ` ${name} `, count);
      li.append(label);
      return li;
    })
  );
}

// --- controls ---------------------------------------------------------------
function wireControls() {
  $("respondent").addEventListener("change", (e) =>
    navigate({ ...readRoute(), respondent: e.target.value })
  );
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(+1));
  document.addEventListener("keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return; // browser shortcuts
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(+1);
  });
  $("variables").addEventListener("change", () => {
    const fields = [...$("variables").querySelectorAll("input:checked")].map((b) => b.value);
    navigate({ ...readRoute(), fields }, { replace: true });
  });
  $("variables-clear").addEventListener("click", () =>
    navigate({ ...readRoute(), fields: [] }, { replace: true })
  );
  $("variables-filter").addEventListener("input", (e) => {
    const text = e.target.value.trim().toLowerCase();
    for (const li of $("variables").children) li.hidden = !li.dataset.name.includes(text);
  });
  window.addEventListener("popstate", render);
}

function step(delta) {
  const i = respondents.indexOf(readRoute().respondent);
  const next = respondents[i + delta];
  if (next) navigate({ ...readRoute(), respondent: next });
}

// --- the view ---------------------------------------------------------------
async function render() {
  const { respondent, fields } = readRoute();
  if (!respondent) return;
  const request = ++latestRequest;
  document.title = `${respondent} · odus explorer`;
  $("respondent").value = respondent;
  const i = respondents.indexOf(respondent);
  $("prev").disabled = i <= 0;
  $("next").disabled = i < 0 || i >= respondents.length - 1;
  for (const box of $("variables").querySelectorAll("input")) {
    box.checked = fields.includes(box.value);
  }
  $("variables-summary").textContent = fields.length ? `${fields.length} selected` : "all";

  const plot = $("plot");
  if (window.Plotly) Plotly.purge(plot); // never leave the previous respondent up
  if (i < 0) {
    plot.replaceChildren();
    plot.removeAttribute("aria-busy");
    return showError(`There is no respondent ${respondent}. Pick one above.`);
  }
  plot.replaceChildren();
  plot.setAttribute("aria-busy", "true");
  setStatus(`Loading ${respondent}…`);
  try {
    const result = await api("trajectory", { respondent, fields });
    if (request !== latestRequest) return; // a newer view was asked for meanwhile
    plot.removeAttribute("aria-busy");
    await Plotly.react(plot, result.figure.data, result.figure.layout, { responsive: true });
    plot.setAttribute("aria-label", `Trajectory of ${respondent}`);
    setStatus(
      result.missing_fields.length
        ? `${respondent} has no ${result.missing_fields.join(", ")}.`
        : ""
    );
  } catch (error) {
    if (request !== latestRequest) return;
    plot.removeAttribute("aria-busy");
    showError(`Couldn't show ${respondent}: ${error.message}`, render);
  }
}

function setStatus(text) {
  const status = $("status");
  status.className = "status";
  status.replaceChildren(text);
}

function showError(text, retry) {
  const status = $("status");
  status.className = "status error";
  status.replaceChildren(`${text} `);
  if (retry) {
    const button = Object.assign(document.createElement("button"), { type: "button", textContent: "Retry" });
    button.addEventListener("click", retry);
    status.append(button);
  }
}

init();
