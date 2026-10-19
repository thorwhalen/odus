// odus explorer: the page's behaviour.
//
// The URL is the state: ?r=<respondent>&f=<comma-separated variables>.
// Choosing another respondent is going somewhere (a new history entry, so Back
// returns to the previous one); choosing variables refines the current view (the
// entry is replaced). The figures are made in Python (odus.explore.trajectory).
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (name, params = {}) => {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`api/${name}${query ? "?" + query : ""}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
};

// --- route: the one place that reads and writes the URL ---------------------
function readRoute() {
  const params = new URLSearchParams(location.search);
  const fields = (params.get("f") || "").split(",").filter(Boolean);
  return { respondent: params.get("r"), fields };
}
function formatRoute({ respondent, fields }) {
  const params = new URLSearchParams();
  if (respondent) params.set("r", respondent);
  if (fields.length) params.set("f", fields.join(","));
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
  try {
    const [r, v] = await Promise.all([api("respondents"), api("variables")]);
    respondents = r.respondents;
    fillRespondents();
    fillVariables(v.variables);
  } catch (error) {
    return showError(`Couldn't load the data: ${error.message}`, init);
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
  plot.replaceChildren();
  plot.setAttribute("aria-busy", "true");
  setStatus(`Loading ${respondent}…`);
  try {
    const result = await api("trajectory", { respondent, fields: fields.join(",") });
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
  const button = Object.assign(document.createElement("button"), { type: "button", textContent: "Retry" });
  button.addEventListener("click", retry);
  status.replaceChildren(`${text} `, button);
}

init();
