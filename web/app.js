/* Spending analyst --- frontend.
   No framework, no build step. Talks to /api/ask, /api/status, /api/files. */

const thread    = document.getElementById("thread");
const composer  = document.getElementById("composer");
const input     = document.getElementById("question");
const send      = document.getElementById("send");
const drop      = document.getElementById("drop");
const filePick  = document.getElementById("file");
const note      = document.getElementById("upload-note");
const clearBtn  = document.getElementById("clear");

// Only user and assistant turns are kept. Tool messages stay server-side;
// the browser never holds a copy of the figures beyond what is on screen.
const history = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function money(value, currency) {
  if (typeof value !== "number") return "—";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: currency || "EUR",
    maximumFractionDigits: 0,
  }).format(value);
}

/* --- sidebar ------------------------------------------------------ */

async function refreshStatus() {
  try {
    const data = await (await fetch("/api/status")).json();
    const usingSample = (data.source || "").includes("sample");

    document.getElementById("fact-source").textContent =
      data.error ? "none loaded" : usingSample ? "sample data" : "your statements";

    document.getElementById("fact-period").textContent =
      data.months?.length ? `${data.months[0]} → ${data.months.at(-1)}` : "—";

    document.getElementById("fact-count").textContent =
      data.transaction_count?.toLocaleString() ?? "—";

    document.getElementById("fact-total").textContent =
      money(data.total_spending, data.currency);

    const tags = document.getElementById("fact-categories");
    tags.replaceChildren();
    for (const category of data.categories || []) {
      tags.appendChild(el("span", "tag", category));
    }

    clearBtn.hidden = usingSample || !!data.error;
  } catch {
    document.getElementById("fact-source").textContent = "unavailable";
  }
}

function say(message, bad) {
  note.textContent = message;
  note.className = bad ? "note bad" : "note";
  note.hidden = false;
}

async function addFiles(files) {
  const csvs = [...files].filter((f) => f.name.toLowerCase().endsWith(".csv"));
  if (!csvs.length) return say("Only .csv files, please.", true);

  say(csvs.length === 1 ? "Adding…" : `Adding ${csvs.length} files…`);

  for (const file of csvs) {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/files", { method: "POST", body });
    if (!response.ok) {
      const { detail } = await response.json().catch(() => ({}));
      return say(detail || `Could not add ${file.name}.`, true);
    }
  }

  say(`Added ${csvs.length} file${csvs.length > 1 ? "s" : ""}. Nothing left this machine.`);
  await refreshStatus();
}

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  addFiles(event.dataTransfer.files);
});

filePick.addEventListener("change", () => {
  addFiles(filePick.files);
  filePick.value = "";
});

clearBtn.addEventListener("click", async () => {
  if (!confirm("Delete the statements in ./data and go back to sample data?")) return;
  await fetch("/api/files", { method: "DELETE" });
  say("Removed. Using sample data again.");
  refreshStatus();
});

/* --- answer rendering --------------------------------------------- */

/* Models put bullets inline more often than they should, producing
   "Increases: * Shopping: €420 * Transport: €210" in one paragraph.
   Pull those onto their own lines, and accept "*" as a bullet marker. */
function normalise(markdown) {
  return markdown
    .replace(/\s+\*\s+(?=\S)/g, "\n- ")
    .replace(/^\s*\*\s+/gm, "- ");
}

/* Paragraphs, **bold**, and "- label: amount" rows. Built with
   textContent throughout, so model output is never parsed as HTML --
   this text derives from the user's own bank data and innerHTML would
   make that an injection path. */
function render(target, markdown) {
  for (const block of normalise(markdown).trim().split(/\n{2,}/)) {
    let list = null;

    for (const line of block.split("\n")) {
      const bullet = line.match(/^\s*-\s+(.*)$/);

      if (!bullet) {
        list = null;
        const text = line.trim();
        if (!text) continue;
        const para = el("p");
        para.appendChild(inline(text));
        target.appendChild(para);
        continue;
      }

      if (!list) {
        list = el("ul");
        target.appendChild(list);
      }

      const item = el("li");
      const split = bullet[1].match(/^(.*?):\s*(.+)$/);
      if (split) {
        item.appendChild(inline(split[1]));
        item.appendChild(el("span", "figure", split[2]));
      } else {
        item.appendChild(inline(bullet[1]));
      }
      list.appendChild(item);
    }
  }
}

function inline(text) {
  const span = el("span");
  for (const part of text.split(/(\*\*[^*]+\*\*)/g)) {
    if (part.startsWith("**") && part.endsWith("**")) {
      span.appendChild(el("strong", null, part.slice(2, -2)));
    } else if (part) {
      span.appendChild(document.createTextNode(part));
    }
  }
  return span;
}

function renderTrace(turn, trace) {
  if (!trace.length) return;

  const details = el("details", "trace");
  details.appendChild(
    el("summary", null, trace.length === 1 ? "1 tool call" : `${trace.length} tool calls`)
  );

  for (const step of trace) {
    const args = JSON.stringify(step.args);
    details.appendChild(el("p", "call", `${step.tool}(${args === "{}" ? "" : args})`));
    const failed = step.result.startsWith("ERROR");
    const first = step.result.split("\n")[0].slice(0, 90);
    details.appendChild(el("p", failed ? "back failed" : "back", first));
  }

  turn.appendChild(details);
}

/* --- asking ------------------------------------------------------- */

function waiting(turn) {
  const row = el("p", "pending");
  row.appendChild(el("span", "spinner"));
  const label = el("span", null, "Working…");
  row.appendChild(label);
  turn.appendChild(row);

  // The free tier is slow enough that silence reads as a crash.
  const started = Date.now();
  const tick = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    label.textContent =
      seconds < 8 ? "Working…" : `Working… ${seconds}s (free-tier models can be slow)`;
  }, 1000);

  return () => {
    clearInterval(tick);
    row.remove();
  };
}

async function ask(question) {
  document.getElementById("opening")?.remove();

  const turn = el("div", "turn");
  turn.appendChild(el("p", "asked", question));
  thread.appendChild(turn);
  const stop = waiting(turn);
  thread.scrollTop = thread.scrollHeight;

  input.value = "";
  input.disabled = send.disabled = true;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });

    const data = await response.json();
    stop();

    if (!response.ok) {
      const problem = el("div", "said failed");
      problem.appendChild(el("p", null, data.detail || "The request failed."));
      turn.appendChild(problem);
      return;
    }

    renderTrace(turn, data.trace || []);
    const said = el("div", "said");
    render(said, data.answer);
    turn.appendChild(said);

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: data.answer });
  } catch (error) {
    stop();
    turn.appendChild(el("p", "said failed", `Could not reach the server: ${error.message}`));
  } finally {
    input.disabled = send.disabled = false;
    input.focus();
    thread.scrollTop = thread.scrollHeight;
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

thread.addEventListener("click", (event) => {
  const starter = event.target.closest("[data-ask]");
  if (starter) ask(starter.dataset.ask);
});

refreshStatus();
input.focus();