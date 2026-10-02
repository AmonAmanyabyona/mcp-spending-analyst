/* Spending analyst --- frontend.
   No framework, no build step. Talks to /api/ask and renders the result. */

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const input = document.getElementById("question");
const send = document.getElementById("send");

// Only user and assistant turns are kept. Tool messages stay server-side;
// the model is given the full detail there, and the browser never holds a
// copy of the figures beyond what is on screen.
const history = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* Models put bullets inline more often than they should, producing
   "Increases: * Shopping: €420 * Transport: €210" in one paragraph.
   Pull those onto their own lines, and accept "*" as a bullet marker. */
function normalise(markdown) {
  return markdown
    .replace(/\s+\*\s+(?=\S)/g, "\n- ")
    .replace(/^\s*\*\s+/gm, "- ");
}

/* Minimal formatting: paragraphs, **bold**, and "- label: amount" rows.
   Built with textContent throughout, so model output is never parsed as
   HTML -- the text passing through here is derived from the user's own
   bank data, and innerHTML would make that an injection path. */
function render(target, markdown) {
  const blocks = normalise(markdown).trim().split(/\n{2,}/);

  for (const block of blocks) {
    let list = null;

    for (const line of block.split("\n")) {
      const bullet = line.match(/^\s*-\s+(.*)$/);

      // Not a bullet: close any open list and emit a paragraph.
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
      // "Groceries: €315.54" -> label left, figure right, so amounts
      // align down the column the way a statement does.
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
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
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
  const label = trace.length === 1 ? "1 tool call" : `${trace.length} tool calls`;
  details.appendChild(el("summary", null, label));

  for (const step of trace) {
    const args = JSON.stringify(step.args);
    details.appendChild(el("p", "call", `${step.tool}(${args === "{}" ? "" : args})`));

    const first = step.result.split("\n")[0].slice(0, 90);
    const failed = step.result.startsWith("ERROR");
    details.appendChild(el("p", failed ? "back failed" : "back", first));
  }

  turn.appendChild(details);
}

function scrollDown() {
  thread.scrollTop = thread.scrollHeight;
}

async function ask(question) {
  document.querySelector(".opening")?.remove();

  const turn = el("div", "turn");
  turn.appendChild(el("p", "asked", question));
  const status = el("p", "pending", "Working...");
  turn.appendChild(status);
  thread.appendChild(turn);
  scrollDown();

  input.value = "";
  input.disabled = true;
  send.disabled = true;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });

    const data = await response.json();
    status.remove();

    if (!response.ok) {
      const problem = el("p", "said failed");
      problem.appendChild(
        inline(data.detail || "The request failed. Check the terminal running uvicorn.")
      );
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
    status.remove();
    turn.appendChild(
      el("p", "said failed", `Could not reach the server: ${error.message}`)
    );
  } finally {
    input.disabled = false;
    send.disabled = false;
    input.focus();
    scrollDown();
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

input.focus();