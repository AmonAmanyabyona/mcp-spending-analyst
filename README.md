# Spending analyst

Ask questions about your bank statements in plain English, without
uploading them anywhere.

This is an [MCP](https://modelcontextprotocol.io) server that reads CSV
exports from a folder on your machine and exposes a handful of analysis
tools to an AI agent. The parsing, the arithmetic and the statement itself
stay local. Only the figures a tool returns are sent to a model.

```
> which subscriptions am I paying for?

  list_months()
  find_recurring_charges()

  You have four recurring charges totalling EUR 46.86 per month:

  - FitX:    EUR 18.89   fitness
  - Netflix: EUR 13.99   entertainment
  - Spotify: EUR 10.99   entertainment
  - Apple:   EUR  2.99   entertainment

  Rent is also charged monthly, at EUR 1,150.00.
```

<!-- TODO: replace with the demo recording -->

## The idea

The obvious way to let an AI answer questions about a spreadsheet is to
paste the spreadsheet into a chat window. That is both a privacy problem
and an accuracy problem: a language model asked to total 400 rows will
produce a number that looks right and isn't.

So the tools here never hand over rows. `spending_by_category` runs a
`groupby` and returns eight totals. `find_recurring_charges` does the
detection in pandas and returns the merchants it found. The model picks
which tool to call and writes the sentence around the result; it does no
arithmetic.

That single decision solves both problems at once. Correct figures,
because code computed them. A small privacy footprint, because eight
numbers leave the machine instead of four hundred transactions.

## Quick start

Requires Python 3.11+ and a free [Google AI Studio](https://aistudio.google.com/api-keys)
key.

```bash
git clone https://github.com/AmonAmanyabyona/mcp-spending-analyst.git
cd mcp-spending-analyst

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key_here" > .env
python generate_sample.py          # fake transactions to try it on
```

Then run the local web UI:

```bash
uvicorn app:app --reload
```

and open http://127.0.0.1:8000. It works immediately on the bundled sample
data, so you can try it before pointing it at anything real.

There is also a command-line client:

```bash
python client.py "which subscriptions am I paying for?"
```

Both are MCP clients over the same server. Nothing changes in `server.py`
to support either one — that portability is the point of the protocol.

## Using your own statements

Export a CSV from your bank and drag it onto the panel in the web UI, or
drop it into `data/` by hand. Both do the same thing: the file is copied
into `data/` on this machine. Despite the browser calling it an upload,
both ends of that request are localhost, so the file moves from one folder
to another on the same disk.

The server prefers `data/` and falls back to the bundled sample when it is
empty. Nothing in `data/` is ever committed, and the UI has a button to
delete it again.

Bank exports differ, so check what the parser made of yours:

```bash
python -c "
from analyst.parse import load_folder
from analyst.categorise import coverage
frame = load_folder('data')
print(frame.head())
print(coverage(frame))
"
```

`coverage()` lists the merchants that fell through the category rules,
worst first by money involved. Uncategorised spending is the number that
tells you whether the category totals can be trusted.

Two places to extend, both plain dictionaries:

- **Unrecognised columns** -> add your bank's header names to `ALIASES` in
  `analyst/parse.py`. The parser finds the header row by looking for a
  date column and an amount column, so junk preamble above it is fine.
- **Unrecognised merchants** -> add a pattern to `KNOWN` in
  `analyst/categorise.py`, and the merchant to a category in `CATEGORIES`.

The parser already handles the usual awkwardness: semicolon delimiters,
German number format (`1.234,50`), `DD.MM.YYYY` dates, cp1252 encoding,
and preamble rows above the real header.

## The tools

| Tool | Returns |
|---|---|
| `list_months` | which months have data, totals, categories present |
| `spending_by_category` | per-category totals and shares for a month |
| `find_recurring_charges` | subscriptions and standing charges |
| `compare_months` | per-category deltas between two months |
| `search_transactions` | up to 20 individual rows, references redacted |

Inspect them without a model:

```bash
npx @modelcontextprotocol/inspector python server.py
```

The Inspector calls tools directly, which is much faster for development
than going through an agent — and it costs no API quota. It is also how
you catch a tool returning plausible nonsense, which a chat interface will
happily narrate as fact.

## Using it from another MCP client

The server is not tied to the UI in this repo. Any MCP-compatible client
can use it — Claude Desktop, an editor, your own script.

For Claude Desktop, add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "spending-analyst": {
      "command": "/absolute/path/to/mcp-spending-analyst/.venv/bin/python",
      "args": ["/absolute/path/to/mcp-spending-analyst/server.py"],
      "env": {
        "ANALYST_DATA_DIR": "/absolute/path/to/mcp-spending-analyst/data"
      }
    }
  }
}
```

Absolute paths everywhere. The app does not inherit a shell's working
directory, which is the most common reason these configs silently fail.

## Data handling

Being precise about this, because "your data never leaves your machine" is
not quite true for any setup that uses a hosted model:

| | Where it goes |
|---|---|
| Raw CSV, all rows, references, account numbers | stays local, never sent, never committed |
| What a tool returns (totals, counts, deltas) | sent to the model provider |
| Sample data in `sample_data/` | fake, safe to demo with |

The privacy property is enforced at the tool boundary rather than
promised in prose: `search_transactions` is the only tool that returns
individual rows, it is capped at 20, and payment references are stripped
of IBANs and long digit strings before they leave the process.

Note that **Gemini's free tier may use prompt data for product
improvement**, and tool results are part of the prompt. Use the sample
data for demos and recordings. For a setup where nothing leaves at all,
point the client at a local model — Ollama exposes the same
OpenAI-compatible interface, so it is a one-line change to `BASE_URL`.

## How it works

```
  browser  ──►  app.py  ──►  server.py  ──►  analyst/  ──►  CSV
              (MCP client)   (MCP server)    (pandas)
                   │
                   ▼
                Gemini
```

`app.py` and `client.py` are MCP clients: they launch the server, ask it
what tools exist, pass those schemas to the model, and run the loop until
the model answers with prose instead of another tool call.

`server.py` contains no AI. It is a thin layer of tool definitions over
`analyst/`, which is ordinary pandas and has no idea the protocol exists.

The tool docstrings are not documentation — they are the integration. The
model chooses tools by reading them, so `list_months` says "call this
FIRST" and `search_transactions` says to prefer the aggregate tools. When
the model picks wrong, the fix is editing prose.

`app.py` opens one MCP session at startup and holds it for the life of the
process, rather than spawning the server per request. The sidebar asks
that session for `list_months` rather than reading the files directly, so
the panel shows exactly what the model sees.

## Known limitations

- **Variable-amount bills are missed.** Recurring detection requires the
  amount to be near-identical month to month, which excludes utilities,
  phone contracts and insurance that vary. Tightening the tolerance was a
  deliberate trade: a loose threshold reported supermarkets as
  subscriptions. Variable recurring charges need a separate heuristic.
- **Categories are rule-based**, so an unfamiliar merchant lands in
  `uncategorised` until you add it. Deliberate — a model guessing
  categories would be unreproducible and would mean sending it every
  merchant name.
- **One question at a time.** The MCP session is a single stream, so the
  web UI serialises requests.
- **Free-tier rate limits are low** — around 5 requests per minute, scoped
  per model. The client rotates through six models, each its own quota
  bucket, and honours the retry delay the API returns. A question still
  takes a few seconds and can occasionally time out; the budget in
  `app.py` caps it rather than letting it hang.

## Project layout

```
server.py            MCP server: tool definitions
client.py            MCP client: command line
app.py               MCP client: FastAPI backend for the web UI
generate_sample.py   fake transactions for development and demos
check_gemini.py      setup diagnostic: key, models, tool calling
analyst/
  parse.py           CSV -> clean DataFrame
  categorise.py      merchant strings -> stable keys and categories
web/                 HTML, CSS, JS. No framework, no build step.
data/                your statements. Gitignored.
sample_data/         generated fake statements. Committed.
tests/               pytest
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The sample generator is seeded, so the tests assert on exact figures. If
you change `generate_sample.py`, the parser and categoriser tests need
updating too.

Before anything else, confirm your key works and that tool calling
survives Google's OpenAI-compatibility layer:

```bash
python check_gemini.py
```

It checks the key, lists the models your key can actually reach, and makes
a real tool call. Worth running first — debugging a model problem from
inside the agent loop is much harder.
