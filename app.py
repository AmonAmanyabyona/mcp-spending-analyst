"""
Local web UI for the spending analyst.

One MCP session is opened at startup and held for the life of the process,
rather than spawning the server per request.

    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload

Then open http://127.0.0.1:8000

Nothing here listens on a public interface and nothing is hosted. The
"upload" endpoint copies a file from the browser's file picker into ./data
on this same machine -- it never crosses a network.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from mcp import Client, StdioServerParameters
from pydantic import BaseModel

from client import (
    SERVER_SCRIPT,
    SYSTEM_PROMPT,
    complete,
    flatten,
    make_llm,
    to_openai_tools,
)

load_dotenv()

DATA_DIR = Path("data")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9._ -]+\.csv$", re.IGNORECASE)

# Budgets. The per-call timeout and model rotation live in client.py; this
# is the ceiling on a whole question, so a pathological loop ends in a
# readable 504 rather than a spinner that never stops.
MAX_TURNS = 6
REQUEST_BUDGET_SECONDS = 150

state: dict = {}

# The MCP session is a single duplex stream; two overlapping tool calls
# would interleave their messages on it. One question at a time.
session_lock = asyncio.Lock()


def log(message: str) -> None:
    """Progress to stderr, so the terminal shows where a slow request is."""
    print(f"  {message}", file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the MCP session once, keep it for the life of the process.

    The MCP context managers have to stay entered, which a plain
    `async with` inside a request handler cannot do -- hence the exit stack
    held on module state.
    """
    stack = AsyncExitStack()
    params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    client = await stack.enter_async_context(Client(params))
    listing = await client.list_tools()
    mcp_tools = getattr(listing, "tools", listing)

    state["client"] = client
    state["tools"] = to_openai_tools(mcp_tools)
    # Shared factory, so the SDK timeout and retry policy are defined once
    # in client.py and the CLI and the web UI cannot drift apart.
    state["llm"] = make_llm()

    log(f"MCP session ready: {', '.join(t.name for t in mcp_tools)}")

    yield

    await stack.aclose()


app = FastAPI(lifespan=lifespan)


# --- models -----------------------------------------------------------


class Turn(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    history: list[Turn] = []


class Answer(BaseModel):
    answer: str
    trace: list[dict]


# --- the agent loop ---------------------------------------------------


async def run_loop(question: str, history: list[Turn]) -> Answer:
    """The agent loop, against the already-open session."""
    client = state["client"]
    llm = state["llm"]
    tools = state["tools"]
    trace: list[dict] = []

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append({"role": "user", "content": question})

    for turn_number in range(MAX_TURNS):
        log(f"turn {turn_number + 1}: asking the model")

        # The OpenAI client is synchronous; running it inline would block
        # the event loop for the whole model call.
        response = await asyncio.to_thread(complete, llm, messages, tools)

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            log("model answered")
            return Answer(answer=message.content or "(no answer)", trace=trace)

        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                content = "ERROR: arguments were not valid JSON"
                args = {}
            else:
                log(f"calling {name}({json.dumps(args)})")
                try:
                    result = await client.call_tool(name, args)
                    content = flatten(result)
                except Exception as exc:
                    content = f"ERROR: {type(exc).__name__}: {exc}"

            trace.append({"tool": name, "args": args, "result": content})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    log(f"gave up after {MAX_TURNS} turns")
    return Answer(
        answer=(
            f"I used all {MAX_TURNS} tool calls without reaching an answer. "
            f"Try a more specific question, for example naming a month."
        ),
        trace=trace,
    )


# --- endpoints --------------------------------------------------------


@app.post("/api/ask", response_model=Answer)
async def ask(body: Question) -> Answer:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Ask a question first.")

    async with session_lock:
        log(f'question: "{question}"')
        try:
            return await asyncio.wait_for(
                run_loop(question, body.history),
                timeout=REQUEST_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            log("gave up: request budget exceeded")
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Gave up after {REQUEST_BUDGET_SECONDS}s. The free-tier "
                    f"models are usually rate limited when this happens — "
                    f"wait a minute and try again."
                ),
            ) from None
        except RuntimeError as exc:
            # complete() raises this when every model refused.
            log(f"no model available: {exc}")
            raise HTTPException(
                status_code=503,
                detail="All models are rate limited right now. Try again shortly.",
            ) from exc
        except Exception as exc:
            log(f"failed: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/status")
async def status() -> dict:
    """What data is currently loaded.

    Deliberately asks the MCP server via list_months rather than reading
    the files directly, so the UI sees exactly what the model sees.
    """
    async with session_lock:
        try:
            result = await state["client"].call_tool("list_months", {})
            return json.loads(flatten(result))
        except Exception as exc:
            return {"error": str(exc), "months": []}


@app.post("/api/files")
async def upload(file: UploadFile = File(...)) -> dict:
    """Copy a CSV into ./data on this machine.

    Called "upload" because that is the browser control, but both ends of
    this request are localhost: the file moves from one folder to another
    on the same disk. The server picks the change up on the next tool call
    via its mtime fingerprint.
    """
    name = Path(file.filename or "").name
    if not SAFE_NAME.match(name):
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted, with plain filenames.",
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name

    size = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File is too large.")
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    log(f"added {name} ({size:,} bytes) to {DATA_DIR}/")
    return {"filename": name, "bytes": size}


@app.delete("/api/files")
async def clear_data() -> dict:
    """Remove the local statements and fall back to the sample data."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("cleared ./data, falling back to sample data")
    return {"cleared": True}


# Mounted last on purpose: a mount at "/" swallows every route registered
# after it, and the API would start returning 404s.
app.mount("/", StaticFiles(directory="web", html=True), name="web")