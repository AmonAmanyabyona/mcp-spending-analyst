"""
Local web UI for the spending analyst.

One MCP session is opened at startup and held for the life of the process,
rather than spawning the server per request. That is what a real client
does, and it keeps each question to a couple of hundred milliseconds.

    pip install fastapi uvicorn
    uvicorn app:app --reload

Then open http://127.0.0.1:8000

Nothing here listens on a public interface and nothing is hosted. The
statement stays on this machine; only the figures a tool returns are sent
to the model.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from mcp import Client, StdioServerParameters
from openai import OpenAI
from pydantic import BaseModel

from client import (
    BASE_URL,
    MAX_TURNS,
    SERVER_SCRIPT,
    SYSTEM_PROMPT,
    complete,
    flatten,
    to_openai_tools,
)

load_dotenv()

state: dict = {}

# The MCP session is a single duplex stream; two overlapping tool calls
# would interleave their messages on it. One question at a time.
session_lock = asyncio.Lock()


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
    state["llm"] = OpenAI(base_url=BASE_URL, api_key=_require_key())

    print(
        f"MCP session ready: {', '.join(t.name for t in mcp_tools)}",
        file=sys.stderr,
    )

    yield

    await stack.aclose()


def _require_key() -> str:
    import os

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env.")
    return key


app = FastAPI(lifespan=lifespan)


class Turn(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    history: list[Turn] = []


class Answer(BaseModel):
    answer: str
    trace: list[dict]


async def run_loop(question: str, history: list[Turn]) -> Answer:
    """The agent loop, against the already-open session."""
    client = state["client"]
    llm = state["llm"]
    tools = state["tools"]
    trace: list[dict] = []

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append({"role": "user", "content": question})

    for _ in range(MAX_TURNS):
        # The OpenAI client is synchronous; running it inline would block
        # the event loop for the whole model call.
        response = await asyncio.to_thread(complete, llm, messages, tools)

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return Answer(answer=message.content or "(no answer)", trace=trace)

        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                content = "ERROR: arguments were not valid JSON"
                args = {}
            else:
                try:
                    result = await client.call_tool(name, args)
                    content = flatten(result)
                except Exception as exc:
                    content = f"ERROR: {type(exc).__name__}: {exc}"

            trace.append({"tool": name, "args": args, "result": content})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    return Answer(
        answer=f"Gave up after {MAX_TURNS} turns without a final answer.",
        trace=trace,
    )


@app.post("/api/ask", response_model=Answer)
async def ask(body: Question) -> Answer:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Ask a question first.")

    async with session_lock:
        try:
            return await run_loop(body.question.strip(), body.history)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


# Mounted last on purpose: a mount at "/" swallows every route registered
# after it, and the API would start returning 404s.
app.mount("/", StaticFiles(directory="web", html=True), name="web")