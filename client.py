"""
A minimal MCP client.

Connects to the local MCP server over stdio, hands its tools to Gemini,
and runs the tool-calling loop until the model produces a final answer.

Inference goes through Google AI Studio's OpenAI-compatible endpoint. The
MCP server itself never calls a model -- it parses CSVs. This file is the
only place an API key appears.

    export GEMINI_API_KEY=...      # or put it in .env
    python client.py "which subscriptions am I paying for?"

Written against mcp 2.x, where Client replaces v1's
stdio_client + ClientSession + initialize() layering.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import OpenAI

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "models/gemini-3.8-flash"

SERVER_SCRIPT = "server.py"
MAX_TURNS = 8          # stop runaway tool loops
MAX_RETRIES = 3        # per model call, for 429/503
RETRYABLE = ("429", "500", "502", "503", "504", "overloaded", "UNAVAILABLE")

SYSTEM_PROMPT = """You answer questions about the user's own bank
transactions using the tools provided.

Rules:
- Call list_months before any question involving a time period. Never
  assume which months exist in the data.
- Prefer a single tool call that computes the answer over pulling raw rows
  and doing arithmetic yourself. The tools return exact figures; report
  them as given and never estimate or re-add them.
- If a tool returns an error, read it and try a corrected call.
- Answer in plain prose. Keep it short. Quote the figures you were given,
  with their currency.
"""


def to_openai_tools(mcp_tools) -> list[dict]:
    """Translate MCP tool definitions into the model's function-calling format.

    This is the whole integration layer. MCP already gives us a name, a
    description and a JSON Schema for each tool -- exactly what the model
    needs -- so this is a rename, not a transformation.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


def flatten(result) -> str:
    """Turn an MCP tool result into a plain string for the model."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{getattr(block, 'type', '?')}]")
    text = "\n".join(parts) or "(empty result)"

    # mcp 2.x renamed isError -> is_error; accept either.
    failed = getattr(result, "is_error", None)
    if failed is None:
        failed = getattr(result, "isError", False)

    return f"ERROR: {text}" if failed else text


def complete(llm: OpenAI, messages: list, tools: list):
    """One model call, retrying when Google is busy.

    The free tier throws 429s and 503s often enough that a loop without
    backoff will die mid-demo.
    """
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return llm.chat.completions.create(
                model=MODEL, messages=messages, tools=tools
            )
        except Exception as exc:
            last = exc
            if not any(code in str(exc) for code in RETRYABLE):
                raise
            wait = 2**attempt
            print(f"  [model busy, retrying in {wait}s]", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Model unavailable after {MAX_RETRIES} attempts: {last}")


async def ask(question: str, history: list | None = None) -> tuple[str, list[str]]:
    """Answer one question. Returns (answer, trace)."""
    llm = OpenAI(base_url=BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
    trace: list[str] = []

    params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    # One object in v2: it launches the subprocess, negotiates the protocol
    # version and initialises the session.
    async with Client(params) as client:
        listing = await client.list_tools()
        mcp_tools = getattr(listing, "tools", listing)
        tools = to_openai_tools(mcp_tools)

        trace.append(f"connected: {', '.join(t.name for t in mcp_tools)}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        for _ in range(MAX_TURNS):
            response = complete(llm, messages, tools)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            # No tool calls means the model is done and is answering.
            if not message.tool_calls:
                return message.content or "(no answer)", trace

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    content = "ERROR: arguments were not valid JSON"
                    trace.append(f"{name}(<invalid json>)")
                else:
                    trace.append(f"{name}({json.dumps(args)})")
                    try:
                        result = await client.call_tool(name, args)
                        content = flatten(result)
                    except Exception as exc:
                        # Hand the failure back to the model rather than
                        # crashing -- it can usually correct itself.
                        content = f"ERROR: {type(exc).__name__}: {exc}"
                    trace.append(f"  -> {content[:120]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )

        return f"Gave up after {MAX_TURNS} turns without a final answer.", trace


def main() -> None:
    load_dotenv()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Put it in .env.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer, trace = asyncio.run(ask(question))

    for line in trace:
        print(f"[{line}]", file=sys.stderr)
    print()
    print(answer)


if __name__ == "__main__":
    main()