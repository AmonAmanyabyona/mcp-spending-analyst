"""
A minimal MCP client.

Connects to the local MCP server over stdio, hands its tools to Gemini,
and runs the tool-calling loop until the model produces a final answer.

Inference goes through Google AI Studio's OpenAI-compatible endpoint. The
MCP server itself never calls a model -- it parses CSVs. This file is the
only place an API key appears.

    export GEMINI_API_KEY=...      # or put it in .env
    python client.py "which subscriptions am I paying for?"

The trace goes to stderr and the answer to stdout, so:
    python client.py "..." 2>/dev/null     # answer only

Written against mcp 2.x, where Client replaces v1's
stdio_client + ClientSession + initialize() layering, and the type system
was snake_cased (inputSchema -> input_schema, isError -> is_error).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# The free-tier quota is roughly 5 requests/minute and is scoped PER MODEL
# (quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier). Each
# entry here is therefore an independent bucket, and rotating on a 429 is
# far faster than waiting one out.
#
# Ordered least-contended first: the lite variants carry less traffic and
# are entirely adequate for choosing between five tools, and the newest
# model is the busiest. Only the last resort is the one everyone else is
# hammering.
#
# To refresh this list for your own key:
#     python -c "from dotenv import load_dotenv; load_dotenv(); import os; \
#     from openai import OpenAI; print('\n'.join(sorted(m.id for m in \
#     OpenAI(api_key=os.environ['GEMINI_API_KEY'], \
#     base_url='https://generativelanguage.googleapis.com/v1beta/openai/' \
#     ).models.list())))"
#
# Skip the live, transcribe, translate, robotics, computer-use and omni
# variants -- different modalities, and they misbehave on a text
# tool-calling call. The "-latest" aliases are moving targets, so they are
# left out of a committed list on purpose.
MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.5-flash",
    "models/gemini-3.6-flash",
    "models/gemini-3.7-flash",
    "models/gemini-3.8-flash",
]

# Index into MODELS of whichever model last answered. Later turns in the
# same question start there instead of re-walking models that just timed
# out, which saves several waits on a multi-turn question.
_preferred = 0

SERVER_SCRIPT = "server.py"

# Budgets. The OpenAI SDK defaults to a 600-second timeout and retries
# internally; left alone, that multiplies with the model rotation and the
# turn loop into a request that can hang for half an hour. Cap the SDK,
# do the retrying here, and bound the whole thing.
REQUEST_TIMEOUT = 40.0
MAX_TURNS = 6        # stop runaway tool loops
MAX_ROUNDS = 2       # how many times to cycle the whole model list
MAX_WAIT = 30.0      # ceiling on any single backoff sleep

# Conditions where another model is worth trying. Matched on exception
# TYPE, not on message text -- APITimeoutError's message is "Request timed
# out.", which a substring check for "timeout" silently misses.
TRY_NEXT_MODEL = (
    RateLimitError,       # 429 -- per-model quota, so the next one is fresh
    APITimeoutError,      # our own REQUEST_TIMEOUT fired
    APIConnectionError,   # transient network trouble
    InternalServerError,  # 5xx
    NotFoundError,        # model retired since this list was written
)

# Google says exactly how long to wait; use that instead of guessing.
DELAY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)

SYSTEM_PROMPT = """You answer questions about the user's own bank
transactions using the tools provided.

Rules:
- Call list_months before any question involving a time period. Never
  assume which months exist in the data.
- Prefer a single tool call that computes the answer over pulling raw rows
  and doing arithmetic yourself. The tools return exact figures; report
  them as given and never estimate or re-add them.
- If a tool returns an error, read it and try a corrected call.
- If the tools cannot answer the question, say so plainly and suggest what
  you could answer instead. Do not keep calling tools hoping for a
  different result.
- Answer in plain prose. Keep it short. Quote the figures you were given,
  with their currency.
- When listing several figures, put each on its own line starting with
  "- ", like "- Groceries: EUR 315.54". Never put list items inline inside
  a sentence, and never use "*" as a bullet marker.
- When a tool returns a total alongside a breakdown, state the total. A
  list of figures without its sum is an incomplete answer.
"""


# --- translation layer ------------------------------------------------


def make_llm() -> OpenAI:
    """The model client, with the SDK's own retrying turned off."""
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ["GEMINI_API_KEY"],
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def to_openai_tools(mcp_tools) -> list[dict]:
    """Translate MCP tool definitions into the model's function-calling format.

    This is the whole integration layer. MCP already gives us a name, a
    description and a JSON Schema for each tool -- exactly what the model
    needs -- so this is a rename, not a transformation.

    mcp 2.x snake_cased the type system, so both spellings are accepted.
    """
    tools = []
    for tool in mcp_tools:
        schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    # A tool with no arguments can report a null schema,
                    # which the API rejects.
                    "parameters": schema or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def flatten(result) -> str:
    """Turn an MCP tool result into a plain string for the model."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{getattr(block, 'type', '?')}]")
    text = "\n".join(parts) or "(empty result)"

    failed = getattr(result, "is_error", None)
    if failed is None:
        failed = getattr(result, "isError", False)

    return f"ERROR: {text}" if failed else text


# --- model calls ------------------------------------------------------


def suggested_delay(message: str) -> float | None:
    """Pull the server's own retry delay out of an error message."""
    for pattern in DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            return float(match.group(1))
    return None


def why(exc: Exception) -> str:
    """A short label for the trace line."""
    return {
        RateLimitError: "quota exhausted",
        APITimeoutError: f"no response in {REQUEST_TIMEOUT:.0f}s",
        APIConnectionError: "connection failed",
        InternalServerError: "overloaded",
        NotFoundError: "model not available",
    }.get(type(exc), type(exc).__name__)


def complete(llm: OpenAI, messages: list, tools: list):
    """One model call, rotating models and honouring server-set delays."""
    global _preferred
    last: Exception | None = None

    for round_number in range(MAX_ROUNDS):
        # Start from whichever model last answered, then wrap around.
        order = MODELS[_preferred:] + MODELS[:_preferred]

        for model in order:
            try:
                response = llm.chat.completions.create(
                    model=model, messages=messages, tools=tools
                )
            except TRY_NEXT_MODEL as exc:
                last = exc
                print(f"  [{model}: {why(exc)}, trying next]", file=sys.stderr)
                continue
            except APIStatusError as exc:
                # Any other 5xx is worth another model; 4xx is ours to fix.
                if exc.status_code >= 500:
                    last = exc
                    print(
                        f"  [{model}: {exc.status_code}, trying next]",
                        file=sys.stderr,
                    )
                    continue
                raise
            else:
                _preferred = MODELS.index(model)
                return response

        # Every model refused. Now the wait is unavoidable.
        if round_number < MAX_ROUNDS - 1:
            wait = min(suggested_delay(str(last)) or 20.0, MAX_WAIT)
            print(f"  [all models unavailable, waiting {wait:.0f}s]", file=sys.stderr)
            time.sleep(wait + 1)

    raise RuntimeError(f"No model available after {MAX_ROUNDS} rounds: {last}")


# --- the loop ---------------------------------------------------------


async def ask(question: str, history: list | None = None) -> tuple[str, list[str]]:
    """Answer one question. Returns (answer, trace)."""
    llm = make_llm()
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

        for turn_number in range(MAX_TURNS):
            print(f"  [turn {turn_number + 1}]", file=sys.stderr)

            # The OpenAI client is synchronous; off the event loop it goes.
            response = await asyncio.to_thread(complete, llm, messages, tools)

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

        return (
            f"I used all {MAX_TURNS} tool calls without reaching an answer. "
            f"Try a more specific question, for example naming a month.",
            trace,
        )


def main() -> None:
    load_dotenv()

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Put it in .env.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    try:
        answer, trace = asyncio.run(ask(question))
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print("\nThe free-tier models are all rate limited. Wait a minute.")
        sys.exit(1)

    for line in trace:
        print(f"[{line}]", file=sys.stderr)
    print()
    print(answer)


if __name__ == "__main__":
    main()