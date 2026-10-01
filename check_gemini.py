"""
Setup diagnostic. Run this before writing any server code.

Checks four things in order, so a failure tells you which one broke:
  1. the key is loaded from .env
  2. the key reaches Google and can list models
  3. a plain chat completion works
  4. tool calling works through the OpenAI-compatible endpoint

Step 4 is the one that matters. The entire project depends on the model
returning tool_calls in OpenAI format via Google's compatibility layer.

    python check_gemini.py
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# models.list() returns deprecated models alongside current ones, so being
# in the list does not mean the key can use it. We try candidates newest
# first and keep going until one actually responds.
EXCLUDE = ("image", "audio", "tts", "lite", "embedding", "vision")

DUMMY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number."},
                    "b": {"type": "number", "description": "Second number."},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def version_key(model_id: str) -> tuple[int, int]:
    """Sort key: (major, minor) from a 'gemini-3.8-flash' style id."""
    match = re.search(r"gemini-(\d+)\.(\d+)", model_id)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def fail(step: str, exc: Exception) -> None:
    print(f"\nFAILED at {step}:\n  {type(exc).__name__}: {exc}\n")
    print("Common causes:")
    print("  404  -> missing trailing slash on base_url, or retired model id")
    print("  401  -> key is wrong, revoked, or not loaded from .env")
    print("  429  -> free-tier rate limit; wait a minute and retry")
    sys.exit(1)


def main() -> None:
    load_dotenv()

    # --- 1. key present -----------------------------------------------
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not found. Is it set in .env?")
        sys.exit(1)
    print(f"[1/4] key loaded, ends in ...{key[-4:]}")

    client = OpenAI(api_key=key, base_url=BASE_URL)

    # --- 2. which models can this key actually use? -------------------
    try:
        models = [m.id for m in client.models.list()]
    except Exception as exc:
        fail("listing models", exc)

    candidates = [
        m for m in models
        if "flash" in m and not any(word in m for word in EXCLUDE)
    ]
    candidates.sort(key=version_key, reverse=True)

    print(f"[2/4] {len(models)} models visible to this key")
    if not candidates:
        print("      no usable flash model found. Full list:")
        for m in sorted(models):
            print(f"        {m}")
        sys.exit(1)
    print(f"      candidates (newest first): {', '.join(candidates[:4])}")

    # --- 3. plain completion ------------------------------------------
    # Walk the candidates until one responds. A 404 here means the model
    # is listed but retired, which is normal -- just try the next.
    model = None
    last_error = None

    for candidate in candidates:
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            )
        except Exception as exc:
            print(f"      {candidate} -> unavailable ({type(exc).__name__})")
            last_error = exc
            continue

        model = candidate
        reply = response.choices[0].message.content
        print(f"[3/4] chat works on {model}, replied: {reply!r}")
        break

    if model is None:
        fail("plain completion (every candidate failed)", last_error)

    # --- 4. tool calling ----------------------------------------------
    # The real test. If this fails, the compatibility layer is not passing
    # function definitions through, and the project needs the native
    # google-genai SDK instead.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 41 plus 1?"}],
            tools=DUMMY_TOOL,
        )
    except Exception as exc:
        fail("tool calling", exc)

    message = response.choices[0].message

    if not message.tool_calls:
        print("[4/4] WARNING: model answered without calling the tool.")
        print(f"      said: {message.content!r}")
        print("\n      Not necessarily fatal -- small models sometimes do easy")
        print("      arithmetic directly. Re-run once. If it never calls the")
        print("      tool, try the next candidate listed above.")
        sys.exit(1)

    call = message.tool_calls[0]
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        print("[4/4] tool called but arguments were not valid JSON:")
        print(f"      {call.function.arguments!r}")
        sys.exit(1)

    print(f"[4/4] tool calling works: {call.function.name}({args})")

    print("\nAll checks passed. Put this in client.py:\n")
    print(f'    MODEL = "{model}"\n')


if __name__ == "__main__":
    main()