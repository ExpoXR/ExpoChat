"""Credential-isolated provider SDK runner. Reads one JSON request from stdin."""
import asyncio
import json
import os
import sys
from typing import Any


def run_codex(payload: dict[str, Any]) -> str:
    from openai_codex import Codex, Sandbox

    os.environ["CODEX_API_KEY"] = payload["api_key"]
    with Codex() as codex:
        thread = codex.thread_start(model=payload["model"], sandbox=Sandbox.read_only)
        result = thread.run(payload["prompt"])
        return result.final_response


async def run_claude(payload: dict[str, Any]) -> str:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    os.environ["ANTHROPIC_API_KEY"] = payload["api_key"]
    allowed = ["WebSearch", "WebFetch"] if payload.get("allow_web") else []
    blocked = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
    if not payload.get("allow_web"):
        blocked.extend(["WebSearch", "WebFetch"])
    result = ""
    async for message in query(
        prompt=payload["prompt"],
        options=ClaudeAgentOptions(
            model=payload["model"],
            allowed_tools=allowed,
            disallowed_tools=blocked,
            max_turns=8,
        ),
    ):
        if isinstance(message, ResultMessage):
            result = message.result
    return result


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        provider = payload["provider"]
        if provider == "codex":
            result = run_codex(payload)
        elif provider == "claude":
            result = asyncio.run(run_claude(payload))
        else:
            raise ValueError("Unsupported brain provider")
        print(json.dumps({"ok": True, "content": result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
