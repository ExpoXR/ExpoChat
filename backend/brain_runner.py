"""Credential-isolated provider SDK runner. Reads one JSON request from stdin."""
import asyncio
import json
import os
import sys
from typing import Any


def run_codex(payload: dict[str, Any]) -> dict[str, Any]:
    from openai_codex import Codex, Sandbox

    with Codex() as codex:
        codex.login_api_key(payload["api_key"])
        thread = codex.thread_start(
            model=payload["model"],
            sandbox=Sandbox.read_only,
            cwd=os.environ.get("HOME"),
            ephemeral=True,
            config={"web_search": "live" if payload.get("allow_web") else "disabled"},
        )
        result = thread.run(payload["prompt"])
        total = getattr(getattr(result, "usage", None), "total", None)
        usage = {
            key: int(getattr(total, key, 0) or 0)
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")
        }
        return {"content": result.final_response or "", "usage": usage}


async def run_claude(payload: dict[str, Any]) -> dict[str, Any]:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    os.environ["ANTHROPIC_API_KEY"] = payload["api_key"]
    allowed = ["WebSearch", "WebFetch"] if payload.get("allow_web") else []
    blocked = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
    if not payload.get("allow_web"):
        blocked.extend(["WebSearch", "WebFetch"])
    result = ""
    usage: dict[str, int | float] = {}
    async for message in query(
        prompt=payload["prompt"],
        options=ClaudeAgentOptions(
            model=payload["model"],
            allowed_tools=allowed,
            disallowed_tools=blocked,
            max_turns=8,
            cwd=os.environ.get("HOME"),
            setting_sources=[],
            skills=[],
        ),
    ):
        if isinstance(message, ResultMessage):
            result = message.result or ""
            usage = {
                key: value
                for key, value in (message.usage or {}).items()
                if isinstance(value, (int, float))
            }
    return {"content": result, "usage": usage}


def run_gemini(payload: dict[str, Any]) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=payload["api_key"])
    config_kwargs: dict[str, Any] = {}
    if payload.get("allow_web"):
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if payload.get("max_output_tokens"):
        config_kwargs["max_output_tokens"] = int(payload["max_output_tokens"])
    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
    response = client.models.generate_content(
        model=payload["model"],
        contents=payload["prompt"],
        config=config,
    )
    meta = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(meta, "total_token_count", 0) or 0),
    }
    return {"content": response.text or "", "usage": usage}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        provider = payload["provider"]
        if provider == "codex":
            result = run_codex(payload)
        elif provider == "claude":
            result = asyncio.run(run_claude(payload))
        elif provider == "gemini":
            result = run_gemini(payload)
        else:
            raise ValueError("Unsupported brain provider")
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
