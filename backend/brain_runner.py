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


def _emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def stream_gemini(payload: dict[str, Any]) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=payload["api_key"])
    config_kwargs: dict[str, Any] = {}
    if payload.get("max_output_tokens"):
        config_kwargs["max_output_tokens"] = int(payload["max_output_tokens"])
    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
    usage: dict[str, int] = {}
    for chunk in client.models.generate_content_stream(
        model=payload["model"], contents=payload["prompt"], config=config
    ):
        text = getattr(chunk, "text", None)
        if text:
            _emit({"token": text})
        meta = getattr(chunk, "usage_metadata", None)
        if meta:
            usage = {
                "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
                "total_tokens": int(getattr(meta, "total_token_count", 0) or 0),
            }
    _emit({"done": True, "usage": usage})


def run_streaming(payload: dict[str, Any]) -> None:
    """Streaming chat mode. Real token streaming for Gemini; buffered single-chunk
    fallback for Codex/Claude so all linked brains work as cloud chat models."""
    provider = payload["provider"]
    if provider == "gemini":
        stream_gemini(payload)
        return
    result = run_codex(payload) if provider == "codex" else asyncio.run(run_claude(payload))
    _emit({"token": result.get("content", "")})
    _emit({"done": True, "usage": result.get("usage") or {}})


def main() -> None:
    payload = json.loads(sys.stdin.read())
    streaming = bool(payload.get("stream"))
    try:
        provider = payload["provider"]
        if provider not in {"codex", "claude", "gemini"}:
            raise ValueError("Unsupported brain provider")
        if streaming:
            run_streaming(payload)
            return
        if provider == "codex":
            result = run_codex(payload)
        elif provider == "claude":
            result = asyncio.run(run_claude(payload))
        else:
            result = run_gemini(payload)
        _emit({"ok": True, **result})
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}"}
        _emit({**error, "done": True} if streaming else {"ok": False, **error})
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
