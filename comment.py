import io
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import requests

from agent_tools import SCHEMAS, load_chunk_index, run_tool_call


CHAT_MODEL = "moonshotai/kimi-k2.6"
EMBED_MODEL = "google/gemini-embedding-2"
CONFIG_DIR = Path("config")
CONFIG = tomllib.loads((CONFIG_DIR / "config.toml").read_text(encoding="utf-8"))
DIARY_ROOT = Path(CONFIG["diary_dir"])
DIARY_NAME = CONFIG.get("diary_name", "diary.txt")
TITLE_NAME = CONFIG.get("title_name", "title.txt")
COMMENT_NAME = CONFIG.get("comment_name", "comment.txt")
TEMP_DIR = Path("temp")
COMMENT_PATH = TEMP_DIR / "comment.txt"
HISTORY_PATH = TEMP_DIR / "agent_reasoning_history.json"
MAX_STEPS = 15


def read_prompt(name: str) -> str:
    return (CONFIG_DIR / name).read_text(encoding="utf-8").strip()


def headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    result = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if site := os.environ.get("OPENROUTER_SITE_URL"):
        result["HTTP-Referer"] = site
    if title := os.environ.get("OPENROUTER_APP_NAME"):
        result["X-Title"] = title
    return result


def chat_stream(messages: list[dict], name: str) -> dict:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    api_messages = [{k: v for k, v in message.items() if k in allowed} for message in messages]
    payload = {
        "model": CHAT_MODEL,
        "messages": api_messages,
        "tools": SCHEMAS,
        "tool_choice": "auto",
        "stream": True,
        "reasoning": {"enabled": True, "effort": "medium", "exclude": False},
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers(),
        json=payload,
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    response.encoding = "utf-8"

    content, reasoning, buffer, tool_calls, done = [], [], "", {}, False
    for piece in response.iter_content(chunk_size=4096, decode_unicode=True):
        buffer += piece
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data = "\n".join(line[5:].strip() for line in event.split("\n") if line.startswith("data:"))
            if not data:
                continue
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                print(delta["content"], end="", flush=True)
                content.append(delta["content"])
            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", len(tool_calls))
                saved = tool_calls.setdefault(index, {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tool_call.get("id"):
                    saved["id"] += tool_call["id"]
                if tool_call.get("type"):
                    saved["type"] = tool_call["type"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    saved["function"]["name"] += function["name"]
                if function.get("arguments"):
                    saved["function"]["arguments"] += function["arguments"]
            if delta.get("reasoning"):
                reasoning.append(delta["reasoning"])
            elif delta.get("reasoning_details"):
                for part in delta["reasoning_details"]:
                    if isinstance(part, dict) and part.get("text"):
                        reasoning.append(part["text"])
        if done:
            break
    print()
    return {
        "name": name,
        "messages": messages,
        "content": "".join(content).strip(),
        "reasoning": "".join(reasoning),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
    }


def embed(texts: list[str]) -> np.ndarray:
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=headers(),
        json={"model": EMBED_MODEL, "input": texts, "encoding_format": "float"},
        timeout=120,
    )
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
    return np.array([item["embedding"] for item in data], dtype=np.float32)


def save_history(messages: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    TEMP_DIR.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(Path("tools") / "build_diary_chunks.py")], check=True)
    diary_dirs = sorted(
        p for p in DIARY_ROOT.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", p.name)
    )
    targets = [p for p in diary_dirs if (p / DIARY_NAME).exists() and not (p / COMMENT_NAME).exists()]
    if not targets:
        raise RuntimeError("No diary without comment found")
    target = targets[0]
    title = (target / TITLE_NAME).read_text(encoding="utf-8").strip() if (target / TITLE_NAME).exists() else ""
    diary = (target / DIARY_NAME).read_text(encoding="utf-8").strip()
    records = load_chunk_index(DIARY_ROOT, target.name)
    state = [
        {"role": "system", "content": read_prompt("agent_system.prompt.md")},
        {"role": "user", "content": f"Target diary id:\n{target.name}\n\nTarget date:\n{target.name[:10]}\n\nTarget time:\n{target.name[11:].replace('-', ':')}\n\nTarget title:\n{title}\n\nTarget diary:\n{diary}"},
    ]
    save_history(state)

    final_comment = ""
    for step_num in range(1, MAX_STEPS + 1):
        print(f"\nAgent step {step_num}...")
        call = chat_stream(state, f"step_{step_num}")
        assistant_message = {"role": "assistant", "content": call["content"] or None}
        if call["reasoning"]:
            assistant_message["reasoning"] = call["reasoning"]
        if call["tool_calls"]:
            assistant_message["tool_calls"] = call["tool_calls"]
        state.append(assistant_message)
        if not call["tool_calls"]:
            final_comment = call["content"]
            save_history(state)
            break
        for tool_call in call["tool_calls"]:
            result = run_tool_call(tool_call, records, target.name, DIARY_ROOT, DIARY_NAME, COMMENT_NAME, embed)
            state.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })
        save_history(state)

    if not final_comment:
        save_history(state)
        raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")

    COMMENT_PATH.write_text(final_comment, encoding="utf-8")
    (target / COMMENT_NAME).write_text(final_comment, encoding="utf-8")
    save_history(state)
    print(f"saved: {COMMENT_PATH}")
    print(f"saved: {target / COMMENT_NAME}")
    print(f"saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
