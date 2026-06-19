import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests


CHAT_MODEL = "moonshotai/kimi-k2.6"
EMBED_MODEL = "google/gemini-embedding-2"
DIARY_ROOT = Path("testing_diaries")
CONFIG_DIR = Path("config")
TEMP_DIR = Path("temp")
COMMENT_PATH = TEMP_DIR / "agent_comment.txt"
HISTORY_PATH = TEMP_DIR / "agent_reasoning_history.json"
MAX_STEPS = 15
MAX_CHUNKS_PER_DIARY = 2


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


def chat_stream(messages: list[dict], name: str, json_mode: bool = False) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "reasoning": {"enabled": True, "effort": "medium", "exclude": False},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers(),
        json=payload,
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    response.encoding = "utf-8"

    content, reasoning, buffer, done = [], [], "", False
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
            if delta.get("reasoning"):
                reasoning.append(delta["reasoning"])
            elif delta.get("reasoning_details"):
                for part in delta["reasoning_details"]:
                    if isinstance(part, dict) and part.get("text"):
                        reasoning.append(part["text"])
        if done:
            break
    print()
    return {"name": name, "messages": messages, "content": "".join(content).strip(), "reasoning": "".join(reasoning)}


def ask_action(state: list[dict], step_name: str) -> dict:
    call = chat_stream(state, step_name, json_mode=True)
    if call["content"].strip():
        return call
    retry_state = state + [{"role": "user", "content": read_prompt("agent_retry.prompt.md")}]
    retry = chat_stream(retry_state, f"{step_name}_retry", json_mode=True)
    retry["messages"] = retry_state
    return retry


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


def load_chunk_index(target_id: str) -> list[dict]:
    records = []
    for diary_dir in sorted(DIARY_ROOT.iterdir()):
        if not diary_dir.is_dir() or diary_dir.name == target_id:
            continue
        chunks_path, embeddings_path = diary_dir / "chunks.json", diary_dir / "chunk_embeddings.npz"
        if not chunks_path.exists() or not embeddings_path.exists():
            continue
        meta = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)["embeddings"]
        for chunk in meta["chunks"]:
            records.append({
                "diary_id": diary_dir.name,
                "date": diary_dir.name[:10],
                "title": meta.get("title"),
                "diary_token_count": meta.get("diary_token_count"),
                "comment_token_count": meta.get("comment_token_count"),
                "chunk_id": chunk["chunk_id"],
                "chunk_token_count": chunk["chunk_token_count"],
                "text": chunk["text"],
                "embedding": embeddings[chunk["embedding_index"]],
            })
    return records


def search_chunks(query: str, records: list[dict], target_id: str, top_k: int = 5, half_life_days: int | None = None) -> list[dict]:
    q = embed([f"task: search result | query: {query}"])[0]
    q_norm = np.linalg.norm(q)
    target_time = datetime.strptime(target_id, "%Y-%m-%d-%H-%M-%S")
    results = []
    for record in records:
        e = record["embedding"]
        record_time = datetime.strptime(record["diary_id"], "%Y-%m-%d-%H-%M-%S")
        days_distance = abs((target_time - record_time).total_seconds()) / 86400
        similarity_score = float(np.dot(q, e) / (q_norm * np.linalg.norm(e)))
        time_factor = 1.0 if half_life_days is None else 0.5 ** (days_distance / half_life_days)
        results.append({k: v for k, v in record.items() if k != "embedding"} | {
            "similarity_score": similarity_score,
            "days_distance": round(days_distance, 3),
            "half_life_days": half_life_days,
            "time_factor": time_factor,
            "final_score": similarity_score * time_factor,
        })

    picked, counts = [], {}
    for item in sorted(results, key=lambda item: item["final_score"], reverse=True):
        if counts.get(item["diary_id"], 0) >= MAX_CHUNKS_PER_DIARY:
            continue
        counts[item["diary_id"]] = counts.get(item["diary_id"], 0) + 1
        picked.append(item)
        if len(picked) >= top_k:
            break
    return picked


def get_neighbor_chunks(diary_id: str, chunk_id: int, before: int = 1, after: int = 1) -> dict:
    meta = json.loads((DIARY_ROOT / diary_id / "chunks.json").read_text(encoding="utf-8"))
    chunks = [c for c in meta["chunks"] if chunk_id - before <= c["chunk_id"] <= chunk_id + after]
    return {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "title": meta.get("title"),
        "diary_token_count": meta.get("diary_token_count"),
        "comment_token_count": meta.get("comment_token_count"),
        "center_chunk_id": chunk_id,
        "chunks": chunks,
    }


def get_diary(diary_id: str, include_comment: bool = False) -> dict:
    diary_dir = DIARY_ROOT / diary_id
    meta = json.loads((diary_dir / "chunks.json").read_text(encoding="utf-8"))
    result = {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "title": meta.get("title"),
        "diary_token_count": meta.get("diary_token_count"),
        "comment_token_count": meta.get("comment_token_count"),
        "text": (diary_dir / "diary.txt").read_text(encoding="utf-8").strip(),
    }
    if include_comment and (diary_dir / "comment.txt").exists():
        result["comment"] = (diary_dir / "comment.txt").read_text(encoding="utf-8").strip()
    return result


def parse_actions(text: str) -> tuple[list[dict], dict | None]:
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        return [], {"error": "No JSON action found.", "got": text}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return [], {"error": f"Invalid JSON action: {e}", "got": text}

    items = data if isinstance(data, list) else [data]
    if not items:
        return [], {"error": "JSON action list is empty.", "got": data}

    actions = []
    for item in items:
        data = item
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return [], {"error": "JSON action string is not an object.", "got": data}
        if not isinstance(data, dict):
            return [], {"error": "JSON action must be an object.", "got": data}

        if isinstance(data.get("action"), dict):
            data = data["action"]
        name = data.get("action") or data.get("tool") or data.get("name")
        args = data.get("arguments") if "arguments" in data else data.get("args")
        if name and isinstance(args, dict):
            data = args | {"action": name}
        elif name:
            data["action"] = name
        elif "comment" in data and not name:
            data["action"] = "final_comment"
        actions.append(data)
    return actions, None


def run_action(action: dict, records: list[dict], target_id: str) -> dict:
    try:
        name = action.get("action")
        if name == "search_chunks":
            return {"results": search_chunks(
                str(action["query"]),
                records,
                target_id,
                top_k=int(action.get("top_k", 5)),
                half_life_days=action.get("half_life_days"),
            )}
        if name == "get_neighbor_chunks":
            return get_neighbor_chunks(
                str(action["diary_id"]),
                int(action["chunk_id"]),
                before=int(action.get("before", 1)),
                after=int(action.get("after", 1)),
            )
        if name == "get_diary":
            return get_diary(str(action["diary_id"]), include_comment=bool(action.get("include_comment", False)))
        if name == "final_comment":
            return {"comment": str(action["comment"])}
        return {"error": f"Unknown action: {name}", "got": action}
    except Exception as e:
        return {"error": f"Tool call failed: {type(e).__name__}: {e}", "got": action}


def save_history(messages: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    TEMP_DIR.mkdir(exist_ok=True)
    diary_dirs = sorted(p for p in DIARY_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", p.name))
    target = diary_dirs[-1]
    title = (target / "title.txt").read_text(encoding="utf-8").strip()
    diary = (target / "diary.txt").read_text(encoding="utf-8").strip()
    records = load_chunk_index(target.name)
    state = [
        {"role": "system", "content": read_prompt("agent_system.prompt.md")},
        {"role": "user", "content": f"Target title:\n{title}\n\nTarget diary:\n{diary}"},
    ]
    save_history(state)

    final_comment = ""
    for step_num in range(1, MAX_STEPS + 1):
        print(f"\nAgent step {step_num}...")
        call = ask_action(state, f"step_{step_num}")
        actions, parse_error = parse_actions(call["content"])
        results = [parse_error] if parse_error else [run_action(action, records, target.name) for action in actions]
        assistant_message = {"role": "assistant", "content": call["content"]}
        if call["reasoning"]:
            assistant_message["reasoning"] = call["reasoning"]
        state.append(assistant_message)
        for action, result in zip(actions or [None], results):
            if action and action.get("action") == "final_comment" and "error" not in result:
                final_comment = result["comment"]
                break
            state.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        save_history(state)
        if final_comment:
            break

    if not final_comment:
        save_history(state)
        raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")

    COMMENT_PATH.write_text(final_comment, encoding="utf-8")
    save_history(state)
    print(f"saved: {COMMENT_PATH}")
    print(f"saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
