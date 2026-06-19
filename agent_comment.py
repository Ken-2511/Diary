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
TEMP_DIR = Path("temp")
COMMENT_PATH = TEMP_DIR / "agent_comment.txt"
HISTORY_PATH = TEMP_DIR / "agent_reasoning_history.json"
MAX_STEPS = 15
MAX_CHUNKS_PER_DIARY = 2


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
    return {"name": name, "messages": messages, "content": "".join(content), "reasoning": "".join(reasoning)}


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


def parse_action(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON action found: {text}")
    return json.loads(match.group(0))


def run_action(action: dict, records: list[dict], target_id: str) -> dict:
    name = action.get("action")
    if name == "search_chunks":
        return {"results": search_chunks(
            action["query"],
            records,
            target_id,
            top_k=int(action.get("top_k", 5)),
            half_life_days=action.get("half_life_days"),
        )}
    if name == "get_neighbor_chunks":
        return get_neighbor_chunks(
            action["diary_id"],
            int(action["chunk_id"]),
            before=int(action.get("before", 1)),
            after=int(action.get("after", 1)),
        )
    if name == "get_diary":
        return get_diary(action["diary_id"], include_comment=bool(action.get("include_comment", False)))
    if name == "final_comment":
        return {"comment": action["comment"]}
    raise ValueError(f"Unknown action: {name}")


def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    TEMP_DIR.mkdir(exist_ok=True)
    diary_dirs = sorted(p for p in DIARY_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", p.name))
    target = diary_dirs[-1]
    title = (target / "title.txt").read_text(encoding="utf-8").strip()
    diary = (target / "diary.txt").read_text(encoding="utf-8").strip()
    records = load_chunk_index(target.name)
    history = {
        "target_diary_id": target.name,
        "target_title": title,
        "model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "steps": [],
    }

    system_prompt = (
        "你是一个日记评论 agent。你可以一步一步调用工具搜索旧日记，最后写评论。"
        "每轮只输出一个 JSON object，不要 markdown。"
        "可用 action："
        "{\"action\":\"search_chunks\",\"query\":\"...\",\"top_k\":5,\"half_life_days\":null或整数天数}；"
        "{\"action\":\"get_neighbor_chunks\",\"diary_id\":\"...\",\"chunk_id\":0,\"before\":1,\"after\":1}；"
        "{\"action\":\"get_diary\",\"diary_id\":\"...\",\"include_comment\":false}；"
        "{\"action\":\"final_comment\",\"comment\":\"...\"}。"
        "搜索结果里 diary_token_count 小时可以读全文；include_comment 默认 false。"
        "引用旧日记时要温柔克制，旧记忆用于陪伴和理解，不要审判用户。"
    )

    state = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"目标标题：{title}\n\n目标日记：\n{diary}"}]
    final_comment = ""
    for step_num in range(1, MAX_STEPS + 1):
        print(f"\nAgent step {step_num}...")
        call = chat_stream(state, f"step_{step_num}", json_mode=True)
        action = parse_action(call["content"])
        result = run_action(action, records, target.name)
        history["steps"].append({"step": step_num, "llm": call, "action": action, "result": result})
        if action.get("action") == "final_comment":
            final_comment = result["comment"]
            break
        state.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        state.append({"role": "user", "content": "工具结果：\n" + json.dumps(result, ensure_ascii=False)})

    if not final_comment:
        raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")

    COMMENT_PATH.write_text(final_comment, encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {COMMENT_PATH}")
    print(f"saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
