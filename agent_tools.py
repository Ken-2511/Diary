import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np


MAX_CHUNKS_PER_DIARY = 2

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": "Search old diary chunks by semantic similarity, optionally weighted by recency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "half_life_days": {"type": ["integer", "null"], "default": None},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbor_chunks",
            "description": "Read chunks around a matched chunk in the same diary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diary_id": {"type": "string"},
                    "chunk_id": {"type": "integer"},
                    "before": {"type": "integer", "default": 1},
                    "after": {"type": "integer", "default": 1},
                },
                "required": ["diary_id", "chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diary",
            "description": "Read a full diary. Comments are excluded unless include_comment is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diary_id": {"type": "string"},
                    "include_comment": {"type": "boolean", "default": False},
                },
                "required": ["diary_id"],
            },
        },
    },
]


def load_chunk_index(diary_root: Path, target_id: str) -> list[dict]:
    records = []
    for diary_dir in sorted(diary_root.iterdir()):
        chunks_path = diary_dir / "chunks.json"
        embeddings_path = diary_dir / "chunk_embeddings.npz"
        if not diary_dir.is_dir() or diary_dir.name == target_id or not chunks_path.exists() or not embeddings_path.exists():
            continue
        meta = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)["embeddings"]
        records += [{
            "diary_id": diary_dir.name,
            "date": diary_dir.name[:10],
            "title": meta.get("title"),
            "diary_token_count": meta.get("diary_token_count"),
            "comment_token_count": meta.get("comment_token_count"),
            "chunk_id": chunk["chunk_id"],
            "chunk_token_count": chunk["chunk_token_count"],
            "text": chunk["text"],
            "embedding": embeddings[chunk["embedding_index"]],
        } for chunk in meta["chunks"]]
    return records


def search_chunks(query: str, records: list[dict], target_id: str, embed: Callable[[list[str]], np.ndarray], top_k: int = 5, half_life_days: int | None = None) -> list[dict]:
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


def get_neighbor_chunks(diary_root: Path, diary_id: str, chunk_id: int, before: int = 1, after: int = 1) -> dict:
    meta = json.loads((diary_root / diary_id / "chunks.json").read_text(encoding="utf-8"))
    return {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "title": meta.get("title"),
        "diary_token_count": meta.get("diary_token_count"),
        "comment_token_count": meta.get("comment_token_count"),
        "center_chunk_id": chunk_id,
        "chunks": [c for c in meta["chunks"] if chunk_id - before <= c["chunk_id"] <= chunk_id + after],
    }


def get_diary(diary_root: Path, diary_id: str, diary_name: str, comment_name: str, include_comment: bool = False) -> dict:
    diary_dir = diary_root / diary_id
    meta = json.loads((diary_dir / "chunks.json").read_text(encoding="utf-8"))
    result = {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "title": meta.get("title"),
        "diary_token_count": meta.get("diary_token_count"),
        "comment_token_count": meta.get("comment_token_count"),
        "text": (diary_dir / diary_name).read_text(encoding="utf-8").strip(),
    }
    if include_comment and (diary_dir / comment_name).exists():
        result["comment"] = (diary_dir / comment_name).read_text(encoding="utf-8").strip()
    return result


def run_tool_call(tool_call: dict, records: list[dict], target_id: str, diary_root: Path, diary_name: str, comment_name: str, embed: Callable[[list[str]], np.ndarray]) -> dict:
    function = tool_call.get("function") or {}
    try:
        name = function.get("name", "")
        args = json.loads(function.get("arguments") or "{}")
        if name == "search_chunks":
            return {"results": search_chunks(str(args["query"]), records, target_id, embed, int(args.get("top_k", 5)), args.get("half_life_days"))}
        if name == "get_neighbor_chunks":
            return get_neighbor_chunks(diary_root, str(args["diary_id"]), int(args["chunk_id"]), int(args.get("before", 1)), int(args.get("after", 1)))
        if name == "get_diary":
            return get_diary(diary_root, str(args["diary_id"]), diary_name, comment_name, bool(args.get("include_comment", False)))
        return {"error": f"Unknown tool: {name}", "got": args}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid tool arguments JSON: {e}", "got": function.get("arguments")}
    except Exception as e:
        return {"error": f"Tool call failed: {type(e).__name__}: {e}", "got": function.get("arguments")}
