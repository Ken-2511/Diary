import argparse
import hashlib
import json
import logging
import os
import re
import time
import tomllib
from datetime import datetime
from pathlib import Path

import jieba
import numpy as np
import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL = "google/gemini-embedding-2"
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
MIN_TOKENS, TARGET_TOKENS, MAX_TOKENS = 40, 120, 250
jieba.setLogLevel(logging.ERROR)


def count_tokens(text: str) -> int:
    return sum(
        1
        for raw_token in jieba.lcut(text)
        if (token := raw_token.strip()) and not re.fullmatch(r"\W+", token)
    )


def chunk_diary(text: str) -> list[str]:
    chunks, current = [], ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    for paragraph in paragraphs:
        units = [paragraph]
        if count_tokens(paragraph) > MAX_TOKENS:
            units = [
                s.strip()
                for s in re.findall(r".+?(?:[\u3002\uff01\uff1f!?;\uff1b.]|$)", paragraph, re.S)
                if s.strip()
            ]

        for unit in units:
            merged = f"{current}\n\n{unit}".strip()
            if not current or count_tokens(merged) <= TARGET_TOKENS:
                current = merged
            elif count_tokens(current) >= MIN_TOKENS:
                chunks.append(current)
                current = unit
            else:
                current = merged

            if count_tokens(current) >= MAX_TOKENS:
                chunks.append(current)
                current = ""

    return chunks + ([current] if current else [])


def embed_texts(texts: list[str]) -> np.ndarray:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for env_name, header_name in [
        ("OPENROUTER_SITE_URL", "HTTP-Referer"),
        ("OPENROUTER_APP_NAME", "X-OpenRouter-Title"),
    ]:
        if value := os.environ.get(env_name):
            headers[header_name] = value

    for attempt in range(3):
        response = requests.post(
            EMBEDDINGS_URL,
            headers=headers,
            json={"model": MODEL, "input": texts, "encoding_format": "float"},
            timeout=120,
        )
        result = response.json()
        if response.ok and "data" in result:
            data = sorted(result["data"], key=lambda item: item.get("index", 0))
            return np.array([item["embedding"] for item in data], dtype=np.float32)
        print(f"embedding error attempt={attempt + 1}: {response.status_code} {result}", flush=True)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("embedding request failed")


def main():
    parser = argparse.ArgumentParser(description="Build chunks and embeddings for diary folders.")
    parser.add_argument("diary_dir", nargs="?")
    args = parser.parse_args()

    with open(PROJECT_DIR / "config" / "config.toml", "rb") as f:
        config = tomllib.load(f)

    if args.diary_dir:
        diary_dirs = [Path(args.diary_dir)]
    else:
        diary_root = Path(config["diary_dir"])
        diary_dirs = sorted(
            p for p in diary_root.iterdir()
            if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", p.name)
        )

    for diary_dir in diary_dirs:
        diary_path = diary_dir / config["diary_name"]
        if not diary_path.exists():
            print(f"skip missing diary: {diary_dir}")
            continue

        title_path = diary_dir / config["title_name"]
        comment_path = diary_dir / config["comment_name"]
        diary_text = diary_path.read_text(encoding="utf-8")
        diary_hash = hashlib.sha256(diary_text.encode("utf-8")).hexdigest()
        chunks_path, embeddings_path = diary_dir / "chunks.json", diary_dir / "chunk_embeddings.npz"
        if chunks_path.exists() and embeddings_path.exists():
            old = json.loads(chunks_path.read_text(encoding="utf-8"))
            if old.get("source_hash") == diary_hash and old.get("embedding_model") == MODEL:
                print(f"skip current: {diary_dir.name}", flush=True)
                continue

        comment_text = comment_path.read_text(encoding="utf-8") if comment_path.exists() else None
        title = title_path.read_text(encoding="utf-8").strip() if title_path.exists() else None
        chunks = chunk_diary(diary_text)
        embeddings = embed_texts([f"title: {title or 'none'} | text: {chunk}" for chunk in chunks])
        norms = np.linalg.norm(embeddings, axis=1)

        metadata = {
            "version": 1,
            "source": config["diary_name"],
            "source_path": str(diary_path),
            "source_hash": diary_hash,
            "title": title,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diary_token_count": count_tokens(diary_text),
            "comment_token_count": count_tokens(comment_text) if comment_text else None,
            "embedding_model": MODEL,
            "embedding_file": "chunk_embeddings.npz",
            "chunking": {
                "min_tokens": MIN_TOKENS,
                "target_tokens": TARGET_TOKENS,
                "max_tokens": MAX_TOKENS,
            },
            "chunks": [
                {
                    "chunk_id": i,
                    "text": chunk,
                    "chunk_token_count": count_tokens(chunk),
                    "embedding_index": i,
                    "embedding_norm": float(norms[i]),
                }
                for i, chunk in enumerate(chunks)
            ],
        }

        chunks_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            chunk_ids=np.arange(len(chunks), dtype=np.int32),
            model=np.array(MODEL),
        )
        print(f"{diary_dir.name}: chunks={len(chunks)}, embedding_shape={embeddings.shape}", flush=True)


if __name__ == "__main__":
    main()
