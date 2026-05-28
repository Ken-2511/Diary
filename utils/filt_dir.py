# filter out the unrelevant diaries

import os
import math
from datetime import datetime, timedelta
import tomllib
import numpy as np
from google import genai
from google.genai import types

__all__ = ['filt_dir']

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load the config
with open(os.path.join(base_dir, 'config', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

# Initialize the client
client = genai.Client()

def get_embedding(dir_name: str) -> np.ndarray:
    path = os.path.join(config['diary_dir'], dir_name, config['embedding_name'])
    if os.path.exists(path):
        return np.load(path)
    else:
        with open(os.path.join(config['diary_dir'], dir_name, config['diary_name']), 'r', encoding='utf-8') as f:
            content = f.read()
        response = client.models.embed_content(
            model='gemini-embedding-2-preview',
            contents=content,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        vec = response.embeddings[0].values
        vec = np.array(vec, dtype=np.float32)
        np.save(path, vec)
        return vec


def filt_dir(dir_names: list[str], target_dir_name: str, top_n: int) -> list[str]:
    dirs = [{'name': dn} for dn in dir_names]
    # similarity score
    target_dir_vec = get_embedding(target_dir_name)
    for dir in dirs:
        dir['vec'] = get_embedding(dir['name'])
        dir['similarity'] = get_similarity(dir['vec'], target_dir_vec)
    target_dir_time = datetime.strptime(target_dir_name, '%Y-%m-%d-%H-%M-%S')
    # recency score
    for dir in dirs:
        dir['time'] = datetime.strptime(dir['name'], '%Y-%m-%d-%H-%M-%S')
        dir['time_distance'] = abs(dir['time'] - target_dir_time)
        dir['time_distance_days'] = dir['time_distance'] / timedelta(days=1)
            # Time decay using half-life: freshness = exp(-ln(2) * days_ago / half_life)
        dir['recency_score'] = math.exp(-0.693 * dir['time_distance_days'] / config['half_life_days'])
            # total score = similarity score * recency score
        dir['total_score'] = config['vec_score_weight'] * dir['similarity'] + config['recency_weight'] * dir['recency_score']
    # sort
    dirs.sort(key=lambda x: x['total_score'], reverse=True)
    return [dir['name'] for dir in dirs[:top_n]]


def get_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))