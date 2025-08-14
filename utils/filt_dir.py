# filter out the unrelevant diaries

import os
import math
from datetime import datetime, timedelta
import tomllib
import numpy as np
from openai import OpenAI

__all__ = ['filt_dir']

# load the config
with open(os.path.join('config', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

# load the embedding model
client = OpenAI()


def get_embedding(dir_name: str) -> np.ndarray:
    path = os.path.join(config['diary_dir'], dir_name, config['embedding_name'])
    if os.path.exists(path):
        return np.load(path)
    else:
        response = client.embeddings.create(
            input=dir_name,
            model="text-embedding-3-large"
        )
        vec = response.data[0].embedding
        vec = np.array(vec, dtype=np.float32)
        np.save(path, vec)
        return vec


def filt_dir(dir_names: list[str], target_dir_name: str, top_n: int) -> list[str]:
    dirs = [{'name': dn} for dn in dir_names]
    # embedding
    target_dir_vec = get_embedding(target_dir_name)
    for dir in dirs:
        dir['vec'] = get_embedding(dir['name'])
        dir['similarity'] = get_similarity(dir['vec'], target_dir_vec)
    # time distance
    target_dir_time = datetime.strptime(target_dir_name, '%Y-%m-%d-%H-%M-%S')
    for dir in dirs:
        dir['time'] = datetime.strptime(dir['name'], '%Y-%m-%d-%H-%M-%S')
        dir['time_distance'] = abs(dir['time'] - target_dir_time)
        dir['time_distance_days'] = dir['time_distance'] / timedelta(days=1)
    # score
    for dir in dirs:
        dir['vec_score'] = 15 * dir['similarity']
        dir['time_score'] = -1 * math.log(dir['time_distance_days'] + 1)
        dir['total_score'] = dir['vec_score'] + dir['time_score']
    # sort
    dirs.sort(key=lambda x: x['total_score'], reverse=True)
    return [dir['name'] for dir in dirs[:top_n]]


def get_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))