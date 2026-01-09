# search for the most similar directory

import os
import re
import tomllib
import numpy as np
from openai import OpenAI

# load config
with open(os.path.join("..", "config", "config.toml"), "rb") as f:
    config = tomllib.load(f)

client = OpenAI()

def get_embedding_dir(dir_name: str) -> np.ndarray:
    path = os.path.join(config["diary_dir"], dir_name, config["embedding_name"])
    return np.load(path)

def get_embedding_query(query: str) -> np.ndarray:
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def get_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# load query
query = input("Enter the query: ")
query_vec = get_embedding_query(query)

# load all dirs
dir_names = [name for name in os.listdir(config["diary_dir"]) if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name)]
dirs = [{'name': dn} for dn in dir_names]

# get embedding and similarity
for dir in dirs:
    dir['vec'] = get_embedding_dir(dir['name'])
    dir['similarity'] = get_similarity(dir['vec'], query_vec)

# sort
dirs.sort(key=lambda x: x['similarity'], reverse=True)

# print the most similar dir
print(dirs[0]['name'])

# open the most similar dir
most_similar_dir_path = os.path.join(config["diary_dir"], dirs[0]['name'])
os.startfile(most_similar_dir_path)
