# search for the most similar directory

import os
import re
import tomllib
import numpy as np
import google.generativeai as genai

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of script_dir)
project_root = os.path.dirname(script_dir)

# load config
config_path = os.path.join(project_root, "config", "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

genai.configure()

def get_embedding_dir(dir_name: str) -> np.ndarray:
    path = os.path.join(config["diary_dir"], dir_name, config["embedding_name"])
    return np.load(path)

def get_embedding_query(query: str) -> np.ndarray:
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    return np.array(response["embedding"], dtype=np.float32)

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