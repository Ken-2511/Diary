# create a title for the diary

import os
import tomllib
from utils.load_diary import load_diary_entry, entry_to_content
from utils.request_llm import request_llm

__all__ = ['create_title']

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load config
with open(os.path.join(base_dir, 'config', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

def create_title(dir_name: str) -> str:
    entry = load_diary_entry(config['diary_dir'], dir_name, config)
    content = entry_to_content(entry)

    with open(os.path.join(base_dir, 'config', 'title.prompt.md'), 'r', encoding='utf-8') as f:
        title_sys_prompt = f.read()

    title, _ = request_llm([
        {'role': 'system', 'content': title_sys_prompt},
        {'role': 'user', 'content': content},
    ], config['model4title'])
    if not title:
        raise ValueError("Failed to generate title: received None from API")
    return title.strip().strip('"')


if __name__ == '__main__':
    dir_name = "2025-08-12-11-25-37"
    title = create_title(dir_name)
    print(title)
