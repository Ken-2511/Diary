# create a title for the diary

import os
import tomllib
from google import genai
from google.genai import types
from datetime import datetime
from utils.load_diary import load_diary_entry, entry_to_content

__all__ = ['create_title']

# load config
with open(os.path.join('config', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

def create_title(dir_name: str) -> str:
    entry = load_diary_entry(config['diary_dir'], dir_name, config)
    content_raw = entry_to_content(entry)
    if isinstance(content_raw, list):
        content = []
        for p in content_raw:
            if isinstance(p, dict) and 'data' in p:
                content.append(types.Part.from_bytes(data=p['data'], mime_type=p['mime_type']))
            else:
                content.append(str(p))
    else:
        content = str(content_raw)

    with open(os.path.join('config', 'title.prompt.md'), 'r', encoding='utf-8') as f:
        title_sys_prompt = f.read()

    client = genai.Client()
    response = client.models.generate_content(
        model=config['model4title'],
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=title_sys_prompt
        )
    )
    title = response.text
    if not title:
        raise ValueError("Failed to generate title: received None from API")
    return title.strip().strip('"')


if __name__ == '__main__':
    dir_name = "2025-08-12-11-25-37"
    title = create_title(dir_name)
    print(title)