# create a title for the diary

import os
import tomllib
import google.generativeai as genai
from datetime import datetime
from utils.load_diary import load_diary_entry, entry_to_content

__all__ = ['create_title']

# load config
with open(os.path.join('config', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

genai.configure()

def create_title(dir_name: str) -> str:
    entry = load_diary_entry(config['diary_dir'], dir_name, config)
    content = entry_to_content(entry)

    with open(os.path.join('config', 'title.prompt.md'), 'r', encoding='utf-8') as f:
        title_sys_prompt = f.read()

    model = genai.GenerativeModel(
        model_name=config['model4title'],
        system_instruction=title_sys_prompt
    )
    
    response = model.generate_content(content)
    title = response.text
    
    if not title:
        raise ValueError("Failed to generate title: received None from API")
    return title.strip().strip('"')


if __name__ == '__main__':
    dir_name = "2025-08-12-11-25-37"
    title = create_title(dir_name)
    print(title)