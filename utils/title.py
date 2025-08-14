# create a title for the diary

import os
import tomllib
from openai import OpenAI

__all__ = ['create_title']

# load config
with open('config/config.toml', 'rb') as f:
    config = tomllib.load(f)

client = OpenAI()


def create_title(dir_name: str) -> str:
    with open(os.path.join(config['diary_dir'], dir_name, config['diary_name']), 'r', encoding='utf-8') as f:
        diary = f.read()
    
    with open(os.path.join('config', 'title.prompt.md'), 'r', encoding='utf-8') as f:
        title_sys_prompt = f.read()

    response = client.chat.completions.create(
        model=config['model4title'],
        messages=[
            {"role": "system", "content": title_sys_prompt},
            {"role": "user", "content": diary}
        ]
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    dir_name = "2025-08-12-11-25-37"
    title = create_title(dir_name)
    print(title)