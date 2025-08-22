# Load all diaries and comments into a single file

print("starting to load diaries...")

import os
import re
import tomllib

def read_diary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_comment(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# load config
with open('config/config.toml', 'rb') as f:
    config = tomllib.load(f)

# load all dir names
dir_names = os.listdir(config['diary_dir'])
dir_names = [name for name in dir_names if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name)]
dir_names.sort(key=lambda x: x.split('-'))

# load all content
all_content = ""
for dir_name in dir_names:
    print(f"loading {dir_name}")
    # for each dir, read the diary
    y, mon, d, h, m, s = [int(i) for i in dir_name.split('-')]
    all_content += f"(date: {y}.{mon}.{d}, time: {h}:{m})\n"
    all_content += f"### Diary ###\n\n"
    diary = read_diary(os.path.join(config['diary_dir'], dir_name, config['diary_name']))
    all_content += diary + '\n\n'
    # read the comment
    comment = read_comment(os.path.join(config['diary_dir'], dir_name, config['comment_name']))
    all_content += f"### Comment ###\n\n"
    all_content += comment + '\n\n\n'
    all_content += f"--------------------------------\n\n"

with open(os.path.join(config['prj_dir'], "temp", "all.txt"), "w", encoding="utf-8") as file:
    file.write(all_content)

print("done")