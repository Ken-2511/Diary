import os
import re
import json
import time
import tomllib
from utils.title import create_title
from utils.request_llm import request_llm

# load config
print('Loading config...')
with open('config/config.toml', 'rb') as f:
    config = tomllib.load(f)

# load all diary dirs
print('Loading diary dirs...')
dir_names = os.listdir(config['diary_dir'])
dir_names = [name for name in dir_names if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name)]
dir_names.sort(key=lambda x: x.split('-'))

# get the first diary dir that has no comment (because this is the one that we are going to comment)
print('Getting the first diary dir that has no comment...')
target_dir_name = None
for dn in dir_names:
    if not os.path.exists(os.path.join(config['diary_dir'], dn, config['comment_name'])):
        target_dir_name = dn
        dir_names = dir_names[:dir_names.index(dn)]  # remove the target dir and all the following dirs
        break
if target_dir_name is None:
    print('All diaries have been commented')
    input('Press Enter to exit...')
    exit()

# load the diaries, currently we only load the last 10 diaries
print('Loading diaries...')
diary_list = []
for dn in dir_names[-10:]:
    with open(os.path.join(config['diary_dir'], dn, config['diary_name']), 'r', encoding='utf-8') as f:
        diary_list.append(f.read())

# create and save the title
print('Creating and saving the title...')
title = create_title(target_dir_name)
with open(os.path.join(config['diary_dir'], target_dir_name, config['title_name']), 'w', encoding='utf-8') as f:
    f.write(title)
print(f"Title saved: {title}")

# prepare the messages for the LLM
print('Preparing the messages for the LLM...')
with open(os.path.join('config', 'init_sys.prompt.md'), 'r', encoding='utf-8') as f:
    init_sys_prompt = f.read()
with open(os.path.join('config', 'last_diary.prompt.md'), 'r', encoding='utf-8') as f:
    last_diary_prompt = f.read()
with open(os.path.join(config['diary_dir'], target_dir_name, config['diary_name']), 'r', encoding='utf-8') as f:
    target_diary = f.read()
messages = [
    {"role": "system", "content": init_sys_prompt},
    {"role": "system", "content": last_diary_prompt},
    {"role": "user", "content": target_diary}
]
for diary in diary_list:
    messages.insert(-2, {"role": "user", "content": diary})
# save the messages to the temp folder (for debugging purposes)
with open(os.path.join(config['prj_dir'], 'temp', 'messages.json'), 'w', encoding='utf-8') as f:
    json.dump(messages, f, ensure_ascii=False, indent=4)

# request the LLM
print('Requesting the LLM...')
start_time = time.time()
response, usage = request_llm(messages, config['model'])
end_time = time.time()

# save the response
with open(os.path.join(config['diary_dir'], target_dir_name, config['comment_name']), 'w', encoding='utf-8') as f:
    f.write(response)

# save the usage
with open(os.path.join(config['diary_dir'], target_dir_name, config['usage_name']), 'w', encoding='utf-8') as f:
    json.dump(usage, f, ensure_ascii=False, indent=4)

# print the important information and exit
print(f"The comment has been saved to {os.path.join(config['diary_dir'], target_dir_name, config['comment_name'])}")
print(f"Input tokens: {usage['prompt_tokens']}, output tokens: {usage['completion_tokens']}, total tokens: {usage['total_tokens']}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
input('Press Enter to exit...')