import os
import re
import tomllib
from utils.request_llm import request_llm

# load config
with open('config/config.toml', 'rb') as f:
    config = tomllib.load(f)

# load all diary dirs
dir_names = os.listdir(config['diary_dir'])
dir_names = [name for name in dir_names if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name)]
dir_names.sort(key=lambda x: x.split('-'))

# get the last diary dir that has no comment