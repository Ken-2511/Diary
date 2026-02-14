def main():
    import os
    import re
    import json
    import time
    import tomllib
    from datetime import datetime
    from utils.title import create_title
    from utils.request_llm import request_llm
    from utils.filt_dir import filt_dir
    from utils.load_diary import load_diary_entry, entry_to_content

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

    # create and save the title
    print('Creating and saving the title...')
    title = create_title(target_dir_name)
    with open(os.path.join(config['diary_dir'], target_dir_name, config['title_name']), 'w', encoding='utf-8') as f:
        f.write(title)
    print(f"Title saved: {title}")

    # filter the diaries
    print('Filtering the diaries...')
    dir_names = filt_dir(dir_names, target_dir_name, config['top_n'])
    dir_names.sort(key=lambda x: x.split('-'))

    # load the diaries and the comments, and add the timestamp to the diaries
    print('Loading diaries...')
    messages = []
    max_chars = config.get('max_history_chars', 0)
    for dn in dir_names:
        entry = load_diary_entry(config['diary_dir'], dn, config)
        messages.append({'role': 'user', 'content': entry_to_content(entry, max_chars=max_chars)})
        comment = entry['comment'] or ''
        if max_chars > 0 and len(comment) > max_chars:
            comment = comment[:max_chars] + '...[truncated]'
        messages.append({'role': 'assistant', 'content': comment})

    # prepare the messages for the LLM
    print('Preparing the messages for the LLM...')
    with open(os.path.join('config', 'init_sys.prompt.md'), 'r', encoding='utf-8') as f:
        init_sys_prompt = f.read()
    with open(os.path.join('config', 'last_diary.prompt.md'), 'r', encoding='utf-8') as f:
        last_diary_prompt = f.read()
    target_entry = load_diary_entry(config['diary_dir'], target_dir_name, config)
    messages.insert(0, {"role": "system", "content": init_sys_prompt})
    messages.append({"role": "system", "content": last_diary_prompt})
    messages.append({"role": "user", "content": entry_to_content(target_entry)})
    # save the messages to the temp folder (for debugging purposes)
    os.makedirs(os.path.join(config['prj_dir'], 'temp'), exist_ok=True)
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


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        input('Press Enter to continue...')
