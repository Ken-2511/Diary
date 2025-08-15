try:
    import os
    import tomllib
    from datetime import datetime

    # load config
    with open('config/config.toml', 'rb') as f:
        config = tomllib.load(f)

    # create the diary file
    now = datetime.now()
    dir_name = os.path.join(config['diary_dir'], now.strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(dir_name)
    with open(os.path.join(dir_name, config['diary_name']), 'w') as f:
        pass

    # open text app
    os.system(f"{config['text_app']} \"{os.path.join(dir_name, config['diary_name'])}\"")

except Exception as e:
    print(e)
    input('Press Enter to continue...')