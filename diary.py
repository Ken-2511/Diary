def main():
    import os
    import tomllib
    from datetime import datetime

    # get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # load config
    with open(os.path.join(base_dir, 'config/config.toml'), 'rb') as f:
        config = tomllib.load(f)

    # create the diary file
    now = datetime.now()
    dir_name = os.path.join(config['diary_dir'], now.strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(dir_name)
    with open(os.path.join(dir_name, config['diary_name']), 'w') as f:
        pass

    # open text app
    os.system(f"{config['text_app']} \"{os.path.join(dir_name, config['diary_name'])}\"")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        input('Press Enter to continue...')
