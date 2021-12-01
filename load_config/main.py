import argparse
from config import get_config, update_config

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, default=None, help='config file')
    parser.add_argument('-batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-dataset', type=str, default='imagenet', help='dataset')
    return parser.parse_args()



def main():
    cfg = get_config()
    print(cfg)
    print('-----------------')


    cfg = get_config("load_config/a.yaml")
    print(cfg)
    print('-----------------')

    args = get_arguments()
    cfg = update_config(cfg, args)
    print(cfg)
    print('-----------------')


if __name__ == "__main__":
    main()