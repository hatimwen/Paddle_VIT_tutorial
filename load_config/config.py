from yacs.config import CfgNode as CN
import yaml

_C = CN()
_C.DATA = CN()
_C.DATA.DATASET = 'cifar10'
_C.DATA.BATCH_SIZE = 128

_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 10

_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.EMBED_DIM = 96
_C.MODEL.TRANS.DEPTHS = [2, 2, 6, 2]
_C.MODEL.TRANS.QKV_BIAS = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    config.merge_from_file(cfg_file)    # yaml

def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    return config

def get_config(cfg_file=None):
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config


def main():
    cfg = get_config("load_config/a.yaml")
    print(cfg)

if __name__ == "__main__":
    main()