import os.path as osp
import numpy as np
from easydict import EasyDict
from cmom.utils import project_root, project_root_ADVENT
from advent.utils.serialization import yaml_load

cfg = EasyDict()

# COMMON CONFIGS (the same as Advent)
cfg.NUM_WORKERS = 4
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
cfg.GPU_ID = 0
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = str(project_root / 'pretrained_models/davsn_viper2city_pretrained.pth')
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 40000
cfg.TRAIN.SAVE_PRED_EVERY = 1000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

# DA-VSN
cfg.TRAIN.DA_METHOD = 'CMOM'
cfg.NUM_CLASSES = 15
cfg.SOURCE = 'Viper'
cfg.TARGET = 'CityscapesSeq'
cfg.TRAIN.INFO_TARGET = str(project_root / 'cmom/dataset/CityscapesSeq_list/info_Viper.json')
cfg.DATA_LIST_SOURCE = str(project_root / 'cmom/dataset/Viper_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'cmom/dataset/CityscapesSeq_list/{}.txt')
cfg.DATA_DIRECTORY_SOURCE = '/mnt/hdd1/kscho/Viper'
cfg.DATA_DIRECTORY_TARGET = '/mnt/hdd1/kscho/Cityscapes'
cfg.TRAIN.MODEL = 'ACCEL_DeepLabv2'
cfg.TRAIN.flow_path_src = '/mnt/hdd1/kscho/Viper/Estimated_optical_flow_Viper_train'
cfg.TRAIN.flow_path = '/mnt/hdd1/kscho/Cityscapes/estimated_optical_flow_cityscapes_train'
cfg.TRAIN.lamda_sa = 1.0

# CMOM
cfg.TRAIN.pseudo_label_path = str(project_root / 'cmom/pseudo_label_viper2city')
cfg.TRAIN.MIX = 0.75
# FATC
cfg.TRAIN.lamda_FA = 0.01

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'video_best'  # {'single', 'best'}
cfg.TEST.MODEL = ('ACCEL_DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 1000  # used in 'best' mode
cfg.TEST.SNAPSHOT_START_ITER = 1000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 40000  # used in 'best' mode
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1 
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'cmom/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True
cfg.TEST.flow_path = '/mnt/hdd1/kscho/Cityscapes/estimated_optical_flow_cityscapes_val'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
