# coding:utf-8
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

"""cfg变量中保存的是默认config，yaml文件保存变动性config"""
__C = edict()
cfg = __C  # 引用传递

__C.GPUS = '2'  # 末尾无逗号
__C.TRAIN_NO = '000'  # 对应log,model存放在logs_0,model_0中，默认情况下是logs_0

# ------Input configuration-------#
__C.INPUT = edict()
__C.INPUT.BASE_INPUT_PATH = '/home/poac/lyx/Project/ECA-LSTM/18_1_Volumn1'  # 输入数据的路径
# __C.INPUT.BASE_INPUT_PATH = '/home/poac/lyx/Project/LSTM/test'  # 输入数据的路径
__C.INPUT.RESTORE_MODEL = False
# __C.INPUT.SEQ_LEN1 = 20
__C.INPUT.SEQ_LEN = 40
# __C.INPUT.SEQ_LEN3 = 5
__C.INPUT.GT_LEN = 1
__C.INPUT.STEP_TRAIN = 1  # 每次输入数据的滑动间隔
__C.INPUT.STEP_TEST = 1
__C.INPUT.PATH_TO_ATTEN_MODEL = ''
__C.INPUT.JOINT_PATH_TO_ATTEN_MODEL = ''
__C.INPUT.JOINT_PATH_TO_ERR_MODEL = ''

# ------Output configuration-------#
__C.OUTPUT = edict()
__C.OUTPUT.BASE_OUTPUTS_PATH = '/home/poac/lyx/Project/ECA-LSTM/out_new'
# __C.OUTPUT.BASE_OUTPUTS_PATH = '/home/poac/lyx/Project/LSTM/out_3MixModel'
__C.OUTPUT.SUMMARY_DIR = 'logs'
__C.OUTPUT.MODEL_DIR = 'models'
__C.OUTPUT.SAVED_MODEL_PATTERN = 'ECA-LSTM.ckpt'
__C.OUTPUT.PRINT_LOG_INTERVAL = 10
__C.OUTPUT.VALID_SAVE_INTERVAL = 1000
__C.OUTPUT.SUM_WRITE_INTERVAL = 5
__C.OUTPUT.MAX_MODELS_TO_KEEP = 100
__C.OUTPUT.BASE_PLOT_PATH = '/home/poac/lyx/Project/ECA-LSTM/plots/prediction_new'

# ------Training configuraton-------#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 10
__C.TRAIN.LEARNING_RATE_BASE = 0.01
# _C.TRAIN.DECAY_STEP = 300
# __C.TRAIN.DECAY_STEP = 10000
__C.TRAIN.DECAY_STEP = 3000
__C.TRAIN.DECAY_RATE = 0.9
__C.TRAIN.LAM_LP = 0.5
__C.TRAIN.LAM_ADV = 0.5
__C.TRAIN.LAM_LP_ERASE = 1.0
__C.TRAIN.LAM_GD_ERASE = 1.0
__C.TRAIN.LAM_ADV_ERASE = 0.05
__C.TRAIN.EPSILON = 0.07

# ------Valid configuraton-------#
__C.VALID = edict()
__C.VALID.BATCH_SIZE = 1
__C.VALID.RESTORE_FROM = 'out/models_0/LSTM.ckpt-91000'

# ------Network configuraton-------#
__C.NETWORK = edict()
__C.NETWORK.HIDDEN_SIZE = 100
__C.NETWORK.DROPOUT_KEEP_PROB = 0.5

# ------Test configuraton-------#
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 1
__C.TEST.RESTORE_FROM = 'out/models_0/LSTM.ckpt-91000'


def get_output_dir(config_file_name):
  """
  Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.
  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.EXP_DIR, osp.basename(config_file_name)))
  if not osp.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """
  Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  # for k, v in a.iteritems(): # python2
  for k, v in a.items():
    # a must specify keys that are in b
    # if not b.has_key(k): # python2
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)
