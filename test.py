# coding:utf-8
import argparse
import pprint
import os
import sys
import time
import tensorflow as tf
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())
from dataloader_pretrain import Reader
import models
from config import cfg, cfg_from_file, get_output_dir
import matplotlib.pyplot as plt


def _parse_args():
  parser = argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
  args = parser.parse_args()
  return args


def test():
  # -------------parse arguments-------------#
  args = _parse_args()
  trainno = 000
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  # read from args.cfg_file, and integrate into cfg
    trainno = args.cfg_file.split(os.sep)[-1][3:-5]
  pprint.pprint(cfg)  # 漂亮的打印

  # -----------some configurations---------#
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO)  # Sets the threshold for what messages will be logged.

  # -------------construct a graph-------------#
  """建立dataset，获取iterator"""
  with tf.name_scope('dataset'), tf.device('/cpu:0'):
    ite_test, num_file_test, file_to_num_samples, num_samples = Reader(cfg, mode='Test').read_test()
    seq_test, gt_test, path2files = ite_test.get_next()

  """前传"""
  with tf.variable_scope('LSTM', reuse=None):
    outputs_test = models.inference(seq_test, cfg, is_training=False)

  """saver"""
  model_variables = {}
  for variable in tf.global_variables():
    if variable.name.find('LSTM') >= 0:
      model_variables[variable.name.replace(':0', '')] = variable
  print('#####################################################')
  for save_item in model_variables.keys():
    print(save_item)
  print('#####################################################')
  saver = tf.train.Saver(var_list=model_variables, max_to_keep=cfg.OUTPUT.MAX_MODELS_TO_KEEP)

  # -------------Start a session-------------#
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    model_file = cfg.TEST.RESTORE_FROM
    saver.restore(sess, model_file)

    """测试效果"""
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Start validation:')
    # last_file = b'/home/poac/lyx/Project/LSTM/process_oneDataPerHour/20180406_3.txt'
    # last_file = None
    # out_per_file = []
    # count = 1
    sum_error = 0.0
    # num = 0
    # set_temp = set()
    i = 0
    for file_name in list(file_to_num_samples.keys()):
      print(file_name)
      i += 1
      #print(i, file_name)
      gts_for_plot = []
      outs_for_plot = []
      out_path = "/home/poac/lyx/Project/LSTM/err_data"
      with open(os.path.join(out_path, os.path.basename(file_name)), 'w') as f:
        for j in range(file_to_num_samples[file_name]):
          _outputs_test, _seq_test, _gt_test, _path2files = sess.run([outputs_test, seq_test, gt_test, path2files])
          sum_error += np.sum(np.square(_outputs_test - _gt_test))
          gts_for_plot.append(_gt_test[0])
          outs_for_plot.append(_outputs_test[0])
          if j != file_to_num_samples[file_name]-1:
            f.write('0000'+'\t'+str((_gt_test[0]-_outputs_test[0])[0])+'\n')
          else:
            f.write('0000' + '\t' + str((_gt_test[0] - _outputs_test[0])[0]))
      # plt.plot(range(0, len(gts_for_plot)), gts_for_plot, color='green')
      # plt.plot(range(0, len(outs_for_plot)), outs_for_plot, color='blue')
      # str_temp = os.path.basename(file_name).split('.')[0] + ".png"
      # path_temp = os.path.join("/home/poac/lyx/Project/LSTM/picture/Sequence_Length20_20191214/", str_temp)
      # print(path_temp)
      # plt.ylim(0, 50)
      # plt.savefig(path_temp)
      # #plt.show()
      # plt.close()


      """
      # if num == 0:
      #   first_file = _path2files[0].decode()
      # sum_error += np.sum(np.square(_outputs_test - _gt_test))
      # num += 1
      # print(num)
      # last_file = _path2files[0].decode()
      # if osp.basename(last_file) != osp.basename(first_file) and osp.basename(first_file) == osp.basename(_path2files[0].decode()):
      #   break
      # print(_seq_test.tolist())
      if osp.basename(_path2files[0].decode()).split('.')[0] != osp.basename(last_file.decode()).split('.')[0]:
        # data_original = np.reshape(np.loadtxt(last_file.decode(), delimiter="\t", unpack=True)[1], [-1, 1]).tolist()[cfg.INPUT.SEQ_LEN:]  # huatu

        # print(len(data_original))
        # print('########################')
        # print(len(out_per_file))

        # plt.plot(range(0, len(data_original)), data_original, color='green')
        # plt.plot(range(0, len(out_per_file)), out_per_file, color='blue')
        # str_temp = "figure_" + str(count) + ".png"
        # path_temp = os.path.join("/home/poac/lyx/Project/LSTM/picture/Sequence_Length20/", str_temp)
        # plt.savefig(path_temp)
        # plt.show()

        last_file = _path2files[0]
        # job_name = bytes(job_name, encoding="utf-8").decode()
        # out_per_file = []
        if osp.basename(_path2files[0].decode()) in set_temp:
          break
        # out_per_file.append(_outputs_test[0][0])
        """

      # set_temp.add(osp.basename(_path2files[0].decode()))
      # num += 1
      # else:
      #   sum_error += np.sum(np.square(_outputs_test - _gt_test))
      #   set_temp.add(osp.basename(_path2files[0].decode()))
      #   num += 1
      #   # out_per_file.append(_outputs_test[0][0])
    mean_sum_error = sum_error/num_samples
    print(mean_sum_error)
    print("#########################")
    # # print(count)
    # print(num)
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(len(set_temp))


if __name__ == '__main__':
  test()
