# coding:utf-8
import argparse
import pprint
import os
import sys
import time
import tensorflow as tf
import numpy as np
import math
import glob
import random
import os.path as osp

sys.path.append(os.getcwd())
from dataloader_pretrain import Reader as Reader_pretrain
from dataloader_joint import Reader as Reader_joint
import matplotlib.pyplot as plt
import models
from config import cfg, cfg_from_file, get_output_dir  # 从参数中import cfg


def _parse_args():
  parser = argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
  args = parser.parse_args()
  return args


def trainval():
  # -------------parse arguments-------------#
  args = _parse_args()
  trainno = 000
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  # read from args.cfg_file, and integrate into cfg
    trainno = args.cfg_file.split(os.sep)[-1][3:-5]
  pprint.pprint(cfg)  # 漂亮的打印

  """建立该有的文件夹"""
  if not osp.exists(cfg.OUTPUT.BASE_OUTPUTS_PATH):  # 输出总路径
    os.system('mkdir ' + cfg.OUTPUT.BASE_OUTPUTS_PATH)
  summary_dir = osp.join(cfg.OUTPUT.BASE_OUTPUTS_PATH, cfg.OUTPUT.SUMMARY_DIR + '_' + str(trainno))

  # -----------some configurations---------#
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # tensorflow设置使用哪一块GPU
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO)  # Sets the threshold for what messages will be logged.

  # -------------construct a graph-------------#
  """建立dataset，获取iterator"""
  with tf.name_scope('input'), tf.device('/cpu:0'):
    ite_test, num_file_test, file_to_num_samples, num_samples = Reader_joint(cfg, mode='Test').read_test()
    seq_test, gt_test, path2files = ite_test.get_next()

    '''构建各个模块的输入-test'''
    # LSTM模型的输入
    seq_test_for_prediction = seq_test[:, cfg.INPUT.SEQ_LEN:, :]  # shape = [batch-size, 4+seq_len, gt_len]
    # 计算历史输入，每一个历史输入都相当与一个batch的元素，相当于batch_size扩展了
    """
    seq_test_for_history_prediction = tf.concat([seq_test[:, x:4 + x + cfg.INPUT.SEQ_LEN, :] for x in range(0, cfg.INPUT.SEQ_LEN)],
                                                axis=1)  # shape = [batch_size, seq_len*(4+seq_len), gt_len]
    seq_test_for_history_prediction = tf.reshape(seq_test_for_history_prediction,
                                                  shape=[cfg.VALID.BATCH_SIZE * cfg.INPUT.SEQ_LEN,
                                                         (4 + cfg.INPUT.SEQ_LEN),
                                                         cfg.INPUT.GT_LEN])  # shape = [batch_size*seq_len, (4+seq_len), gt_len]
    """

  """验证前传"""
  with tf.name_scope('model_test'), tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
    '''主模型前传'''
    prediction_outputs_test = models.prediction_inference_pure_bilstm(cfg, seq_test_for_prediction, is_training=False)  # shape=[batch_size, gt_len]
    '''计算所有的error'''
    """
    history_prediction_test = models.prediction_inference_with_attention(cfg, seq_test_for_history_prediction, is_training=False) # shape=[batch_size*seq_len, gt_len]
    history_prediction_test = tf.reshape(history_prediction_test, [-1, cfg.INPUT.SEQ_LEN, cfg.INPUT.GT_LEN])
    err_test = seq_test_for_prediction[:, 4:, :] - history_prediction_test  # shape=[batch_size, seq_len, gt_len]
    '''error correction'''
    err_correction_test = models.err_correction_inference(cfg, err_test, is_training=False)
    """

  """saver"""
  model_variables = {}
  for variable in tf.global_variables():
    if variable.name.find('LSTM') >= 0:
      model_variables[variable.name.replace(':0', '')] = variable
  print('#####################################################')
  for save_item in model_variables.keys():
    print(save_item)
  print('#####################################################')
  with tf.name_scope('saver'):
    saver = tf.train.Saver(var_list=model_variables, max_to_keep=cfg.OUTPUT.MAX_MODELS_TO_KEEP)


  # -------------Start a session-------------#
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
  config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    """initialize global variables(or load pretrained models)"""
    tf.global_variables_initializer().run()
    # model_file = '/home/poac/lyx/Project/ECA-LSTM/out_new/models_4/ECA-LSTM.ckpt-90001'  # Bi-LSTM 效果最好的模型
    model_file = '/home/poac/lyx/Project/ECA-LSTM/out_new/models_4/ECA-LSTM.ckpt-1001'  # Bi-LSTM 效果最差的模型
    # model_file = cfg.VALID.RESTORE_FROM
    saver.restore(sess, model_file)
    """测试效果"""
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Start validation:')
    i = 0
    k = 0
    import time
    start_time = time.time()
    errs_for_plot = []
    outs_for_plot = []
    gts_for_plot = []
    out_path = "/home/poac/lyx/Project/ECA-LSTM/plots/prediction_new/Bi-LSTM_cha"
    for file_name in list(file_to_num_samples.keys()):
      i += 1
      print(i, file_name)
      for j in range(file_to_num_samples[file_name]):
        k += 1
        # print(k)
        _outputs_test, _seq_test, _gt_test, _path2files = sess.run(
          [prediction_outputs_test, seq_test, gt_test, path2files])
        gts_for_plot.append(_gt_test[0])
        outs_for_plot.append(_outputs_test[0])
        err = np.mean(np.square(_outputs_test[0] - _gt_test[0]))
        errs_for_plot.append(err)

    errs_for_plot = np.array(errs_for_plot)
    errs_for_plot -= errs_for_plot.min()
    errs_for_plot /= errs_for_plot.max()
    plt.plot(range(0, len(errs_for_plot)), errs_for_plot, linewidth=0.18, color='orange', label='Error Curve')
    import pandas as pd
    data_save = pd.DataFrame(errs_for_plot)
    data_save1 = pd.DataFrame(gts_for_plot)
    data_save2 = pd.DataFrame(outs_for_plot)
    # str_temp1 = os.path.basename(file_name).split('.')[0] + "_%05d" % i + ".txt"

    path = os.path.join(cfg.OUTPUT.BASE_PLOT_PATH, 'error.txt')
    data_save.to_csv(path, index=False, header=None)
    path1 = os.path.join(cfg.OUTPUT.BASE_PLOT_PATH, 'gt.txt')
    data_save1.to_csv(path1, index=False, header=None)
    path2 = os.path.join(cfg.OUTPUT.BASE_PLOT_PATH, 'prediction_BiLSTM.txt')
    data_save2.to_csv(path2, index=False, header=None)
    plt.plot(range(0, len(gts_for_plot)), gts_for_plot, linewidth=0.18, color='red', label='Groundtruth')
    plt.plot(range(0, len(outs_for_plot)), outs_for_plot, linewidth=0.18,  color='blue', label='Prediction')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.title('Visualization of real and predicted temperature.')
    str_temp = os.path.basename(file_name).split('.')[0] + "_%05d" % i + ".png"
    path_temp = os.path.join(out_path, str_temp)
    plt.savefig(path_temp, dpi=2048)
    plt.show()
    plt.close()

    end_time = time.time()
    print('time consuming is:', end_time - start_time)
    print('num samples is:', i)
    print('time consuming per sample is:', (end_time - start_time) / i)


if __name__ == '__main__':
  trainval()
