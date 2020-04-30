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
    ite_train = Reader_joint(cfg, mode='Train').read()
    ite_test, num_file_test, file_to_num_samples, num_samples = Reader_joint(cfg, mode='Test').read_test()
    seq_train, gt_train = ite_train.get_next()
    seq_test, gt_test, path2files = ite_test.get_next()

    '''构建各个模块的输入-train'''
    # LSTM模型的输入
    seq_train_for_prediction = seq_train[:, cfg.INPUT.SEQ_LEN:, :]  # shape = [batch-size, 4+seq_len, gt_len]
    # 计算历史输入，每一个历史输入都相当与一个batch的元素，相当于batch_size扩展了
    seq_train_for_history_prediction = tf.concat([seq_train[:, x:4+x+cfg.INPUT.SEQ_LEN, :] for x in range(0, cfg.INPUT.SEQ_LEN)],
                                                 axis=1)  # shape = [batch_size, seq_len*(4+seq_len), gt_len]
    seq_train_for_history_prediction = tf.reshape(seq_train_for_history_prediction,
                                                  shape=[cfg.TRAIN.BATCH_SIZE*cfg.INPUT.SEQ_LEN,
                                                         (4+cfg.INPUT.SEQ_LEN), cfg.INPUT.GT_LEN])  # shape = [batch_size*seq_len, (4+seq_len), gt_len]
    '''构建各个模块的输入-test'''
    # LSTM模型的输入
    seq_test_for_prediction = seq_test[:, cfg.INPUT.SEQ_LEN:, :]  # shape = [batch-size, 4+seq_len, gt_len]
    # 计算历史输入，每一个历史输入都相当与一个batch的元素，相当于batch_size扩展了
    seq_test_for_history_prediction = tf.concat([seq_test[:, x:4 + x + cfg.INPUT.SEQ_LEN, :] for x in range(0, cfg.INPUT.SEQ_LEN)],
                                                axis=1)  # shape = [batch_size, seq_len*(4+seq_len), gt_len]
    seq_test_for_history_prediction = tf.reshape(seq_test_for_history_prediction,
                                                  shape=[cfg.VALID.BATCH_SIZE * cfg.INPUT.SEQ_LEN,
                                                         (4 + cfg.INPUT.SEQ_LEN),
                                                         cfg.INPUT.GT_LEN])  # shape = [batch_size*seq_len, (4+seq_len), gt_len]


  """Model"""
  with tf.name_scope('model_train'), tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
    '''主模型前传'''
    prediction_outputs_train = models.prediction_inference_with_attention(cfg, seq_train_for_prediction, is_training=True)  # shape=[batch_size, gt_len]
    print(prediction_outputs_train.get_shape().as_list())
    '''计算所有的error'''
    history_prediction = models.prediction_inference_with_attention(cfg, seq_train_for_history_prediction, is_training=True) # shape=[batch_size*seq_len, gt_len]
    history_prediction = tf.reshape(history_prediction, [-1, cfg.INPUT.SEQ_LEN, cfg.INPUT.GT_LEN])
    err = seq_train_for_prediction[:, 4:, :] - history_prediction  # shape=[batch_size, seq_len, gt_len]
    '''error correction'''
    err_correction = models.err_correction_inference(cfg, err, is_training=True)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # # error 量级 concat
    # pin_hou = tf.concat([seq_train_liangjibianhua, err], axis=2)  # [cfg.INPUT.BATCH_SIZE, cfg.INPUT.SEQ_LEN, cfg.INPUT.GT_LEN+5])
    # # 计算attention weights
    # with tf.variable_scope('Attention', reuse=None):
    #   attention_weights = tf.contrib.layers.fully_connected(pin_hou, cfg.INPUT.SEQ_LEN)
    #   attention_weights = tf.sparse_softmax(attention_weights)  # ????????????????????????????
    # # 计算prediction
    # outputs_train = tf.reduce_sum(outputs_train * attention_weights, axis=1)  # shape=[batch_size, hidden_size]
    # outputs_train = tf.contrib.layers.dropout(outputs_train, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)  # fully connected
    # outputs_train = tf.contrib.layers.fully_connected(outputs_train, cfg.INPUT.GT_LEN)


  """loss for training"""
  with tf.name_scope('loss'):
    loss_prediction = tf.losses.mean_squared_error(gt_train, prediction_outputs_train)
    loss_err_correction = tf.losses.mean_squared_error(gt_train-prediction_outputs_train, err_correction)
    loss_mse = 0.5*loss_prediction + 0.5*loss_err_correction

  """训练模型"""
  global_step = tf.get_variable('global_step', [], dtype=None, initializer=tf.constant_initializer(0),
                                trainable=False)
  lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE_BASE, global_step, cfg.TRAIN.DECAY_STEP,
                                  cfg.TRAIN.DECAY_RATE, staircase=True)  # learning rate
  opt = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')  # optimizer

  """variables"""
  vars_model = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='LSTM')

  """optimize"""
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.name_scope('train'):
    with tf.control_dependencies(update_ops):
      train_op = opt.minimize(loss_mse, global_step=global_step, var_list=vars_model, name='train_op')

  """验证前传"""
  with tf.name_scope('model_test'), tf.variable_scope('LSTM', reuse=True):
    '''主模型前传'''
    prediction_outputs_test = models.prediction_inference_with_attention(cfg, seq_test_for_prediction, is_training=False)  # shape=[batch_size, gt_len]
    '''计算所有的error'''
    history_prediction_test = models.prediction_inference_with_attention(cfg, seq_test_for_history_prediction, is_training=False) # shape=[batch_size*seq_len, gt_len]
    history_prediction_test = tf.reshape(history_prediction_test, [-1, cfg.INPUT.SEQ_LEN, cfg.INPUT.GT_LEN])
    err_test = seq_test_for_prediction[:, 4:, :] - history_prediction_test  # shape=[batch_size, seq_len, gt_len]
    '''error correction'''
    err_correction_test = models.err_correction_inference(cfg, err_test, is_training=False)


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

  """add all to summaries"""
  with tf.name_scope('summary'):
    tf.summary.scalar(tensor=loss_prediction, name='train_loss_prediction')
    tf.summary.scalar(tensor=loss_err_correction, name='train_loss_err_correction')
    tf.summary.scalar(tensor=loss_mse, name='train_mse')
    tf.summary.scalar(tensor=lr, name='learning_rate')
    # tf.summary.histogram(tensor=grad, name='gradient')
    summary_op = tf.summary.merge_all()

  # -------------Start a session-------------#
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
  config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    """initialize global variables(or load pretrained models)"""
    tf.global_variables_initializer().run()
    if cfg.INPUT.RESTORE_MODEL:
      # model_file = tf.train.latest_checkpoint('/home/yzy/workspace/SBD-FCN-Improv/models')
      model_file = 'models/SBD_FCN.ckpt-3300'
      saver.restore(sess, model_file)
    # sess.graph.finalize()
    """训练及验证"""
    history_valid_result_mse = [999999990.0]
    history_valid_result_mape = [999999990.0]
    try:
      while True:
        # 读取数据
        start_time = time.time()
        _, _lr, _step, _loss_mse, _loss_prediction, _loss_err_correction, _summaries, _seq_train, _gt_train \
          = sess.run([train_op, lr, global_step, loss_mse, loss_prediction, loss_err_correction,
                      summary_op, seq_train, gt_train])
        # _grad = np.array(_grad).shape  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """打印结果"""
        if _step % cfg.OUTPUT.PRINT_LOG_INTERVAL == 0:
          end_time = time.time()
          avg_time = (end_time - start_time) / float(1)
          print('Step {} | Global Loss: {:.6f}, prediction Loss: {:.6f}, err prediction Loss: {:.6f} lr = {:.6f}, time consuming = {}'
                .format(_step, _loss_mse, _loss_prediction,_loss_err_correction, _lr, avg_time))
          sys.stdout.flush()
        """写入日志"""
        if _step % cfg.OUTPUT.SUM_WRITE_INTERVAL == 0:
          summary_writer.add_summary(_summaries, _step)

        """测试效果"""
        if _step != 0 and _step % cfg.OUTPUT.VALID_SAVE_INTERVAL == 0: #_step != 0 and
          print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
          print('Start validation:')
          accumulate_mse = 0.0
          accumulate_mape = 0.0
          """验证每一个video"""
          i = 0
          for file_name in list(file_to_num_samples.keys()):
            i += 1
            print(i, file_name)
            for j in range(file_to_num_samples[file_name]):
              _prediction_outputs_test, _err_correction_test, _seq_test, _gt_test, _path2files \
                = sess.run([prediction_outputs_test, err_correction_test, seq_test, gt_test, path2files])
              res = _prediction_outputs_test + _err_correction_test
              mse_test = np.sum(np.mean(np.square(res - _gt_test), axis=1))
              mape_test = abs(res[0][0] - _gt_test[0][0])/abs(_gt_test[0][0])
              accumulate_mse += mse_test
              accumulate_mape += mape_test
          mean_mse_test = accumulate_mse / num_samples
          mean_mape_test = accumulate_mape / num_samples
          test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_error_mse', simple_value=float(mean_mse_test))])
          test_summary2 = tf.Summary(value=[tf.Summary.Value(tag='test_error_mape', simple_value=float(mean_mape_test))])
          summary_writer.add_summary(summary=test_summary, global_step=_step)
          summary_writer.add_summary(summary=test_summary2, global_step=_step)

          print('mean test mse is:%.4f' % mean_mse_test)
          print('mean test mape is:%.4f' % mean_mape_test)
          """选择是否保存模型"""
          if mean_mse_test < min(history_valid_result_mse) and mean_mape_test < min(history_valid_result_mape):
            print('saving model...')
            save_path = osp.join(cfg.OUTPUT.BASE_OUTPUTS_PATH, cfg.OUTPUT.MODEL_DIR + '_' + str(trainno),
                                 cfg.OUTPUT.SAVED_MODEL_PATTERN)
            saver.save(sess, save_path, global_step=global_step)
            print('successfully saved !')
            history_valid_result_mse.append(mean_mse_test)
            history_valid_result_mape.append(mean_mape_test)
          else:
            print('Best performance in history is mse: %.4f, mape: %.4f. Do not save model'
                  % (min(history_valid_result_mse), min(history_valid_result_mape)))

          print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
          sess.graph.finalize()
    finally:
      summary_writer.close()


if __name__ == '__main__':
  trainval()
