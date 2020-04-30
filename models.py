# coding:utf-8
import tensorflow as tf
import numpy as np



def prediction_inference_with_attention(cfg, inputs, is_training):
  """
    Args: inputs: shape = [None, 4+seq_len, gt_len]
  """
  '''构建量级变化: 在inputs基础上计算每个时刻的量级变化序列  '''
  inputs_liangjibianhua = tf.concat([inputs[:, x - 4:x + 1, :][:, 1:, :] - inputs[:, x - 4:x + 1, :][:, 0:-1, :]
                                                                        for x in range(4, 4+cfg.INPUT.SEQ_LEN)],
                                        axis=1)  # [batch_size, seq_len*(5-1),1]
  inputs_liangjibianhua = tf.reshape(inputs_liangjibianhua,
                                     shape=[-1, cfg.INPUT.SEQ_LEN, 5 - 1])  # [batch_size, seq_len, 5-1]
  # print(inputs_liangjibianhua.get_shape().as_list())



  with tf.variable_scope('LSTM_Prediction'):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE)
    initial_state = lstm_cell.zero_state(cfg.TRAIN.BATCH_SIZE, tf.float32)

    # Dropout
    # if is_training:
    #   lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB,
    #                                        output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)

    outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, inputs[:, 4:, :], dtype=tf.float32)  # outputs shape=[batch_size, seq_len, hidden_size]
    # print(outputs.get_shape().as_list())
    '''计算attention权重'''
    inp_for_attention_cal = tf.concat([outputs, inputs_liangjibianhua], axis=2)
    attention_weights = cal_attention_weights(cfg, inp_for_attention_cal, is_training=True)  # shape [batch_size, seq_len]
    # print(attention_weights.get_shape().as_list())
    '''计算加权后的特征向量'''
    weighted_feature = tf.reshape(attention_weights, [-1, cfg.INPUT.SEQ_LEN, 1]) * outputs  # shape [batch_size, seq_len, hidden_size]
    weighted_feature = tf.reduce_sum(weighted_feature, axis=1)
    '''计算主模型的最终输出'''
    # if is_training:
    #   weighted_feature = tf.contrib.layers.dropout(weighted_feature, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
    prediction_outputs = tf.contrib.layers.fully_connected(weighted_feature, cfg.INPUT.GT_LEN)  # 输出在这里(＠ω＠)

  return prediction_outputs  # shape=[batch_size, gt_len]


def err_correction_inference(cfg, inputs_for_err_correction, is_training):
  with tf.variable_scope('LSTM_Error_Correction'):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE)
    initial_state = lstm_cell.zero_state(cfg.TRAIN.BATCH_SIZE, tf.float32)

    # Dropout
    # if is_training:
    #   lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB,
    #                                        output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)

    outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, inputs_for_err_correction, dtype=tf.float32)

    # 接全连接层，输出参数
    with tf.name_scope('output'):
      # if is_training:
      #   outputs = tf.contrib.layers.dropout(outputs, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
      logits = tf.contrib.layers.fully_connected(outputs, cfg.INPUT.GT_LEN)

  return logits[:, -1, :]

def cal_attention_weights(cfg, inputs, is_training):
  '''inputs shape [batch_size, seq_len, hidden_size+4]'''
  with tf.variable_scope('attention'):
    # if is_training:
    #   inputs = tf.contrib.layers.dropout(inputs, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
    # lll = tf.reshape(inputs, [cfg.TRAIN.BATCH_SIZE*cfg.INPUT.SEQ_LEN, cfg.NETWORK.HIDDEN_SIZE+4])
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(lll.get_shape().as_list())
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    attention_weights = tf.contrib.layers.fully_connected(tf.reshape(inputs, [-1, cfg.NETWORK.HIDDEN_SIZE+4]),
                                                          1)  # shape [batch_size*seq_len, 1]
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(attention_weights.get_shape().as_list())
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    attention_weights = tf.reshape(attention_weights, [-1, cfg.INPUT.SEQ_LEN])  # shape [batch_size, seq_len] 每一个元素为seq_len个attention权重
    attention_weights = tf.nn.softmax(attention_weights)
  return attention_weights



def prediction_inference_pure_lstm(cfg, inputs, is_training=True):
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE)
  initial_state = lstm_cell.zero_state(cfg.TRAIN.BATCH_SIZE, tf.float32)

  # Dropout
  # if is_training:
  #   lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB, output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)

  outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
  # 接全连接层，输出参数
  with tf.name_scope('output'):
    # if is_training:
    #   outputs = tf.contrib.layers.dropout(outputs, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
    # 将输出映射成GT_LEN纬度的数据
    logits = tf.contrib.layers.fully_connected(outputs, cfg.INPUT.GT_LEN)

  return logits[:, -1, :]


def prediction_inference_pure_bilstm(cfg, inputs, is_training=True):
  lstm_forward = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE)
  lstm_backward = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE)

  # Dropout
  # if is_training:
  #   lstm_forward = tf.contrib.rnn.DropoutWrapper(cell=lstm_forward, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB, output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
  #   lstm_backward = tf.contrib.rnn.DropoutWrapper(cell=lstm_backward, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB, output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)

  # bidirectional rnn   outputs shape = ([batch_size,lstm_len,hidden_size], bw)
  # outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward, lstm_backward, inputs, dtype = tf.float32)
  outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_forward, lstm_backward, inputs,
                                                           # initial_state_fw = lstm_forward.zero_state(self.cfg.TRAIN.BATCH_SIZE, tf.float32),
                                                           # initial_state_bw = lstm_backward.zero_state(self.cfg.TRAIN.BATCH_SIZE, tf.float32),
                                                           dtype=tf.float32)
  outputs_concate = tf.concat(outputs, 2)[:, -1, :]

  # 接全连接层，输出参数
  with tf.name_scope('output'):
    # if is_training:
    #   outputs_concate = tf.contrib.layers.dropout(outputs_concate, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
    logits = tf.contrib.layers.fully_connected(outputs_concate, cfg.INPUT.GT_LEN)

  # output
  return logits


def prediction_inference_multi_layer_bilstm(cfg, inputs, is_training=True):
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.NETWORK.HIDDEN_SIZE, state_is_tuple=True)
  initial_state = lstm_cell.zero_state(cfg.TRAIN.BATCH_SIZE, tf.float32)

  lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
  print('2222222222222222222222222222222222222222')
  print('2222222222222222222222222222222222222222')
  print('2222222222222222222222222222222222222222')
  print('2222222222222222222222222222222222222222')
  print('2222222222222222222222222222222222222222')

  # Dropout
  # if is_training:
  #   lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB, output_keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)

  #outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
  outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, tf.expand_dims(inputs, -1), dtype=tf.float32)
  print('1111111111111111111111111111111111111111111111111')
  print('1111111111111111111111111111111111111111111111111')
  print('1111111111111111111111111111111111111111111111111')
  print('1111111111111111111111111111111111111111111111111')
  print('1111111111111111111111111111111111111111111111111')
  print('CCCCCCCCCCCCCCCCCCCC')
  print(outputs.get_shape().as_list())
  print('CCCCCCCCCCCCCCCCCCCC')
  #outputs = outputs[:, -1, :]
  # 接全连接层，输出参数
  with tf.name_scope('output'):
    # if is_training:
    #   outputs = tf.contrib.layers.dropout(outputs, keep_prob=cfg.NETWORK.DROPOUT_KEEP_PROB)
    logits = tf.contrib.layers.fully_connected(outputs, cfg.INPUT.GT_LEN)

  return logits[:, -1, :]


if __name__ == '__main__':
  def test_reader():
    import sys
    import os
    sys.path.append(os.getcwd())
    from config import cfg
    inp = tf.constant(1, dtype=tf.float32, shape=[2, cfg.INPUT.SEQ_LEN, 1])
    out = prediction_inference_pure_bilstm(cfg, inp, True)
    print(out.get_shape().as_list())
    print(out)
  test_reader()
