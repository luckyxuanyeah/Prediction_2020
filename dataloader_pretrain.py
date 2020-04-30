import tensorflow as tf
import numpy as np
import scipy.io as scio
from collections import OrderedDict
import os
import os.path as osp
import glob
import json


class Reader(object):
  def __init__(self, cfg, mode):
    """
    :param cfg:
    :param mode: 'Train' or 'Test'
    """
    self.cfg = cfg
    self.mode = mode
    self.files_train = []
    self.files_test = []
    self.file_to_num_samples = {}  # {path2file1:num_file1_sample, path2file2:num_file2_sample, ...}
    self.num_samples = 0
    self._setup(self.cfg, self.mode)

  def _setup(self, cfg, mode):
    files = glob.glob(osp.join(cfg.INPUT.BASE_INPUT_PATH, '*.txt'))
    print(osp.join(cfg.INPUT.BASE_INPUT_PATH))
    num_files = len(files)
    self.files_train = files[0: int(num_files*0.8)+1]
    self.files_test = files[int(num_files*0.8)+1:]
    # self.files_train = files[int(num_files*0.8)+1:]
    # self.files_test = files[0: int(num_files * 0.8) + 1]
    for path2file in self.files_test:
      data = []
      with open(path2file, 'r') as f:
        datas = f.readlines()
        for item in datas:
          data.append(item.split('\t')[1])
        #f.close()
        # data = np.reshape(np.loadtxt(path2file, delimiter="\t", unpack=True)[1], [-1, 1]).tolist()
      data = np.reshape(np.array(data), [-1,1]).tolist()
      data_len = len(data)
      num_file_samples = (data_len-(cfg.INPUT.SEQ_LEN+cfg.INPUT.GT_LEN)+1)//cfg.INPUT.STEP_TEST + 1
      if num_file_samples < 0:
        continue
      self.file_to_num_samples[path2file] = num_file_samples

    for _, value in self.file_to_num_samples.items():
      # print(value)
      self.num_samples += value
      # print(self.num_samples)


  def read(self):
    cfg = self.cfg

    def generator_train():
      files_train = self.files_train
      while True:
        for path2file in files_train:
          data = []
          with open(path2file, 'r') as f:
            datas = f.readlines()
            for item in datas:
              data.append(item.split('\t')[1])
          data = np.reshape(np.array(data), [-1,1]).tolist()
          data_len = len(data)
          begin = 0
          while True:
            if begin + cfg.INPUT.SEQ_LEN - 1 > data_len - 1 - cfg.INPUT.GT_LEN:
              break
            else:
              seq = data[begin:begin+cfg.INPUT.SEQ_LEN]
              gt = np.reshape(data[begin+cfg.INPUT.SEQ_LEN:begin+cfg.INPUT.SEQ_LEN+cfg.INPUT.GT_LEN], [-1]).tolist()
              yield np.array(seq), np.array(gt)
              begin += cfg.INPUT.STEP_TRAIN

    def generator_test():
      files_test = self.files_test
      while True:
        for path2file in files_test:
          data = []
          with open(path2file, 'r') as f:
            datas = f.readlines()
            for item in datas:
              data.append(item.split('\t')[1])
          data = np.reshape(np.array(data), [-1,1]).tolist()
          data_len = len(data)
          begin = 0
          while True:
            if begin + cfg.INPUT.SEQ_LEN - 1 > data_len - 1 - cfg.INPUT.GT_LEN:
              break
            else:
              seq = data[begin:begin + cfg.INPUT.SEQ_LEN]
              gt = np.reshape(data[begin + cfg.INPUT.SEQ_LEN:begin + cfg.INPUT.SEQ_LEN + cfg.INPUT.GT_LEN], [-1]).tolist()
              yield np.array(seq), np.array(gt), path2file
              begin += cfg.INPUT.STEP_TEST

    # video clip paths
    if self.mode == 'Train':
      dataset_train = tf.data.Dataset.from_generator(generator=generator_train,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=([cfg.INPUT.SEQ_LEN, 1], [cfg.INPUT.GT_LEN]))
      dataset_train = dataset_train.prefetch(buffer_size=10)
      dataset_train = dataset_train.shuffle(buffer_size=1000).batch(cfg.TRAIN.BATCH_SIZE)
      ite_train = dataset_train.make_one_shot_iterator()
      return ite_train
    else:
      dataset_test = tf.data.Dataset.from_generator(generator=generator_test,
                                                    output_types=(tf.float32, tf.float32),
                                                    output_shapes=([cfg.INPUT.SEQ_LEN, 1], [cfg.INPUT.GT_LEN]))
      dataset_test = dataset_test.prefetch(buffer_size=10)
      dataset_test = dataset_test.batch(cfg.VALID.BATCH_SIZE)
      ite_test = dataset_test.make_one_shot_iterator()
      return ite_test, len(self.file_to_num_samples.keys())

  def read_test(self):
    cfg = self.cfg

    def _generator():
      files_test = self.files_test
      while True:
        for i in range(len(files_test)):
          # if i == 0:
          #   first_file = files_test[i]
          path2file = files_test[i]
          # data = np.reshape(np.loadtxt(path2file, delimiter="\t", unpack=True)[1], [-1, 1]).tolist()
          data = []
          with open(path2file, 'r') as f:
            datas = f.readlines()
            for item in datas:
              data.append(item.split('\t')[1])
          data = np.reshape(np.array(data), [-1, 1]).tolist()
          data_len = len(data)
          begin = 0
          while True:
            if begin + cfg.INPUT.SEQ_LEN - 1 > data_len - 1 - cfg.INPUT.GT_LEN:
              break
            else:
              seq = data[begin:begin + cfg.INPUT.SEQ_LEN]
              gt = np.reshape(data[begin + cfg.INPUT.SEQ_LEN:begin + cfg.INPUT.SEQ_LEN + cfg.INPUT.GT_LEN],
                              [-1]).tolist()
              yield np.array(seq), np.array(gt), path2file
              begin += cfg.INPUT.STEP_TEST

    dataset_test = tf.data.Dataset.from_generator(generator=_generator,
                                                  output_types=(tf.float32, tf.float32, tf.string),
                                                  output_shapes=([cfg.INPUT.SEQ_LEN, 1], [cfg.INPUT.GT_LEN], []))
    dataset_test = dataset_test.batch(cfg.TEST.BATCH_SIZE)
    dataset_test = dataset_test.prefetch(buffer_size=cfg.TEST.BATCH_SIZE)
    ite_test = dataset_test.make_one_shot_iterator()  # 构建一个迭代器
    return ite_test, len(self.file_to_num_samples.keys()), self.file_to_num_samples, self.num_samples


if __name__ == '__main__':

  def test_reader():
    import sys
    import time
    sys.path.append(os.getcwd())
    from config import cfg
    reader = Reader(cfg, 'Train')
    # reader = ReaderPredAE(cfg, 'Test')
    # ite, num_files, file_to_num_samples, num_samples = reader.read()
    ite = reader.read()
    # print(num_files)
    # print(file_to_num_samples)
    # print(num_samples)
    seq, gt = ite.get_next()
    seq1 = seq
    seq2 = seq[:, int(1/2*cfg.INPUT.SEQ_LEN):, :]
    seq3 = seq[:, int(3/4*cfg.INPUT.SEQ_LEN):, :]
    count = 0
    with tf.Session() as sess:
      while True:
        #try:
        count += 1
        #print(count)
        seq1_,seq2_,seq3_, gt_ = sess.run([seq1, seq2, seq3, gt])
        # print(seq_.shape)
        print(seq1_.shape)
        print(seq2_.shape)
        print(seq3_.shape)
        print('###################')
        # print(gt_)
        # print(_path2file)
        #print(count)
        #except tf.errors.OutOfRangeError:
          # print("!!!!!!!!!!!!!!!")
          # time.sleep(5)


  test_reader()
