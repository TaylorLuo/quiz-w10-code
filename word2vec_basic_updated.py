# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from flags import parse_args

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


# filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words."""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data

def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data

# vocabulary = read_data(filename)
# print('Data size', len(vocabulary))

FLAGS, unparsed = parse_args()
vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000

#将words集合中的单词按频数排序，将频率最高的前n_words-1个单词以及他们的出现的个数按顺序输出到count中，
# 将频数排在n_words-1之后的单词设为UNK。同时，count的规律为索引越小，单词出现的频率越高
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  # most_common方法： 去top5000的频数的单词，创建一个dict,放进去。以词频排序
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  # 全部单词转为编号
  # 先判断这个单词是否出现在dictionary，如果是，就转成编号，如果不是，则转为编号0（代表UNK）
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory. 删除原始单词列表，节约内存
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

#这个函数的功能是对数据data中的每个单词，分别与前一个单词和后一个单词生成一个batch，
# 即[data[1],data[0]]和[data[1],data[2]]，其中当前单词data[1]存在batch中，前后单词存在labels中
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0  #batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
  assert num_skips <= 2 * skip_window #可以联系的距离（skip_window）必须满足每个单词生成样本数量（num_skips）的要求，即可以联系的距离（可以往左也可以往右，所以×2）要大于等于要生成样本数量
  batch = np.ndarray(shape=(batch_size), dtype=np.int32) #建一个batch大小的数组，保存任意单词
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) #建一个（batch，1）大小的二位数组，保存任意单词前一个或者后一个单词，从而形成一个pair
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span) #deque还可以设置队列的长度，使用 deque(maxlen=N) 构造函数会新建一个固定大小的队列。当新的元素加入并且这个队列已满的时候， 最老的元素会自动被移除掉
  if data_index + span > len(data): #如果索引超过了数据长度，则重新从数据头部开始
    data_index = 0
  buffer.extend(data[data_index:data_index + span]) #将数据index到index+3段赋值给buffer，大小刚好为span
  data_index += span  #将index向后移3位
  for i in range(batch_size // num_skips):   #128//2 四舍五入       每个批次训练样本数量//每个单词生成样本数量  即要遍历几个单词
    context_words = [w for w in range(span) if w != skip_window]  #[0,2]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data): #  如果到达数据尾部
      # buffer[:] = data[:span] #重新开始，将数据前三位存入buffer中
      # 简单理解就是，不断累加的data_index等于len(data)时，就是滑窗滑到最后3个字母了
      # 这时要把他们匹配成batch和labels,就需要都放到deque中去，巴拉巴拉
      for word in data[:span]: # 取最后3个字母
        buffer.append(word)    # 塞到deque中，将原来的挤出去
      data_index = span        # 如此往复循环，当语料太少时，会重复生成
    else:
      buffer.append(data[data_index]) # 在循环遍历单词时，新加入一个单词，把后面一个单词从deque中挤出去，滑窗后移1位
      data_index += 1 # 下标加1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  # 一下子想不过来，分两下子想：1.如果(batch_size/num_skips)%(len(date)-num_skips)整除
  # 即data_index == len(data)时刚好够batch_size个样本，此时data_index=span=3
  # 2.如果不整除，则data_index=余数+span
  # 但是下次生成batch_size个样本时，要从头开始，然鹅，132行，每次大循环开始都向后移动了3位，所以这里要向前回退3位，否则就不对了
  return batch, labels

# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#   print(batch[i], reverse_dictionary[batch[i]],
#         '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128      #每个批次训练多少样本

embedding_size = 128  # Dimension of the embedding vector. 要生成的词向量维度

skip_window = 1       # How many words to consider left and right.
                      # 单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips

num_skips = 2         # How many times to reuse an input to generate a label.
                      #为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。

num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on. 抽取的验证单词数
valid_window = 100  # Only pick dev samples in the head of the distribution. 验证单词只从频率最高的100个单词中获取
valid_examples = np.random.choice(valid_window, valid_size, replace=False) # 不重复的从0-99里选择16个index


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    # 初始化 embedding vector [5000, 128]
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) # tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 生成均匀分布平均数，范围为[-1.0,1.0]
    # 使用tf.nn.embedding_lookup(embedding, train_inputs)查找输入train_input对应的embed
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    # 用tf.truncated_normal(截断正态分布随机数)初始化nce_weights
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    # nce_biases初始化为0
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  # 学习率为1.0，L2范式标准化后的enormalized_embedding
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # 通过cos方式来测试  两个之间的相似性，与向量的长度没有关系
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

  # 除以其L2范数后得到标准化后的normalized_embeddings
  normalized_embeddings = embeddings / norm
  # 获取16个验证单词的词向量
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
  # 对矩阵a和矩阵b进行乘法，也就是a * b。两个参数输入必须是矩阵形式（张量的行列大于2），符合矩阵乘法的前后矩阵行列形式，
  # 包括转置之后。两个矩阵必须具有相同的数据类型，支持的数据类型：float16, float32, float64, int32, complex64, complex128。
  # 也可以通过参数 transpose_a或transpose_b来设置矩阵在乘法之前进行转置，这时这些标志位应该设置为True，默认是False。
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 200 == 0:
      if step > 0:
        average_loss /= 200
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 1000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
