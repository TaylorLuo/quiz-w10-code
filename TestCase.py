#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))


# with open(FLAGS.dictionary, encoding='utf-8') as inf:
#     dictionary = json.load(inf, encoding='utf-8')
#
# with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
#     reverse_dictionary = json.load(inf, encoding='utf-8')


model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build()

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
train_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])

datalist = utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)

for i in range(FLAGS.num_steps):
    datadict = datalist[i]
    batch = datadict['train_inputs']
    labels = datadict['train_labels']
    reverse_dictionary = datadict['reverse_dictionary']
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



