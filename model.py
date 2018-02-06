#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='self.keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                # 构建一个矩阵，就叫embedding好了，尺寸为[num_words, dim_embedding]，
                # 分别表示词典中单词数目，以及要转化成的向量的维度。一般来说，向量维度越高，能够表现的信息也就越丰富。
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])   #[5000,128]
                tf.summary.histogram('embeddings', embed)

            # 使用tf.nn.embedding_lookup(embedding, input_ids)
            # 使用tf.nn.embedding_lookup(embedding, train_inputs)查找输入train_input对应的embed
            data = tf.nn.embedding_lookup(embed, self.X)  # input



        ######################
        # My Code here start #
        ######################
        outputs_tensor = []
        with tf.variable_scope('rnn'):

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.rnn_layers, state_is_tuple=True)  # 多层lstm cell 堆叠起来
            print(cell.state_size)
            self.state_tensor = cell.zero_state(self.batch_size, tf.float32)  # 参数初始化,rnn_cell.RNNCell.zero_state
            outputs, last_state = tf.nn.dynamic_rnn(cell, data, initial_state=self.state_tensor)
            self.outputs_state_tensor = last_state

        ####################
        # My Code here end #
        ####################

        outputs = tf.reshape(outputs, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ######################
            # My Code here start #
            ######################
            # softmax_w , shape=[hidden_size, vocab_size], 用于将distributed表示的单词转化为one-hot表示
            softmax_w = tf.get_variable(
                "softmax_w", [self.dim_embedding, self.num_words], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.num_words], dtype=tf.float32)
            # [batch*numsteps, vocab_size] 从隐藏语义转化成完全表示
            logits = tf.matmul(outputs, softmax_w) + softmax_b



        ####################
        # My Code here end #
        ####################
        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
