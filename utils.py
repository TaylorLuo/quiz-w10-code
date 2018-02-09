#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np
from six.moves import xrange
import json


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


vocabulary_size = 5000
data_index = 0
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # My Code here
    ##################

    dl = list()
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    # print('Most common words (+UNK)', count[:5])
    # print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    # datalist = list()

    batch = np.ndarray(shape=(batch_size, num_steps), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, num_steps), dtype=np.int32)
    for step in xrange(num_steps):
        datadict = {}
        global data_index
        # assert batch_size % num_steps == 0
        # assert num_steps <= 2 * skip_window
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j, step] = buffer[skip_window]
                labels[i * num_skips + j, step] = buffer[context_word]
            if data_index == len(data):
                buffer[:] = data[:span]
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        datadict['train_inputs'] = batch
        datadict['train_labels'] = labels
        dl.append(datadict)

    return dl


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # with open('dictionary.json', 'w') as file_object:
    #     json.dump(dictionary, file_object)
    # with open('reverse_dictionary.json', 'w') as file_object:
    #     json.dump(reversed_dictionary, file_object)

    return data, count, dictionary, reversed_dictionary
