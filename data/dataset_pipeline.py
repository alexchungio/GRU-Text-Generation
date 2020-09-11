#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/11 下午1:43
# @ Software   : PyCharm
#-------------------------------------------------------


import json
import numpy as np
import tensorflow as tf

from libs.configs import cfgs


def dataset_batch(text, char_index, seq_length=100, batch_size=32, epoch=None):

    # map strings to a numerical representation
    text_sequence = np.array([char_index[char] for char in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_sequence)

    sequences = char_dataset.batch(seq_length, drop_remainder=True)

    # use map apply method to each batch
    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(buffer_size=10000).repeat(epoch).batch(batch_size=batch_size, drop_remainder=True)

    return dataset.make_one_shot_iterator()


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def load_sparse_data():
    # -----------------------download dataset------------------------------------
    file_path = tf.keras.utils.get_file(cfgs.DATASET_PATH,
                                        origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # ----------------------read and decode data------------------------------------
    with open(file_path, 'rb') as f:
        text = f.read().decode(encoding='utf-8')

    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    # get and save char index
    char_index = {char: index for index, char in enumerate(vocab)}
    index_char = np.array(vocab)

    with open(cfgs.CHAR_INDEX, 'w') as f:
        f.write(json.dumps(char_index))

    return text, vocab, char_index, index_char


if __name__ == "__main__":
    text, vocab, char_index, index_char = load_sparse_data()

    examples_per_epoch = len(text)

    dataset = dataset_batch(text, char_index=char_index, seq_length=cfgs.SEQUENCE_LENGTH, batch_size=cfgs.BATCH_SIZE)


    # for input_example, target_example in dataset.take(5):
    #     print("Input data: ", repr(''.join(index_char[input_example.numpy()[0]])))
    #     print("target data: ", repr(''.join(index_char[target_example.numpy()[0]])))

    input_example, target_example = dataset.get_next()
    with tf.Session() as sess:
        input_example = input_example.eval()
        target_example = target_example.eval()
        print("Input data: ", repr(''.join(index_char[input_example[0]])))
        print("target data: ", repr(''.join(index_char[target_example[0]])))


