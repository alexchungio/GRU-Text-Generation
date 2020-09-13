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
from queue import Queue

from libs.configs import cfgs


def batch_generator(text, char_index, seq_length=100, batch_size=64, buffer_size=2000):

    # map strings to a numerical representation
    text_sequence = np.array([char_index[char] for char in text])

    # char_dataset = tf.data.Dataset.from_tensor_slices(text_sequence)
    #
    # sequences = char_dataset.batch(batch_size=seq_length, drop_remainder=True)

    # use map apply method to each batch
    # dataset = sequences.map(split_input_target)
    #
    # dataset = dataset.shuffle(10000).repeat(epoch).batch(batch_size=batch_size, drop_remainder=True)
    # #
    # return dataset

    seq_size = int(len(text_sequence) / seq_length)

    # drop remainder
    remainder_length = len(text_sequence) % seq_length
    text_sequence = text_sequence[:-remainder_length]

    text_sequence = text_sequence.reshape(seq_size, seq_length)
    # sequence queue(FIFO)
    sequence_buffer = Queue(maxsize=buffer_size)
    buffer_index = 0

    while True:
        # construct batch
        input_data = np.zeros(shape=(batch_size, seq_length - 1), dtype=np.int32)
        target_data = np.zeros(shape=(batch_size, seq_length - 1), dtype=np.int32)
        for i in range(batch_size):
            # produce queue
            if sequence_buffer.empty():
                # full buffer
                buffer_start = buffer_index
                if buffer_index >= seq_size:
                    buffer_index = 0
                    buffer_start = 0
                buffer_end = buffer_index + buffer_size
                if buffer_end > seq_size:
                    buffer_end = seq_size
                seq_index = np.arange(buffer_start, buffer_end)
                np.random.shuffle(seq_index)
                for j in seq_index:
                    sequence_buffer.put(text_sequence[j])
                buffer_index += len(seq_index)

            # consumer queue
            sequence = sequence_buffer.get()

            # split sequence to input and target
            input_sequence, target_sequence = split_input_target(sequence)
            input_data[i] = input_sequence
            target_data[i] = target_sequence

        yield  input_data, target_data


def split_input_target(text_sequence):
    """

    :param text_sequence:
    :param seq_index:
    :return:
    """

    input_sequence = text_sequence[:-1]
    target_sequence = text_sequence[1:]

    return input_sequence, target_sequence




def load_parse_data(path):
    # -----------------------download dataset------------------------------------
    file_path = tf.keras.utils.get_file(path,
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
    text, vocab, char_index, index_char = load_parse_data(cfgs.DATASET_PATH)
    #
    # examples_per_epoch = len(text)
    #
    # dataset = batch_generator(text, char_index=char_index, seq_length=cfgs.SEQUENCE_LENGTH, batch_size=2)
    #
    #
    # # for input_example, target_example in dataset.take(5):
    # #     print("Input data: ", repr(''.join(index_char[input_example.numpy()[0]])))
    # #     print("target data: ", repr(''.join(index_char[target_example.numpy()[0]])))
    #
    #
    # with tf.Session() as sess:
    #     for _ in range(1):
    #
    #         input_batch, target_batch = dataset.take(100).make_one_shot_iterator().get_next()
    #         # print(dataset.get_next().eval())
    #         print(input_batch.shape)
    #         print(target_batch.shape)
    #         input_example = input_batch.eval()
    #         target_example = target_batch.eval()
    #         print(input_example)
    #         print(target_example)
    #         print("Input data: ", repr(''.join(index_char[input_example[0]])))
    #         print("target data: ", repr(''.join(index_char[target_example[0]])))



    train_generate = batch_generator(text, char_index, seq_length=100, batch_size=64)
    for _ in range(200000):

        input_example, target_example = next(train_generate)
        print("Input data: ", repr(''.join(index_char[input_example[0]])))
        print("target data: ", repr(''.join(index_char[target_example[0]])))







