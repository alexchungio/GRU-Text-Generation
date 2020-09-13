#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/9 下午4:41
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import time
import json
import numpy as np
import tensorflow as tf

from libs.configs import cfgs
from libs.nets.model import build_model


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


if __name__ == "__main__":

    # ------------get gpu and cpu list------------------
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    # # print(gpus)
    # # print(cpus)
    #
    # # ------------------set visible of current program-------------------
    # # method 1 Terminal input
    # # $ export CUDA_VISIBLE_DEVICES = 2, 3
    # # method 1
    # # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    # # method 2
    # tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    # # ----------------------set gpu memory allocation-------------------------
    # # method 1: set memory size dynamic growth
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # method 2: set allocate static memory size
    # tf.config.experimental.set_virtual_device_configuration(
    #     device=gpus[0],
    #     logical_devices = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    # )


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

    # map strings to a numerical representation
    text_sequence = np.array([char_index[char] for char in text])

    seq_length = 100
    examples_per_epoch = len(text)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_sequence)

    sequences = char_dataset.batch(seq_length, drop_remainder=True)


    # use map apply method to each batch
    dataset = sequences.map(split_input_target)

    # for input_example, target_example in dataset.take(5):
    #     print("Input data: ", repr(''.join(index_char[input_example.numpy()])))
    #     print("target data: ", repr(''.join(index_char[target_example.numpy()])))

    # data config
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE, drop_remainder=True)

    # for input_example, target_example in dataset.take(5):
    #     print("Input data: ", repr(''.join(index_char[input_example.numpy()[0]])))
    #     print("target data: ", repr(''.join(index_char[target_example.numpy()[0]])))

    # ---------------------------------construct network----------------------
    # length of vocab_size
    vocab_size = len(vocab)
    # embedding dimension
    embedding_dim = 256
    # rnn units
    num_units = 1024

    model = build_model(vocab_size, embedding_dim, num_units, batch_size=BATCH_SIZE)

    # print(model.summary())

    # predict
    # for input_example_batch, target_example_batch in dataset.take(1):
    #     example_batch_predictions = model(input_example_batch)
    #     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    #
    #     sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    #     sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    #     print(sampled_indices)
    #
    #     print("Input: \n", repr("".join(index_char[input_example_batch[0]])))
    #     print()
    #     print("Next Char Predictions: \n", repr("".join(index_char[sampled_indices])))
    #
    #
    #     example_batch_loss = loss(target_example_batch, example_batch_predictions)
    #     print(example_batch_loss.shape)

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    summary_writer = tf.summary.create_file_writer(cfgs.SUMMARY_PATH)

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(input, target):
        with tf.GradientTape() as tape:
            predictions = model(input)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return loss

    # name of checkpoint files
    checkpoint_prefix = os.path.join(cfgs.TRAINED_CKPT, "ckpt_{epoch}")

    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    # EPOCHS= 10
    # history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    for epoch in range(cfgs.NUM_EPOCH):

        start = time.time()
        # resetting the hidden state at the start of every epoch
        # hide state is None at start
        hidden = model.reset_states()

        for num_step, (input, target) in enumerate(dataset):
            loss = train_step(input, target)

            if (num_step+1) % cfgs.SHOW_TRAIN_INFO_INTE == 0:
                print('Epoch {} Step {} Loss {}'.format(epoch+1, num_step, loss))

        with summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # save checkpoint
        if (epoch+1) % cfgs.SAVE_WEIGHTS_INER == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))
        print('Epoch {} train Loss {}'.format(epoch+1, train_loss.result()))
        print ('Time taken for epoch {} sec\n'.format(time.time() - start))

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    model.save_weights(checkpoint_prefix.format(epoch=cfgs.NUM_EPOCH))




