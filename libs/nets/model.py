#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/10 下午2:06
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf



def build_model(vocab_size, embedding_dim, num_units, batch_size):
    """

    :param vocabs_size:
    :param embedding_size:
    :param num_units:
    :param batch_size:
    :return:
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                        batch_input_shape=[batch_size, None]))

    model.add(tf.keras.layers.GRU(units=num_units, return_sequences=True, stateful=True,
                                  recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(vocab_size))
    model.reset_states()
    return model