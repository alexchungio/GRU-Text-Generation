#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/10 下午2:15
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import json

from libs.configs import cfgs
from libs.nets.model import GRU


def inference(start_string, temperature=1.0, num_generate=1000):
    """

    :param start_string: Low temperatures results in more predictable text.
                         Higher temperatures results in more surprising text.
                         Experiment to find the best setting.
    :param temperature:
    :param num_generate: Number of characters to generate
    :return:
    """

    with open(cfgs.CHAR_INDEX, 'r') as f:
        char_index = json.loads(f.read())

    index_char = np.array(list(char_index.keys()))

    # embedding dimension

    latest_ckpt = tf.train.latest_checkpoint(cfgs.TRAINED_CKPT)
    model = GRU(vocab_size=len(index_char), embedding_dim=cfgs.EMBEDDING_DIM, num_units=cfgs.NUM_UNITS, batch_size=1)

    # Converting our start string to numbers (vectorizing)
    input_eval = [char_index[s] for s in start_string]
    input_eval = np.expand_dims(input_eval, 0)
    # Empty string to store our results
    text_generated = []

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, latest_ckpt)
        print('Successful load weights from {}'.format(latest_ckpt))
        new_states = sess.run(model.initial_satate)
        # Here batch size == 1
        for i in range(num_generate):

            # feed_dict = model.fill_feed_dict(input_data=input_eval,
            #                                  keep_prob=1.0)\
            feed_dict = {model.input_data: input_eval,
                         model.initial_satate: new_states,
                         model.keep_prob: 1.0}
            predictions, new_states = sess.run([model.predict, model.gru_states], feed_dict=feed_dict)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].eval()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = np.expand_dims([predicted_id],axis=0)

            text_generated.append(index_char[predicted_id])

        return (start_string + ''.join(text_generated))


if __name__ == "__main__":

    text = inference(start_string=u"ROMEO: ", num_generate=100, temperature=0.5)
    print(text)

