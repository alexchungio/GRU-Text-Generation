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
from libs.nets.model import build_model



def inference(temperature=1.0):
    with open(cfgs.CHAR_INDEX, 'r') as f:
        char_index = json.loads(f.read())

    index_char = np.array(list(char_index.keys()))

    # embedding dimension
    embedding_dim = 256
    # rnn units
    num_units = 1024

    model = build_model(vocab_size=65, embedding_dim=embedding_dim, num_units=num_units, batch_size=1)
    latest_checkpoint = tf.train.latest_checkpoint(cfgs.TRAINED_CKPT)

    model.load_weights(latest_checkpoint)

    model.build(tf.TensorShape([1, None]))

    model.summary()

    new_text = generate_text(model, start_string=u"ROMEO: ", char_index=char_index, index_char=index_char,
                             temperature=temperature)

    return new_text

def generate_text(model, start_string, char_index, index_char, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_char[predicted_id])

    return (start_string + ''.join(text_generated))


if __name__ == "__main__":


    text = inference(temperature=0.5)
    print(text)

