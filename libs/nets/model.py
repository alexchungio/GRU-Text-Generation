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

import tensorflow.compat.v1 as tf
from libs.configs import cfgs


class GRU(object):
    def __init__(self, vocab_size, embedding_dim, num_units, max_grad_norm=5, batch_size=64):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_units = num_units
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        # assert num_layers == len(num_units), "the number of units must equal to number layers"
        self.global_step = tf.train.get_or_create_global_step()
        self.build_inputs()
        self.build_gru()
        self.loss = self.losses()
        # self.acc = self.accuracy()
        self.train = self.training()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name="input_data")
            self.input_target = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name="input_target")
            self.keep_prob = tf.placeholder(shape=(), dtype=tf.float32, name="keep_prob")


    def build_gru(self):

        # embedding layer
        self.encode_outputs = tf.nn.embedding_lookup(tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_dim], -1, 1),
                                                                 name="embedding"),
                                                       ids=self.input_data)
        # multi lstm cell
        cells = [self.get_gru_cell(num_units=units, keep_prob=self.keep_prob) for units in self.num_units]

        # multi layer gru
        gru_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.initial_satate = gru_cells.zero_state(self.batch_size, dtype=tf.float32)

        self.gru_outputs, self.gru_states = tf.nn.dynamic_rnn(cell=gru_cells, inputs=self.encode_outputs, dtype=tf.float32,
                                                             scope="gru")

        self.logits = self.dense(inputs=self.gru_outputs, output_size=self.vocab_size)
        self.predict = tf.nn.softmax(self.logits, axis=-1, name="predict")


    def fill_feed_dict(self, input_data, input_target=None, state=None, keep_prob=1.0):

        feed_dict = {
            self.input_data: input_data,
            self.input_target: input_target,
            self.initial_satate: state,
            self.keep_prob: keep_prob
        }
        return feed_dict

    # def accuracy(self):
    #
    #     predict = tf.cast(tf.greater(self.predict, 0.5), tf.int32)
    #     acc_mask = tf.equal(predict, self.input_target)
    #     acc = tf.reduce_mean(tf.cast(acc_mask, dtype=tf.float32))
    #     tf.summary.scalar("acc", acc)
    #
    #     return acc

    def losses(self):
        with tf.variable_scope("loss"):
            target = tf.one_hot(self.input_target, depth=self.vocab_size)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=target,
                                                                    name='entropy')
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.input_target,
                                                                           name='entropy')
            loss = tf.reduce_mean(input_tensor=cross_entropy, name='entropy_mean')
            tf.summary.scalar("loss", loss)
            return loss


    def build_optimizer(self):

        pass

    def training(self):

        global_step_update = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_update]):
            # use clipping gradient to forbid gradient explosion
            train_variable = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_variable), clip_norm=self.max_grad_norm)
            train_op = tf.train.AdamOptimizer(learning_rate=cfgs.LEARNING_RATE)

            return train_op.apply_gradients(zip(grads, train_variable))

    def get_gru_cell(self, num_units=128, keep_prob=1.0, activation='tanh'):
        """

        :param num_units:
        :param keep_prob:
        :param activation:
        :return:
        """
        gru = tf.nn.rnn_cell.GRUCell(num_units=num_units, activation=activation)
        drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=keep_prob)
        return drop


    def dense(self, inputs, output_size, scope="dense", use_bias=True, activation=None):

        inputs = tf.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        last_dim = shape[-1]
        rank = len(shape)

        # initial kernel
        kernel = tf.get_variable(shape=(last_dim, output_size), initializer=tf.orthogonal_initializer(), name='W')

        bias = tf.get_variable(shape=(output_size, ), initializer=tf.zeros_initializer(), name='b')

        with tf.variable_scope(name_or_scope=scope):
            if rank > 2:
                # Broadcasting is required for the inputs.
                outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            else:
                # Cast the inputs to self.dtype, which is the variable dtype. We do not
                # cast if `should_cast_variables` is True, as in that case the variable
                # will be automatically casted to inputs.dtype.
                outputs = tf.matmul(inputs, kernel)
            if use_bias:
                outputs = tf.nn.bias_add(outputs, bias)
            if activation is not None:
                return activation(outputs) # pylint: disable=not-callable
            return outputs