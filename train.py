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
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from libs.configs import cfgs
from libs.nets.model import GRU
from data.dataset_pipeline import load_parse_data, dataset_batch


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

    # load dataset
    text, vocab, char_index, index_char = load_parse_data(cfgs.DATASET_PATH)

    examples_per_epoch = len(text)

    dataset = dataset_batch(text, char_index=char_index, seq_length=cfgs.SEQUENCE_LENGTH, batch_size=cfgs.BATCH_SIZE)

    # ---------------------------------construct network----------------------
    model = GRU(vocab_size=len(vocab), embedding_size=cfgs.EMBEDDING_DIM, num_units=cfgs.NUM_UNITS)
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

    saver = tf.train.Saver(max_to_keep=30)

    # get computer graph
    graph = tf.get_default_graph()

    write = tf.summary.FileWriter(logdir=cfgs.SUMMARY_PATH, graph=graph)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.op.name, var.shape)

        # ------------------load embedding pretrained weights---------------------
        # parse glove pretrained model
        # -----------------------train part------------------------------------------------
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()

        train_step_per_epoch = examples_per_epoch // cfgs.BATCH_SIZE

        # generate batch
        train_data_batch, train_label_batch = dataset.get_next()

        # use k folder validation
        for epoch in range(cfgs.NUM_EPOCH):
            train_bar = tqdm(range(1, train_step_per_epoch + 1))
            train_loss = []
            for step in train_bar:
                x_train, y_train = sess.run([train_data_batch, train_label_batch])
                feed_dict = model.fill_feed_dict(x_train, y_train, keep_prob=cfgs.KEEP_PROB)
                summary, global_step, loss, _ = sess.run(
                    [summary_op, model.global_step, model.loss, model.train],
                    feed_dict=feed_dict)
                train_loss.append(loss)
                if step % cfgs.SMRY_ITER == 0:
                    write.add_summary(summary=summary, global_step=global_step)
                    write.flush()

                train_bar.set_description("Epoch {0} : Step {1} => Train Loss: {2:.4f} ".
                                          format(epoch + 1, step, train_loss))

            # save model
            ckpt_file = os.path.join(cfgs.TRAINED_CKPT, 'model_loss={0:4f}.ckpt'.format(sum(train_loss) / len(train_loss)))
            if epoch % cfgs.SAVE_WEIGHTS_ITER == 0:
                saver.save(sess=sess, save_path=ckpt_file, global_step=global_step)
        saver.save(sess=sess, save_path=ckpt_file, global_step=global_step)
    sess.close()
    print('model training has complete')





