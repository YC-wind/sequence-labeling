#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-03-21 10:13
"""
import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(model_folder, output_graph="nlp_model.pb"):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    output_node_names = []  # NOTE: Change here

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        opts = sess.graph.get_operations()
        for v in opts:
            print(v.name)
            output_node_names.append(v.name)

        # var_list = tf.global_variables()
        # output_node_names_ = [var_list[i].name for i in range(len(var_list))]

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            # input_graph_def,  # The graph_def is used to retrieve the nodes
            sess.graph_def,
            ["keep_prob", "loss/ReverseSequence_1"]
            # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print("[INFO] output_graph:", output_graph)
        print("[INFO] all done")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Tensorflow graph freezer\nConverts trained models to .pb file",
    #                                  prefix_chars='-')
    # parser.add_argument("--mfolder", type=str, help="model folder to export", default="./ckpt")
    # parser.add_argument("--ograph", type=str, help="output graph name", default="./text_cnn.pb")
    # args = parser.parse_args()
    # print(args, "\n")
    freeze_graph("../model/")
