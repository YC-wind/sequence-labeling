#!/usr/bin/python
# coding:utf-8
"""
@author: YC
@contact: yc175798@gongdao.com
"""
from flask import Flask, request
from dataHelper import *
import logging  # 引入logging模块
import time
import tensorflow as tf


def load_pos_model(path):
    output_graph_def = tf.GraphDef()

    with open(path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    input_x = sess.graph.get_tensor_by_name("placeholder/input_x:0")
    input_m = sess.graph.get_tensor_by_name("placeholder/input_m:0")
    # input_y = sess.graph.get_tensor_by_name("placeholder/input_y:0")
    viterbi_sequence = sess.graph.get_tensor_by_name("doc_classification/ReverseSequence_1:0")

    sent_len = 128
    word_len = 8
    ""
    text = u'宏观 和 处理 流程 就 是 这么 回 事'
    ws = text.split()
    x = []
    m = 0
    for w in ws:
        # 单词层面
        arr = []
        for c in w:
            id = char2id.get(c)
            if id:
                arr.append(id)
            else:
                arr.append(len(char2id))
        # 对 w 进行 padding
        l = len(arr)
        if l <= word_len:
            arr = arr + [0] * (word_len - l)
        else:
            arr = arr[:word_len]
        x.append(arr)
    if len(ws) <= sent_len:
        x = x + [[0] * word_len] * (sent_len - len(ws))
        m = len(ws)
    else:
        x = x[:sent_len]
        m = sent_len

    X = np.array([x], dtype='int32')
    # print(X)
    M = np.array([m], dtype='int32')

    feed_dict = {
        input_x: X,
        input_m: M,
    }
    predicts_d = sess.run([viterbi_sequence], feed_dict)[0]
    p = predicts_d.tolist()[0]
    for word, pos in zip(text.split(), p[:len(ws)]):
        print(word + "\t\t" + id2label.get(pos))
    return sess


char2id, label2id = load_dict()
id2label = {value: key for key, value in label2id.items()}
sess = load_pos_model("./model/pos_model.pb")
input_x = sess.graph.get_tensor_by_name("placeholder/input_x:0")
input_m = sess.graph.get_tensor_by_name("placeholder/input_m:0")
# input_y = sess.graph.get_tensor_by_name("placeholder/input_y:0")
viterbi_sequence = sess.graph.get_tensor_by_name("doc_classification/ReverseSequence_1:0")


def pos_tag_batch(sentences):
    """
        输入 为 string 类型 数据
    :param sentence:
    :return:
    """
    # ip = request.remote_addr

    sent_len = 128
    word_len = 8

    # 对于很多文档，需自行分句 ，使用 \n 分隔
    X = []
    M = []
    for s in sentences:
        if s.strip() == "":
            continue
        ws = s.strip().split()
        x = []
        m = 0
        for w in ws:
            # 单词层面
            arr = []
            for c in w:
                id = char2id.get(c)
                if id:
                    arr.append(id)
                else:
                    arr.append(len(char2id))
            # 对 w 进行 padding
            l = len(arr)
            if l <= word_len:
                arr = arr + [0] * (word_len - l)
            else:
                arr = arr[:word_len]
            x.append(arr)
        if len(ws) <= sent_len:
            x = x + [[0] * word_len] * (sent_len - len(ws))
            m = len(ws)
        else:
            x = x[:sent_len]
            m = sent_len
        X.append(x)
        M.append(m)

    X = np.array(X, dtype='int32')
    M = np.array(M, dtype='int32')

    feed_dict = {
        input_x: X,
        input_m: M,
    }
    predicts_d = sess.run([viterbi_sequence], feed_dict)[0]
    result = []
    index = 0
    for sentence, predict in zip(sentences, predicts_d):
        temp_l = len(sentence.split())
        sent = []
        for word, pos in zip(sentence.split(), predict[:temp_l]):
            label = id2label.get(pos)
            sent.append({"word": word, "pos": label})
        result.append({str(index): sent})
        index += 1
    return result


if __name__ == '__main__':
    # 读入数据
    1
    sentences = ["我 爱 我 的 祖国"]
    result = pos_tag_batch(sentences)
    print(result)
