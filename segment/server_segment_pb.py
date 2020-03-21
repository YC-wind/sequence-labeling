#!/usr/bin/python
# coding:utf-8
"""
@author: YC
"""
from flask import Flask, request
from dataHelper import *
import time
import tensorflow as tf


def load_segment_model(path):
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
    viterbi_sequence = sess.graph.get_tensor_by_name("doc_segment/ReverseSequence_1:0")

    ""
    text = '也就是说，如果不 实行 计划 生育 今天 的 中国'
    ws = re.sub("\s", "", text, 99999)
    print(ws)
    max_num = 256
    x = []
    m = 0
    for w in ws:
        # 单词层面
        id = char2id.get(w)
        if id:
            x.append(id)
        else:
            x.append(len(char2id))
    if len(ws) <= max_num:
        x = x + [0] * (max_num - len(ws))
        m = len(ws)
    else:
        x = x[:max_num]
        m = max_num
    # Y = np.array([np.zeros(sent_len)], dtype='int32')
    X = np.array([x], dtype='int32')
    # print(X)
    M = np.array([m], dtype='int32')
    feed_dict = {
        input_x: X,
        input_m: M,
    }
    predicts_d = sess.run([viterbi_sequence], feed_dict)[0]
    p = predicts_d.tolist()[0]
    for word, label in zip(ws, p[:len(ws)]):
        print(word + "\t\t" + id2label.get(label))
    return sess


# 读入数据
char2id, label2id = load_dict()
id2label = {value: key for key, value in label2id.items()}

id2pos_label = id2label
# 加载下 模型
sess = load_segment_model("./model/segment_model.pb")
input_x = sess.graph.get_tensor_by_name("placeholder/input_x:0")
input_m = sess.graph.get_tensor_by_name("placeholder/input_m:0")
# input_y = sess.graph.get_tensor_by_name("placeholder/input_y:0")
viterbi_sequence = sess.graph.get_tensor_by_name("doc_segment/ReverseSequence_1:0")


def segment_batch(sentences):
    """
        输入 为 string 类型 数据
    :param sentence:
    :return:
    """

    max_num = 256

    ""
    # 对于很多文档，需自行分句 ，使用 \n 分隔
    X = []
    M = []
    new_s = []
    for s in sentences:
        if s.strip() == "":
            continue
        ws = re.sub("\s", "", s, 99999)
        print(ws)
        new_s.append(ws)
        x = []
        m = 0
        for w in ws:
            # 单词层面
            id = char2id.get(w)
            if id:
                x.append(id)
            else:
                x.append(len(char2id))
        if len(ws) <= max_num:
            x = x + [0] * (max_num - len(ws))
            m = len(ws)
        else:
            x = x[:max_num]
            m = max_num
        # Y = np.array([np.zeros(sent_len)], dtype='int32')
        X.append(x)
        # print(X)
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
    for sentence, predict in zip(new_s, predicts_d):
        temp_l = len(sentence)
        words = []
        temp_word = ""
        for word, pos in zip(sentence, predict[:temp_l]):
            label = id2pos_label.get(pos)
            temp_word += word
            if label == "E" or label == "S":
                words.append(temp_word)
                temp_word = ""
        result.append({str(index): {"sentence": sentence, "segment": words, "segmentString": " ".join(words)}})
        index += 1
    return result


if __name__ == '__main__':
    sentences = ["15日，备受关注的电影《黄金时代》在北京举行了电影发布会，导演许鞍华和编剧李樯及汤唯、冯绍峰等众星悉数亮相。"]
    result = segment_batch(sentences)
    print(result)
