#!/usr/bin/python
# coding:utf-8
"""
@author: YC
@software: PyCharm
"""
import os, sys, re, json
sys.path.append("../")
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import math

# 哈工大词性 标注说明  https://www.ltp-cloud.com/intro/#pos_how
# bosonNlp 的词性说明  http://docs.bosonnlp.com/tag_rule.html
# 计算所汉语词性标记集 http://ictclas.nlpir.org/nlpir/html/readme.htm

"""
    哈工大 词性标注的 粒度很粗   26多
    bosonnlp 词性标注相对较好   70多
    计算所汉语词性标记集 词性标注较完整， 相当细  90多

    terminal 终端需要把 ../ 上层目录 加进去
    python IDE工具 自动创建 项目路径 ， 可以不要 sys.path.append("../") 这条语句
"""


def prepare_data_boson(sent_len=128, word_len=8):
    """
    :return:
    """
    X = []
    Y = []
    M = []
    char2id = OrderedDict()
    char2id["<pad>"] = 0
    index_c = 1
    label2id = OrderedDict()
    label2id["<pad>"] = 0
    index_l = 1
    files = os.listdir("./data/")
    for file in files:
        with open("./data/" + file, 'r') as fr:
            for line in fr:
                if line.encode('utf-8').decode('utf-8-sig').strip() == "":
                    continue
                x = []
                y = []
                m = 0
                sentence = line.encode('utf-8').decode('utf-8-sig').strip("\t\n").split("\t")
                for s in sentence:
                    if s == "":
                        continue
                    w_t = s.split(" ")
                    w = w_t[0]
                    t = w_t[1]
                    # 对 单词 进行 处理
                    w_ids = []
                    for c in w:
                        if char2id.get(c):
                            w_ids.append(char2id[c])
                        else:
                            w_ids.append(index_c)
                            char2id[c] = index_c
                            index_c += 1
                    # 对 w 进行 padding
                    l = len(w_ids)
                    if l <= word_len:
                        w_ids = w_ids + [0] * (word_len - l)
                    else:
                        w_ids = w_ids[:word_len]
                    x.append(w_ids)
                    # 对 label 进行 处理
                    l_ = label2id.get(t)
                    if l_:
                        y.append(l_)
                    else:
                        y.append(index_l)
                        label2id[t] = index_l
                        index_l += 1

                # 进行 padding 操作
                l = len(sentence)
                if l <= sent_len:
                    x = x + [[0] * word_len] * (sent_len - l)
                    y = y + [0] * (sent_len - l)
                    m = l
                else:
                    x = x[:sent_len]
                    y = y[:sent_len]
                    m = sent_len
                X.append(x)
                Y.append(y)
                M.append(m)
            print("处理完 3000 文档，文件名称："+"./data/" + file)

    print('读取数据完成')
    # 保存char2id 更好的可视化
    with open('./TrainData/char2id_boson.txt', 'w') as fw:
        for k, v in char2id.items():
            fw.write(k + '\t' + str(v) + '\n')
    # 保存为json格式的数据，方便读取，存储
    open('./TrainData/char2id_boson.json', 'w').write(json.dumps(char2id))

    # 保存label2id 更好的可视化
    with open('./TrainData/label2id_boson.txt', 'w') as fw:
        for k, v in label2id.items():
            fw.write(k + '\t' + str(v) + '\n')
    # 保存为json格式的数据，方便读取，存储
    open('./TrainData/label2id_boson.json', 'w').write(json.dumps(label2id))

    # 保存格式化后的训练文件
    df = pd.DataFrame()
    df['chars'] = X
    df['labels'] = Y
    df['masks'] = M
    df.to_csv('./TrainData/XYM_train_pos_boson.csv')
    # 使用 numpy 进行存储
    X = np.array(X, dtype="int32")
    Y = np.array(Y, dtype="int32")
    M = np.array(M, dtype="int32")
    np.savez("./TrainData/XYM_train_pos_boson.npz", X=X, Y=Y, M=M)
    print('保存输入输出文件为csv')


def load_dict():
    """
        加载char、label字典
    :return:
    """
    # Print args
    print('-' * 50 + '\t准备数据\t' + '-' * 50)
    char2id = json.loads(open('./model/char2id_boson.json').read())
    # 新增 <unk>
    char_value_dict_len = len(char2id) + 1

    label2id = json.loads(open('./model/label2id_boson.json').read())

    print('字集合大小:%d', char_value_dict_len)
    print('标签个数:%d', len(label2id))
    return char2id, label2id


def load_data():
    """
        加载训练数据
    :return:
    """
    XYM = np.load("./TrainData/XYM_train_pos_boson.npz", mmap_mode="r")
    X = XYM["X"]
    Y = XYM["Y"]
    M = XYM["M"]
    X_train, X_val, M_train, M_val, Y_train, Y_val = train_test_split(X, M, Y, random_state=10, test_size=0.1)
    return X_train, X_val, M_train, M_val, Y_train, Y_val


def batch_iter(data, batch_size, num_epoches, reset=True):
    '''
        生成批量的数据
        该 环境 只在 python 27 下 有效
    :return:
    '''
    data_size = len(data)
    data = pd.Series(data)
    num_batches_per_epoches = int(math.ceil(data_size / batch_size))
    for epoch in range(num_epoches):
        if reset:
            shuffle_data = shuffle(data).reset_index(drop=True)
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoches):
            s_index = batch_num * batch_size
            e_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[s_index:e_index]


if __name__ == "__main__":
    prepare_data_boson()
