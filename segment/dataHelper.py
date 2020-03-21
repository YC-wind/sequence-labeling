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

"""
    使用的数据集， 网上公开的数据集
    + 人民日报数据集，--- 使用boson nlp 进行 分词与词性标注
"""


def prepare_data_segment(max_len=256):
    """
        人民日报 数据集 + CTB 数据集
        使用 bson nlp 开放接口 进行 词性标注
        共 501000 行句子

        预处理数据, 使用 字 层面的 embedding 进行训练
        目前 设置 最多 256 个 字 组成

    :return:
    """
    punc = ["，", "。", "；", "？", "！", "?"]

    X = []
    Y = []
    M = []
    char2id = OrderedDict()
    char2id["<pad>"] = 0
    index_c = 1

    label2id = {
        "<pad>": 0,
        "S": 1,
        "B": 2,
        "M": 3,
        "E": 4,
    }

    # 内部函数

    def do_words(words_):
        """
            对 分隔好的单词 进行 处理
        :param words_:
        :return:
        """
        tag_list = []
        for word in words_:
            if len(word) > 2:
                tag_list.append((word[0], 'B'))
                for i in word[1:-1]:
                    tag_list.append((i, 'M'))
                tag_list.append((word[-1], 'E'))
            elif len(word) == 2:
                tag_list.append((word[0], 'B'))
                tag_list.append((word[1], 'E'))
            else:
                tag_list.append((word, 'S'))
        return tag_list

    files = os.listdir("./data/")
    for file in files:
        with open("./data/" + file, 'r') as fr:
            if re.search("boson_", file):
                for line in fr:
                    # 这里是一行
                    line = line.encode('utf-8').decode('utf-8-sig')
                    if line.strip() == "":
                        continue
                    words = [_.split(" ")[0] for _ in line.strip().split("\t")]
                    # do
                    w_ts = do_words(words)
                    if len(w_ts) > max_len:
                        # 进行截取，使用
                        index_j = max_len - 1
                        for i in range(max_len - 1, 0, -1):
                            if w_ts[i][0] in punc:
                                index_j = i
                                break
                        w_ts = w_ts[:index_j + 1]
                        # 优化 #TODO 充分利用数据集，可以使用后面 截断的数据，用作训练
                    #
                    x = []
                    y = []
                    m = len(w_ts)
                    for w, t in w_ts:
                        temp_w = char2id.get(w)
                        if temp_w:
                            x.append(temp_w)
                        else:
                            x.append(index_c)
                            char2id[w] = index_c
                            index_c += 1
                        y.append(label2id.get(t))
                    x = x + [0] * (max_len - m)
                    y = y + [0] * (max_len - m)
                    X.append(x)
                    Y.append(y)
                    M.append(m)
            else:
                for line in fr:
                    line = line.encode('utf-8').decode('utf-8-sig')
                    if line.strip() == "":
                        continue
                    words = [_ for _ in line.strip().split()]
                    # do
                    w_ts = do_words(words)
                    if len(w_ts) > max_len:
                        # 进行截取，使用
                        index_j = max_len - 1
                        for i in range(max_len - 1, 0, -1):
                            if w_ts[i][0] in punc:
                                index_j = i
                                break
                        w_ts = w_ts[:index_j + 1]
                        # 优化 #TODO 充分利用数据集，可以使用后面 截断的数据，用作训练
                    #
                    x = []
                    y = []
                    m = len(w_ts)
                    for w, t in w_ts:
                        temp_w = char2id.get(w)
                        if temp_w:
                            x.append(temp_w)
                        else:
                            x.append(index_c)
                            char2id[w] = index_c
                            index_c += 1
                        y.append(label2id.get(t))
                    x = x + [0] * (max_len - m)
                    y = y + [0] * (max_len - m)
                    X.append(x)
                    Y.append(y)
                    M.append(m)
        print("处理完文件：", file)

    print('读取数据完成')
    # 保存char2id 更好的可视化
    with open('./TrainData/char2id_segment.txt', 'w') as fw:
        for k, v in char2id.items():
            fw.write(k + '\t' + str(v) + '\n')
    # 保存为json格式的数据，方便读取，存储
    open('./TrainData/char2id_segment.json', 'w').write(json.dumps(char2id))

    # 保存label2id 更好的可视化
    with open('./TrainData/label2id_segment.txt', 'w') as fw:
        for k, v in label2id.items():
            fw.write(k + '\t' + str(v) + '\n')
    # 保存为json格式的数据，方便读取，存储
    open('./TrainData/label2id_segment.json', 'w').write(json.dumps(label2id))

    # 保存格式化后的训练文件
    df = pd.DataFrame()
    df['chars'] = X
    df['labels'] = Y
    df['masks'] = M
    df.to_csv('./TrainData/XYM_train_segment.csv')
    print('保存输入输出文件为csv')
    # 使用 numpy 进行存储
    X = np.array(X, dtype="int32")
    Y = np.array(Y, dtype="int32")
    M = np.array(M, dtype="int32")
    np.savez("./TrainData/XYM_train_segment.npz", X=X, Y=Y, M=M)
    print('保存输入输出文件为npz')


def load_dict():
    """
        加载char、label字典
    :return:
    """
    # Print args
    print('-' * 50 + '\t准备数据\t' + '-' * 50)
    char2id = json.loads(open('./model/char2id_segment.json').read())
    # 新增 <unk>
    char_value_dict_len = len(char2id) + 1

    label2id = json.loads(open('./model/label2id_segment.json').read())

    print('字集合大小:%d', char_value_dict_len)
    print('标签个数:%d', len(label2id))
    return char2id, label2id


def load_data():
    """
        加载训练数据
    :return:
    """
    XYM = np.load("./TrainData/XYM_train_segment.npz", mmap_mode="r")
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
    print("hahaha")
    # prepare_data_segment()
