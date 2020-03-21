#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-12-07 20:51
"""
import os
import re
import json
import tensorflow as tf
import tokenization

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
model_path = "./model/"
vocab_file = f"./{model_path}/vocab.txt"
tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
label2id = json.loads(open(f"./{model_path}/label2id.json").read())
id2label = [k for k, v in label2id.items()]
label2desc = json.loads(open(f"./{model_path}/label2desc.json").read())


def process_one_example_p(tokenizer, text, max_seq_len=128):
    textlist = list(text)
    tokens = []
    # labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        # labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        # label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    sess_ = tf.Session()
    saver.restore(sess_, input_checkpoint)

    # opts = sess_.graph.get_operations()
    # for v in opts:
    #     print(v.name)
    return sess_


def load_pb_model(path):
    tf.reset_default_graph()
    output_graph_def = tf.GraphDef()

    with open(path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess_ = tf.Session()
    init = tf.global_variables_initializer()
    sess_.run(init)
    return sess_


# sess = load_model(os.path.join(f"{model_path}", ""))
sess = load_pb_model(os.path.join(f"{model_path}", "nlp_model.pb"))
input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")  # fc/dense/Relu  cnn_block/Reshape
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
p = sess.graph.get_tensor_by_name("loss/ReverseSequence_1:0")


def split_sentence(text):
    sentences = re.split(r"([。!！?？；;，])", text)
    aaa = []
    for i in zip(sentences[0::2], sentences[1::2]):
        _ = "".join(i)
        if _:
            aaa.append(_)
    return aaa


def predict(text):
    add = False
    if not re.search(r"([。!！?？；;，])", text[-1]):
        add = True
        text += "。"
    text = re.sub(r"\s+", "", text)
    data = []
    if len(text) > 62:
        sentences = split_sentence(text.strip())
        temp = ""
        for s in sentences:
            if len(s) >= 62:
                continue
            if len(temp + s) >= 62:
                data.append(temp)
                temp = s
            else:
                temp += s
        if len(temp):
            data.append(temp)
    else:
        data.append(text)
    # print(data)
    # 逐个分成 最大62长度的 text 进行 batch 预测
    features = []
    for i in data:
        feature = process_one_example_p(tokenizer_, i, max_seq_len=64)
        features.append(feature)
    feed = {input_ids: [feature[0] for feature in features],
            input_mask: [feature[1] for feature in features],
            segment_ids: [feature[2] for feature in features],
            keep_prob: 1.0
            }

    [probs] = sess.run([p], feed)
    result = []
    for index, prob in enumerate(probs):
        for v in prob[1:len(data[index]) + 1]:
            result.append(id2label[int(v)])
    # 解码需要再优化一下，输出还算可以，95% 拟合了百度的nlp生成数据
    # print(result)
    segment_words = []
    segment_tags = []
    # 默认 不会编码出错，即不符合 S B M E 顺序(解码可能不一定会完全符合吧)
    temp = ""
    for w, t in zip("".join(data), result):
        temp += w
        if re.search("^[ES]", t):
            segment_words.append(temp)
            segment_tags.append(t[2:])
            temp = ""
    if add:
        segment_words = segment_words[:-1]
        segment_tags = segment_tags[:-1]
    res = []
    for w, t in zip(segment_words, segment_tags):
        res.append({
            "word": w,
            "tag": t,
            "desc": label2desc.get(t, "未知")
        })
    return res


if __name__ == "__main__":
    import time
    import jieba

    # text_ = "15日，备受关注的电影《黄金时代》在北京举行了电影发布会，导演许鞍华和编剧李樯及汤唯、冯绍峰等众星悉数亮相。" \
    #         "据悉，电影确定将于10月1日公映。本片讲述了“民国四大才女”之一的萧红短暂而传奇的一生，" \
    #         "通过她与萧军、汪恩甲、端木蕻良、洛宾基四人的情感纠葛，与鲁迅、丁玲等人一起再现上世纪30年代的独特风貌。" \
    #         "电影原名《穿过爱情的漫长旅程》，后更名《黄金时代》，这源自萧红写给萧军信中的一句话：“这不正是我的黄金时代吗？”"

    # text_ = "为什么商户交易会失败？如何查询"

    text_ = "1996年，曾经是微软员工的加布·纽维尔和麦克·哈灵顿一同创建了Valve软件公司。他们在1996年下半年从id software取得了雷神之锤引擎的使用许可，用来开发半条命系列。"
    # text_ = "据悉，电影确定将于10月1日公映。本片讲述了“民国四大才女”之一的萧红短暂而传奇的一生，"

    # text_ = "这源自萧红写给萧军信中的一句话"

    # 有的ner确实不对 ['B_r', 'M_r', 'E_r', 'B_d', 'E_d', 'B_v', 'E_v', 'S_w', 'B_r', 'E_r', 'S_n', 'S_u', 'B_d',
    # 'E_d', 'B_a', 'E_a', 'S_w', 'B_PER', 'E_PER', 'E_PER', 'S_p', 'B_n', 'E_n', 'B_v', 'E_v', 'S_xc', 'S_w']
    print(" ".join(jieba.cut(text_)))
    res = predict(text_)
    t1 = time.time()
    for i in range(100):
        res = predict(text_)
    t2 = time.time()
    for i in res:
        print(i)
