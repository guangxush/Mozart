# -*- coding:utf-8 -*-

import numpy as np
import pickle
import json
import math, codecs
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def load_vec_txt(fname, vocab, k=100):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f:
        if len(line) < k:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)
    for word in vocab:
        if not w2v.__contains__(word):
            print('UNK---------------- ', word)
            w2v[word] = w2v["**UNK**"]
            unknowtoken += 1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]
    print('UnKnown tokens in w2v', unknowtoken)
    return k, W


def load_vec_character(vocab_c_inx, k=30):
    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))
    for i in vocab_c_inx:
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)
    return W, k


def load_vec_onehot(vocab_w_inx):
    """
    Loads 100x1 word vecs from word2vec
    """
    k = vocab_w_inx.__len__()
    W = np.zeros(shape=(vocab_w_inx.__len__()+1, k+1))
    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.
    return k, W


def make_idx_posi_index(file, max_s):
    data_s_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        p_sent = sent['positions']
        data_p = []
        if len(p_sent) > max_s:
            for i in range(0, max_s):
                list = p_sent[i]
                data_p.append(list)
        else:
            for i in range(len(p_sent)):
                list = p_sent[i]
                data_p.append(list)
            while len(data_p) < max_s:
                list = np.zeros(4)
                data_p.append(list.tolist())
        data_s_all.append(data_p)
    f.close()
    return data_s_all


def make_idx_word_index(file, max_s, max_c, source_vob, target_vob, target_1_vob, source_char):

    data_s_all = []
    data_t_all = []
    data_c_all = []
    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    for num, line in enumerate(fr):
        print(num)
        if len(line) <=1:
            continue
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['words']
        t_sent = sent['label']
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)
        targetvec = np.zeros(len(target_vob))
        targetvec[target_vob[t_sent]] = 1
        data_t_all.append(targetvec)
        data_w = []
        for ii in range(0, min(max_s, len(s_sent))):
            word = s_sent[ii]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])

            num = max_c - word.__len__()
            for i in range(0, max(num, 0)):
                data_c.append(0)

            data_w.append(data_c)

        num = max_s - len(s_sent)
        for inum in range(0, num):
            data_tmp = []
            for i in range(0, max_c):
                data_tmp.append(0)
            data_w.append(data_tmp)
        data_c_all.append(data_w)

    f.close()
    return data_s_all, data_t_all, data_c_all


def get_char_index(files):

    source_vob = {}
    sourc_idex_word = {}
    count = 1
    max_s = 0
    dict = {}
    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for word in sourc:
                for i in range(len(word)):
                    if not source_vob.__contains__(word[i]):
                        source_vob[word[i]] = count
                        sourc_idex_word[count] = word[i]
                        count += 1
                if word.__len__() in dict.keys():
                    dict[word.__len__()] = dict[word.__len__()]+1
                else:
                    dict[word.__len__()] = 1
                if word.__len__() > max_s:
                    max_s = word.__len__()
        f.close()

    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count += 1
    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_s


def get_feature_index(file):
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0
    for labelingfile in file:
        f = open(labelingfile, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue
            sourc = line.strip('\r\n').rstrip('\n').split(' ')[1]
            if not label_vob.__contains__(sourc):
                label_vob[sourc] = count
                label_idex_word[count] = sourc
                count += 1
        f.close()
    if not label_vob.__contains__("**UNK**"):
        label_vob["**UNK**"] = count
        label_idex_word[count] = "**UNK**"
        count += 1
    return label_vob, label_idex_word


def get_word_index(files, testfile):
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    max_s = 0
    tarcount = 0
    count = 1
    dict = {}
    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for word in sourc:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1

            if sourc.__len__() in dict.keys():
                dict[sourc.__len__()] = dict[sourc.__len__()] + 1
            else:
                dict[sourc.__len__()] = 1
            if sourc.__len__() > max_s:
                max_s = sourc.__len__()
                # print('max_s  ', max_s, sourc)
            target = sent['label']
            if not target_vob.__contains__(target):
                target_vob[target] = tarcount
                target_idex_word[tarcount] = target
                tarcount += 1
        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    f = codecs.open(testfile, 'r', encoding='utf-8')
    fr = f.readlines()
    for line in fr:
        if line.__len__() <= 1:
            continue
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['words']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

    f.close()
    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def make_idx_char_index(trainfile, max_s, max_c, source_char):
    data_c_all = []

    f1 = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f1.readlines()
    for num, line in enumerate(lines):
        print(num)
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['words']
        data_w = []
        for word in sourc:
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])

            num = max_c - word.__len__()
            for i in range(0, max(num, 0)):
                data_c.append(0)
            data_w.append(data_c)
        num = max_s - len(sourc)
        for inum in range(0, num):
            data_tmp = []
            for i in range(0, max_c):
                data_tmp.append(0)
            data_w.append(data_tmp)
        data_c_all.append(data_w)
    f1.close()
    return data_c_all


def get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen = 50):
    char_vob, vob_idex_char, max_c = get_char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6
    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))
    max_s = 800

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:', char_W.shape)

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, None,
                                                                char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))

    extra_test_num = int(len(train_all) / 10)
    left = 0
    right = 1
    test = train_all[extra_test_num * left:extra_test_num * right]
    test_label = target_all[extra_test_num * left:extra_test_num * right]
    train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
    train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
    print('extra_test_num', extra_test_num)
    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))

    test_char = train_all_char[extra_test_num * left:extra_test_num * right]
    train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
    print('test_char len  ', test_char.__len__(), )
    print('train_char len  ', train_char.__len__())

    print("dataset created!")
    out = codecs.open(datafile, 'wb')
    pickle.dump([train, train_char, train_label,
                 test, test_char, test_label,
                 word_vob, vob_idex_word, word_W, word_k,
                 target_vob, vob_idex_target,
                 char_vob, vob_idex_char, char_W, char_k,
                 max_s, max_c
                 ], out, 0)
    out.close()


def get_part_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen = 50, left=0):
    char_vob, vob_idex_char, max_c = get_char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6
    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))
    max_s = 800

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:', char_W.shape)

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, None,
                                                                char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))

    extra_test_num = int(len(train_all) / 10)
    right = left + 1
    test = train_all[extra_test_num * left:extra_test_num * right]
    test_label = target_all[extra_test_num * left:extra_test_num * right]
    train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
    train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
    print('extra_test_num', extra_test_num)
    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))

    test_char = train_all_char[extra_test_num * left:extra_test_num * right]
    train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
    print('test_char len  ', test_char.__len__(), )
    print('train_char len  ', train_char.__len__())

    print("dataset created!")
    out = codecs.open(datafile, 'wb')
    pickle.dump([train, train_char, train_label,
                 test, test_char, test_label,
                 word_vob, vob_idex_word, word_W, word_k,
                 target_vob, vob_idex_target,
                 char_vob, vob_idex_char, char_W, char_k,
                 max_s, max_c
                 ], out, 0)
    out.close()


def get_part_train_test_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen = 50, left=0):
    char_vob, vob_idex_char, max_c = get_char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6
    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))
    max_s = 800

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:', char_W.shape)

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, None,
                                                                char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))

    test_all, test_target_all, test_all_char = make_idx_word_index(testfile, max_s, max_c, word_vob, target_vob, None,
                                                                   char_vob)
    print('test_all size', len(test_all), 'test_target_all', len(test_target_all))
    print('test_all_char size', len(test_all_char))

    extra_train_num = int(len(train_all) / 10)
    extra_test_num = int(len(test_all) / 10)

    right = left + 1
    test = test_all[extra_test_num * left:extra_test_num * right]
    test_label = test_target_all[extra_test_num * left:extra_test_num * right]
    train = train_all[extra_train_num * left:extra_train_num * right]
    train_label = target_all[extra_train_num * left:extra_train_num * right]

    print('extra_train_num', extra_train_num)
    print('train len  ', train.__len__(), len(train_label))
    print('extra_test_num', extra_test_num)
    print('test len  ', test.__len__(), len(test_label))

    test_char = test_all_char[extra_test_num * left:extra_test_num * right]
    train_char = train_all_char[extra_train_num * left:extra_train_num * right]
    print('test_char len  ', test_char.__len__(), )
    print('train_char len  ', train_char.__len__())

    print("dataset created!")
    out = codecs.open(datafile, 'wb')
    pickle.dump([train, train_char, train_label,
                 test, test_char, test_label,
                 word_vob, vob_idex_word, word_W, word_k,
                 target_vob, vob_idex_target,
                 char_vob, vob_idex_char, char_W, char_k,
                 max_s, max_c
                 ], out, 0)
    out.close()


def data_divide(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen = 50, part=10):
    char_vob, vob_idex_char, max_c = get_char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6
    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))
    max_s = 800

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:', char_W.shape)

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, None,
                                                                char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))

    test_all, test_target_all, test_all_char = make_idx_word_index(testfile, max_s, max_c, word_vob, target_vob, None,
                                                                   char_vob)
    print('test_all size', len(train_all), 'test_target_all', len(target_all))
    print('test_all_char size', len(train_all_char))

    extra_train_num = int(len(train_all) / 10)
    extra_test_num = int(len(test_all) / 10)
    for left in range(0, part):
        right = left + 1
        data_file = datafile + str(right) + ".pkl"

        test = test_all[extra_test_num * left:extra_test_num * right]
        test_label = test_all[extra_test_num * left:extra_test_num * right]
        train = train_all[extra_train_num * left:extra_train_num * right]
        train_label = target_all[extra_train_num * left:extra_train_num * right]

        print('extra_train_num', extra_train_num)
        print('train len  ', train.__len__(), len(train_label))
        print('extra_test_num', extra_test_num)
        print('test len  ', test.__len__(), len(test_label))

        test_char = test_all_char[extra_test_num * left:extra_test_num * right]
        train_char = train_all_char[extra_train_num * left:extra_train_num * right]
        print('test_char len  ', test_char.__len__(), )
        print('train_char len  ', train_char.__len__())

        print("dataset created!")
        out = codecs.open(data_file, 'wb')
        pickle.dump([train, train_char, train_label,
                     test, test_char, test_label,
                     word_vob, vob_idex_word, word_W, word_k,
                     target_vob, vob_idex_target,
                     char_vob, vob_idex_char, char_W, char_k,
                     max_s, max_c
                     ], out, 0)
        out.close()


def get_imdb_part_data(pos_file, neg_file):
    pos_list = []
    with open(pos_file, 'r', encoding='utf8')as f:
        line = f.readlines()
        pos_list.extend(line)
    neg_list = []
    with open(neg_file, 'r', encoding='utf8')as f:
        line = f.readlines()
        neg_list.extend(line)
    # 创建标签
    label = [1 for i in range(250)]
    label.extend([0 for i in range(250)])
    # 评论内容整合
    content = pos_list.extend(neg_list)
    content = pos_list
    # 去掉停用词和标点符号
    seq = []
    seqtence = []
    # nltk.download("stopwords")
    # nltk.download("punkt")
    stop_words = set(stopwords.words('english'))
    for con in content:
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        seq.append(line)
        seqtence.extend(line)

    # 获取词索引
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(content)
    one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    word_index = tokenizer.word_index
    #  sourcevocabsize = len(word_index)
    sequences = tokenizer.texts_to_sequences(seq)
    # 此处设置每个句子最长不超过 800
    final_sequences = sequence.pad_sequences(sequences, maxlen=800)

    # 转换为numpy类型
    label = np.array(label)
    # 随机打乱数据
    indices = np.random.permutation(len(final_sequences) - 1)
    X = final_sequences[indices]
    y = label[indices]
    # 划分测试集和训练集
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    print("dataset created!")
    return Xtrain, Xtest, ytrain, ytest


def get_imdb_part_data2(raw_file):
    fr = open(raw_file, 'r', encoding='utf8')
    content = []
    label = []
    for line in fr:
        line = line.split('@@@')
        content.append(line[0])
        label.append(int(line[1]))

    # 去掉停用词和标点符号
    seq = []
    seqtence = []
    # nltk.download("stopwords")
    # nltk.download("punkt")
    stop_words = set(stopwords.words('english'))
    for con in content:
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        seq.append(line)
        seqtence.extend(line)

    # 获取词索引
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(content)
    one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    word_index = tokenizer.word_index
    #  sourcevocabsize = len(word_index)
    sequences = tokenizer.texts_to_sequences(seq)
    # 此处设置每个句子最长不超过 800
    final_sequences = sequence.pad_sequences(sequences, maxlen=800)

    # 转换为numpy类型
    label = np.array(label)
    # 随机打乱数据
    indices = np.random.permutation(len(final_sequences) - 1)
    X = final_sequences[indices]
    y = label[indices]
    # 划分测试集和训练集
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    print("dataset created!")
    return Xtrain, Xtest, ytrain, ytest


def get_imdb_test_data(pos_file, neg_file):
    pos_list = []
    with open(pos_file, 'r', encoding='utf8')as f:
        line = f.readlines()
        pos_list.extend(line)
    neg_list = []
    with open(neg_file, 'r', encoding='utf8')as f:
        line = f.readlines()
        neg_list.extend(line)
    # 创建标签
    label = [1 for i in range(500)]
    label.extend([0 for i in range(500)])
    # 评论内容整合
    content = pos_list.extend(neg_list)
    content = pos_list
    # 去掉停用词和标点符号
    seq = []
    seqtence = []
    # nltk.download("stopwords")
    # nltk.download("punkt")
    stop_words = set(stopwords.words('english'))
    for con in content:
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        seq.append(line)
        seqtence.extend(line)

    # 获取词索引
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(content)
    one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    word_index = tokenizer.word_index
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(seq)
    # 此处设置每个句子最长不超过 800
    final_sequences = sequence.pad_sequences(sequences, maxlen=800)

    # 转换为numpy类型
    label = np.array(label)
    # buneng随机打乱数据
    # indices = np.random.permutation(len(final_sequences))
    X = final_sequences
    y = label
    # 划分测试集和训练集
    print("testdata created!")
    return X, y


def get_imdb_test_data2(raw_file):
    fr = open(raw_file, 'r', encoding='utf8')
    content = []
    label = []
    for line in fr:
        line = line.split('@@@')
        content.append(line[0])
        label.append(int(line[1]))

    # 去掉停用词和标点符号
    seq = []
    seqtence = []
    # nltk.download("stopwords")
    # nltk.download("punkt")
    stop_words = set(stopwords.words('english'))
    for con in content:
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        seq.append(line)
        seqtence.extend(line)

    # 获取词索引
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(content)
    one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    word_index = tokenizer.word_index
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(seq)
    # 此处设置每个句子最长不超过 800
    final_sequences = sequence.pad_sequences(sequences, maxlen=800)

    # 转换为numpy类型
    label = np.array(label)
    # buneng随机打乱数据
    # indices = np.random.permutation(len(final_sequences))
    X = final_sequences
    y = label
    # 划分测试集和训练集
    print("testdata created!")
    return X, y


if __name__ == "__main__":
    maxlen = 50
    trainfile = "../data/mix_data_train_data.json"
    testfile = "../data/mix_data_test_data.json"
    w2v_file = "../modfile/Word2Vec.mod"
    char2v_file = "../modfile/Char2Vec.mod"
    w2v_k = 100
    c2v_k = 100
    datafile = "../modfile/model1_data/data_"
    modelfile = "../modfile/model.pkl"
    datafile = "../modfile/data.pkl"
    get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen=50)
    # data_divide(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen=50, part=5)