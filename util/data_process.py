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


def get_imdb_part_data(raw_file):
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
    # one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    # word_index = tokenizer.word_index
    # sourcevocabsize = len(word_index)
    sequences = tokenizer.texts_to_sequences(seq)
    # 此处设置每个句子最长不超过 800
    final_sequences = sequence.pad_sequences(sequences, maxlen=800)

    # 转换为numpy类型
    label = np.array(label)
    # 随机打乱数据
    # indices = np.random.permutation(len(final_sequences) - 1)
    # X = final_sequences[indices]
    # y = label[indices]
    X = final_sequences
    y = label
    # 划分测试集和训练集
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    print("dataset created!")
    return Xtrain, Xtest, ytrain, ytest


def get_imdb_test_data(raw_file):
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
    # one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
    # word_index = tokenizer.word_index
    # word_index = tokenizer.word_index
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
    get_imdb_part_data("../data/part_data_all/train_1.txt")