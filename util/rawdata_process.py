# -*- encoding:utf-8 -*-
import json
import codecs
import pandas as pd
import os


# imdb genarate train data
def generate_train_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        out_pos_file = "../data/part_data/train_pos_" + str(i) + ".txt"
        out_neg_file = "../data/part_data/train_neg_" + str(i) + ".txt"
        fw1 = open(out_pos_file, 'w', encoding='utf-8')
        fw2 = open(out_neg_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[count * i:count * (i + 1)]
        for pos in pos_line:
            fw1.write(pos)
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[count * i:count * (i + 1)]
        for neg in neg_line:
            fw2.write(neg)
        print("data "+str(i)+" processed!")
    return


# imdb generate test data
def generate_test_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        out_pos_file = "../data/part_data/test_pos_" + str(i) + ".txt"
        out_neg_file = "../data/part_data/test_neg_" + str(i) + ".txt"
        fw1 = open(out_pos_file, 'w', encoding='utf-8')
        fw2 = open(out_neg_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[2500+count * i:2500+count * (i + 1)]
        for pos in pos_line:
            fw1.write(pos)
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[2500+count * i:2500+count * (i + 1)]
        for neg in neg_line:
            fw2.write(neg)
        print("data "+str(i)+" processed!")
    return


def generate_full_datafile(filetype='train'):
    pos_location = '../raw_data/' + filetype + '/pos'
    pos_files = os.listdir(pos_location)
    neg_location = '../raw_data/' + filetype + '/neg'
    neg_files = os.listdir(neg_location)
    pos_all = codecs.open('../data/' + filetype + '_pos_all.txt', 'a', encoding='utf8')
    neg_all = codecs.open('../data/' + filetype + '_neg_all.txt', 'a', encoding='utf8')
    all = []
    for file in pos_files:
        whole_location = os.path.join(pos_location, file)
        with open(whole_location, 'r', encoding='utf8') as f:
            line = f.readlines()
            all.extend(line)
    for file in all:
        pos_all.write(file)
        pos_all.write('\n')
    alls = []
    for file in neg_files:
        whole_location = os.path.join(neg_location, file)
        with open(whole_location, 'r', encoding='utf8') as f:
            line = f.readlines()
            alls.extend(line)
    for file in alls:
        neg_all.write(file)
        neg_all.write('\n')
    return


def neg_data_process(input_file, output_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        json_data = json.load(f)
    length = len(json_data['data'])
    print(length)
    print(json_data['data'][2]['_source']['result']['content'])
    result_json = {}
    fw = codecs.open(output_file, 'w', encoding='utf-8')
    fw1 = codecs.open('../data/word2vec.train.data', 'a', encoding='utf-8')
    for i in range(length):
        print(i)
        try:
            content = json_data['data'][i]['_source']['result']['content']
        except IndexError:
            content = None
        if content is None:
            continue
        content = content.replace('‘', '').replace('’', '').replace('<', '').replace('>', '').replace('\\"', '') \
            .replace('\\n', '').replace('\n', '').replace('\"', '').replace('【', '').replace('】', '') \
            .replace('\n', '').replace('\\\\', '').replace(',', '').replace(':', '').replace(';', '') \
            .replace('[', '').replace(']', '').replace('(', '').replace(')', '')  # 去掉标点符
        if content.isalnum():
            continue
        result_json['content'] = content
        fw1.write(content + '\n')
        result_json['tag'] = 0
        fw.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        i += 1
    fw.close()
    return


def pos_data_process(input_file, output_file):
    raw_data = pd.read_csv(input_file, usecols=[1], header=0)
    fw = codecs.open(output_file, 'w', encoding='utf-8')
    result_json = {}
    print(len(raw_data))
    fw1 = codecs.open('../data/word2vec.train.data', 'a', encoding='utf-8')
    for i in range(len(raw_data)):
        print(i)
        try:
            content = raw_data.ix[i][0]
        except IndexError:
            result_json['content'] = None
        if content == '[]' or content is None:
            continue
        else:
            content = content.replace('‘', '').replace('’', '').replace('<', '').replace('>', '').replace('\\"', '') \
                .replace('\\n', '').replace('\n', '').replace('\"', '').replace('【', '').replace('】', '') \
                .replace('\n', '').replace('\\\\', '').replace(',', '').replace(':', '').replace(';', '') \
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '')  # 去掉标点符号
            if content.isalnum():
                continue
        result_json['content'] = content
        word2vec_data = content
        result_json['tag'] = 1
        fw.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        for j in word2vec_data.split('\\n'):
            fw1.write(j + '\n')
        i += 1
    fw.close()
    return


def mix_two_dataset(input_file1, input_file2, output_file):
    fr1 = codecs.open(input_file1, 'r', encoding='utf-8')  # neg data
    fr2 = codecs.open(input_file2, 'r', encoding='utf-8')  # pos data
    json_data1 = []
    for line in fr1:
        json_data1.append(line)
    json_data2 = []
    for line in fr2:
        json_data2.append(line)
    fw = codecs.open(output_file, 'w', encoding='utf-8')
    result_json1 = {}
    result_json2 = {}
    for i in range(2000):
        print(i)
        try:
            # print(json_data1[i])
            result_json1['content'] = json.loads(json_data1[i])['content']
        except IndexError:
            result_json1['content'] = None
        result_json1['tag'] = 0
        result_json1['id'] = str(i)
        fw.write(json.dumps(result_json1, ensure_ascii=False) + '\n')
        i += 1

        try:
            result_json2['content'] = json.loads(json_data2[i])['content']
        except IndexError:
            result_json2['content'] = None
        result_json2['tag'] = 1
        result_json2['id'] = str(i)
        fw.write(json.dumps(result_json2, ensure_ascii=False) + '\n')
        i += 1
    fw.close()
    return


def mix_test_dataset(input_file1, input_file2, output_file):
    fr1 = codecs.open(input_file1, 'r', encoding='utf-8')  # neg data
    fr2 = codecs.open(input_file2, 'r', encoding='utf-8')  # pos data
    json_data1 = []
    for line in fr1:
        json_data1.append(line)
    json_data2 = []
    for line in fr2:
        json_data2.append(line)
    fw = codecs.open(output_file, 'w', encoding='utf-8')
    result_json1 = {}
    result_json2 = {}
    # generate different dataset 2000-2100 1900-2000
    for i in range(1900, 2000):
        print(i)
        try:
            # print(json_data1[i])
            result_json1['content'] = json.loads(json_data1[i])['content']
            result_json1['tag'] = 0
            result_json1['id'] = str(i)
            fw.write(json.dumps(result_json1, ensure_ascii=False) + '\n')
        except IndexError:
            result_json1['content'] = None
        i += 1

        try:
            result_json2['content'] = json.loads(json_data2[i])['content']
            result_json2['tag'] = 1
            result_json2['id'] = str(i)
            fw.write(json.dumps(result_json2, ensure_ascii=False) + '\n')
        except IndexError:
            result_json2['content'] = None
        i += 1
    fw.close()
    return


if __name__ == '__main__':
    # generate_full_datafile(filetype='train')
    # generate_full_datafile(filetype='test')
    # 'data/train_pos_all.txt' 'data/train_neg_all.txt'
    generate_train_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt', part=10, count=250)
    generate_test_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt', part=2, count=1000)
