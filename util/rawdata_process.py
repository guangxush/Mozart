# -*- encoding:utf-8 -*-
import json
import codecs
import pandas as pd


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
        content = content.replace('‘', '').replace('’','').replace('<', '').replace('>', '').replace('\\"', '')\
                  .replace('\\n', '').replace('\n', '').replace('\"', '').replace('【', '').replace('】', '')\
                  .replace('\n', '').replace('\\\\', '').replace(',', '').replace(':', '').replace(';', '')\
                  .replace('[', '').replace(']', '').replace('(', '').replace(')', '')  # 去掉标点符号
        # content.replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace(
        #     '“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        #     .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        #     .replace('’', '').replace('<', '').replace('>', '').replace('\\"', '').replace('\\n', '') \
        #     .replace('\n', '').replace('\"', '').replace('【', '').replace('】', '').replace('\n', '') \
        #     .replace('\\\\', '').replace(',', '').replace(':', '').replace(';', '')  # 去掉标点符号
        if content.isalnum():
            continue
        result_json['content'] = content
        fw1.write(content + '\n')
        result_json['tag'] = 0
        fw.write(json.dumps(result_json, ensure_ascii=False)+'\n')
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
            content = content.replace('‘', '').replace('’','').replace('<', '').replace('>', '').replace('\\"', '')\
                .replace('\\n', '').replace('\n', '').replace('\"', '').replace('【', '').replace('】', '')\
                .replace('\n', '').replace('\\\\', '').replace(',', '').replace(':', '').replace(';', '')\
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '')  # 去掉标点符号
            if content.isalnum():
                continue
        result_json['content'] = content
        word2vec_data = content
        result_json['tag'] = 1
        fw.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        for j in word2vec_data.split('\\n'):
            fw1.write(j+'\n')
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
        fw.write(json.dumps(result_json1, ensure_ascii=False)+'\n')
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
    # neg_data_process(input_file='../raw_data/dajiyuan_2w.json', output_file='../data/neg_data.json')
    # pos_data_process(input_file='../raw_data/renming.csv', output_file='../data/pos_data.json')
    # mix_two_dataset(input_file1='../data/neg_data.json', input_file2='../data/pos_data.json',
    #                 output_file='../data/mix_data.json')
    # mix_test_dataset(input_file1='../data/neg_data.json', input_file2='../data/pos_data.json',
    #                  output_file='../data/mix_test_data.json')
    # mix_test_dataset(input_file1='../data/neg_data.json', input_file2='../data/pos_data.json',
    #                  output_file='../data/mix_model2_train_data.json')
    mix_test_dataset(input_file1='../data/neg_data.json', input_file2='../data/pos_data.json',
                     output_file='../data/mix_model2_test_data.json')