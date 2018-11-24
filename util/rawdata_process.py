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
            result_json['content'] = json_data['data'][i]['_source']['result']['content']
            fw1.write(json_data['data'][i]['_source']['result']['content']+'\n')
        except IndexError:
            result_json['content'] = None
        result_json['tag'] = -1
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
            result_json['content'] = raw_data.ix[i][0]
            word2vec_data = raw_data.ix[i][0]
        except IndexError:
            result_json['content'] = None
        result_json['tag'] = -1
        fw.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        for j in word2vec_data.split('\\n'):
            fw1.write(j+'\n')
        i += 1
    fw.close()
    return


if __name__ == '__main__':
    neg_data_process(input_file='../raw_data/dajiyuan_2w.json', output_file='../data/neg_data.json')
    # pos_data_process(input_file='../raw_data/renming.csv', output_file='../data/pos_data.json')