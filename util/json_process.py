# -*- encoding:utf-8 -*-
import json
import codecs


def json_process(input_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        json_data = json.load(f)
    print(json_data['data'][0]['_type'])
    return


if __name__ == '__main__':
    json_process(input_file='../raw_data/dajiyuan_2w.json')