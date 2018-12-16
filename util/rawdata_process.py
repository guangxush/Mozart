# -*- encoding:utf-8 -*-
import json
import codecs
import pandas as pd
import os


# imdb genarate train data
def generate_imdb_train_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        # out_pos_file = "../data/part_data/train_pos_" + str(i) + ".txt"
        # out_neg_file = "../data/part_data/train_neg_" + str(i) + ".txt"
        out_all_file = "../data/part_data_all/train_" + str(i) + ".txt"
        # fw1 = open(out_pos_file, 'w', encoding='utf-8')
        # fw2 = open(out_neg_file, 'w', encoding='utf-8')
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[count * i:count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[count * i:count * (i + 1)]
        for k in range(len(pos_line)):
            fw.write(pos_line[k].rstrip('\n')+'@@@1\n')
            fw.write(neg_line[k].rstrip('\n')+'@@@0\n')
        print("data "+str(i)+" processed!")
    return


# imdb generate test data
def generate_imdb_test_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        # out_pos_file = "../data/part_data/test_pos_" + str(i) + ".txt"
        # out_neg_file = "../data/part_data/test_neg_" + str(i) + ".txt"
        # fw1 = open(out_pos_file, 'w', encoding='utf-8')
        # fw2 = open(out_neg_file, 'w', encoding='utf-8')
        out_all_file = "../data/part_data_all/test_" + str(i) + ".txt"
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[2500+count * i:2500+count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[2500+count * i:2500+count * (i + 1)]
        for k in range(len(neg_line)):
            fw.write(pos_line[k].rstrip('\n') + '@@@1\n')
            fw.write(neg_line[k].rstrip('\n') + '@@@0\n')
        print("data "+str(i)+" processed!")
    return


if __name__ == '__main__':
    generate_imdb_train_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt', part=10, count=250)
    generate_imdb_test_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt', part=2, count=50)
