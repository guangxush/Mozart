# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import backend as K
import codecs
from make_predict import generate_result
from util import data_process
from model.model1 import lstm_model
import pickle
import os
import json
K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)


# load the data which model1 train
def load_data1(trainfile, testfile):
    X_train, y_train, num_classes = pickle.load(open(trainfile, 'rb'))

    print('X train shape:', X_train.shape)
    print('y train shape:', y_train.shape)

    X_test, y_test, num_classes = pickle.load(open(testfile, 'rb'))
    print('X test shape:', X_test.shape)
    print('y test shape:', y_test.shape)

    return X_train, y_train, X_test, y_test, num_classes


# load the data which model2 train
def load_data2(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    total_count = train_dataframe.shape[0]
    train_level = int(total_count * 0.7)
    train_dataset = train_dataframe.values
    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    y_train = train_dataset[0:train_level, -1].astype('int')
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)
    # print(y_train)

    test_dataframe = pd.read_csv(data_path, header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[train_level:, 0:-1].astype('float')
    y_test = test_dataset[train_level:, -1].astype('int')
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)
    y_test = np_utils.to_categorical(y_test, num_classes=2)

    return x_train, y_train, x_test, y_test


# load the data which modle2 test
def load_data3(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    test_dataset = train_dataframe.values
    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    print('X test shape:', x_test.shape)
    print('y test shape:', x_test.shape)
    print('finished!')
    return x_test, y_test


# load the data which generate test dataset
def load_testset(data_path):
    test_dataframe = pd.read_csv(data_path, header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)
    return x_test, y_test


# load the total data which model train
def load_all_data(train_file):
    train_dataframe = pd.read_csv(train_file, header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    total_count = train_dataframe.shape[0]
    train_level = int(total_count * 0.7)
    test_level = int(total_count * 0.9)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    y_train = train_dataset[0:train_level, -1].astype('int')
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    x_dev = train_dataset[train_level:test_level, 0:-1].astype('float')
    y_dev = train_dataset[train_level:test_level, -1].astype('int')
    y_dev = np_utils.to_categorical(y_dev, num_classes=2)
    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    x_test = train_dataset[test_level:, 0:-1].astype('float')
    y_test = train_dataset[test_level:, -1].astype('int')
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


# generate the model labels from model1 result
def generate_model2_label(file_name, mlp_model, x_test):
    group_count = 60
    if not os.path.exists(file_name):
        print(file_name)
        print("file not found!")
        # if file not exists, return [0]*30
        return np.array([0] * group_count)
    mlp_model.load_weights(file_name)
    results = mlp_model.predict(x_test)
    label = np.argmax(results, axis=1)
    # print(label)
    return label
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


# generate model2 data
def generate_model2_data_old(model_name, datafile, model_file, testfile, result_path, batch_size):
    y1_test = generate_result(model_name=model_name, datafile=datafile + "1.pkl", model_file=model_file + "1.pkl",
                              testfile=testfile, batch_size=batch_size)
    y2_test = generate_result(model_name=model_name, datafile=datafile + "2.pkl", model_file=model_file + "2.pkl",
                              testfile=testfile, batch_size=batch_size)
    y3_test = generate_result(model_name=model_name, datafile=datafile + "3.pkl", model_file=model_file + "3.pkl",
                              testfile=testfile, batch_size=batch_size)
    y4_test = generate_result(model_name=model_name, datafile=datafile + "4.pkl", model_file=model_file + "4.pkl",
                              testfile=testfile, batch_size=batch_size)
    y5_test = generate_result(model_name=model_name, datafile=datafile + "5.pkl", model_file=model_file + "5.pkl",
                              testfile=testfile, batch_size=batch_size)
    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    y_test = []
    for num, line in enumerate(lines):
        item = json.loads(line.rstrip('\n'))
        label = item['label']
        y_test.append(label)
    print(len(y_test))
    print(len(y1_test))
    z_data = np.c_[y1_test, y2_test, y3_test, y4_test, y5_test, np.array(y_test)]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'test']
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


# generate model2 data
def generate_model2_data(model_name, datafile, model_file, testfile, result_path, batch_size, count):
    labels = []
    for i in range(1, count+1):
        yi_test = generate_result(model_name=model_name, datafile=datafile + str(i) + ".pkl", model_file=model_file
                                  + str(i) + ".pkl", testfile=testfile, batch_size=batch_size, count=count)
        print("yi_test len: " + str(len(yi_test)))
        if i == 1:
            print("----------")
            z_data = yi_test
        else:
            z_data = np.c_[z_data, yi_test]
        labels.append("test" + str(i+1))
    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    y_test = []
    for num, line in enumerate(lines):
        item = json.loads(line.rstrip('\n'))
        label = item['label']
        y_test.append(label)
    print(len(y_test))
    # train, train_char, train_label, \
    # test, test_char, test_label, \
    # word_vob, vob_idex_word, word_W, word_k, \
    # target_vob, vob_idex_target, \
    # char_vob, vob_idex_char, char_W, char_k, \
    # max_s, max_c = pickle.load(open(datafile + str(i) + ".pkl", 'rb'))
    # test_length = len(test_label)
    # print(test_length)
    labels.append("test")
    z_data = np.c_[z_data, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = labels
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


# save the error dataset depends on the model result and the truth label
def make_err_dataset(result_path, label, x_test, y_test):
    count = 0
    err_data_list = []
    for i in label:
        if i != y_test[count]:
            err_result = np.append(x_test[count], y_test[count]).tolist()
            err_data_list.append(err_result)
        count += 1
    err_data = pd.DataFrame(err_data_list)
    err_data.to_csv(result_path, encoding='utf-8', header=1, index=0)


def generate_imdb_model2_data(model_file, test_pos_file, test_neg_file, result_path, count):
    labels = []
    model = lstm_model()
    x_test, y_test = data_process.get_imdb_part_data(pos_file=test_pos_file,
                                                     neg_file=test_neg_file)
    for i in range(0, count):
        yi_test = generate_imdb_model2(model_name=model_file + str(i) + ".h5", lstm_model=model, x_test=x_test,
                                       line_count=1000)
        print("yi_test len: " + str(len(yi_test)))
        if i == 1:
            print("----------")
            z_data = yi_test
        else:
            z_data = np.c_[z_data, yi_test]
        labels.append("test" + str(i+1))
    print(len(y_test))
    labels.append("test")
    z_data = np.c_[z_data, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = labels
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


# generate the model labels from model1 result
def generate_imdb_model2(model_name, lstm_model, x_test, line_count):
    if not os.path.exists(model_name):
        print(model_name)
        print("file not found!")
        # if file not exists, return [0]*30
        return np.array([0] * line_count)
    lstm_model.load_weights(model_name)
    results = lstm_model.predict(x_test)
    label = np.argmax(results, axis=1)
    return label
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    # load_data1(data_path='../data/')
    # load_data2(data_path='../data/model2_data/iris_2_data.csv')
    load_all_data()