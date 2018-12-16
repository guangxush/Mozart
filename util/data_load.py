# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from util import data_process
from model.model1 import lstm_mul_model
from sklearn.utils import shuffle
from util.data_process import get_imdb_vocab_size
import os
K.set_image_dim_ordering('th')


# load the data which model2 train
def load_data2(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    total_count = train_dataframe.shape[0]
    train_level = int(total_count * 0.7)
    train_dataset = train_dataframe.values
    train_dataset = shuffle(train_dataset)
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


# load the all data which model2 train
def load_all_data2(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    total_count = train_dataframe.shape[0]
    train_dataset = train_dataframe.values
    x_train = train_dataset[0:total_count, 0:-1].astype('float')
    y_train = train_dataset[0:total_count, -1].astype('int')
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    return x_train, y_train


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


# save the error dataset depends on the model result and the truth label
def make_err_dataset(result_path, label, x_test, y_test):
    print("pred:", end='')
    print(label)
    print("true:", end='')
    print(y_test)
    count = 0
    err_data_list = []
    for i in label:
        if i != y_test[count]:
            err_result = np.append(x_test[count], y_test[count]).tolist()
            err_data_list.append(err_result)
        count += 1
    err_data = pd.DataFrame(err_data_list)
    err_data.to_csv(result_path, encoding='utf-8', header=1, index=0)


def generate_imdb_model2_data(model_file, test_file, result_path, count):
    labels = []
    x_test, y_test = data_process.get_imdb_test_data(raw_file=test_file)
    for i in range(1, count+1):
        yi_test = generate_imdb_model2(model_name=model_file + str(i) + ".h5", x_test=x_test,
                                       line_count=100, train_file="./data/part_data_all/train_" + str(i) + ".txt")
        print("yi_test len: " + str(len(yi_test)))
        if i == 1:
            print("----------")
            z_data = yi_test
        else:
            z_data = np.c_[z_data, yi_test]
        labels.append("test" + str(i))
    print(len(y_test))
    labels.append("test")
    z_data = np.c_[z_data, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = labels
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


# generate the model labels from model1 result
def generate_imdb_model2(model_name, x_test, line_count, train_file):
    if not os.path.exists(model_name):
        print(model_name)
        print("file not found!")
        # if file not exists, return [0]*30
        return np.array([0] * line_count)
    vocab_size = get_imdb_vocab_size(train_file)
    lstm_model = lstm_mul_model(vocab_size)
    lstm_model.load_weights(model_name)
    # results = lstm_model.predict(x_test)
    results = lstm_model.predict_classes(x_test)
    return results
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    load_all_data()