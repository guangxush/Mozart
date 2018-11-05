# -*- encoding:utf-8 -*-
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np


def load_data1(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values

    x_train = train_dataset[0:21, 0:-1].astype('float')
    y_train = train_dataset[0:21, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[21:, 0:-1].astype('float')
    y_dev = dev_dataset[21:, -1].astype('int')
    encoder = LabelBinarizer()
    y_dev = encoder.fit_transform(y_dev)
    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'iris_2_data.csv'), header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def load_data2(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    x_train = train_dataset[0:21, 0:-1].astype('float')
    y_train = train_dataset[0:21, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[21:, 0:-1].astype('float')
    y_test = test_dataset[21:, -1].astype('int')
    encoder = LabelBinarizer()
    y_test = encoder.fit_transform(y_test)
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_test, y_test


def make_err_dataset(result_path, label, x_test, y_test):
    count = 0
    err_data_list = []
    for i in label:
        if i != y_test[count]:
            err_result = np.append(x_test[count], y_test[count]).tolist()
            print(err_result)
            err_data_list.append(err_result)
        count += 1
    err_data = pd.DataFrame(err_data_list)
    err_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_label']
    print(err_data)
    err_data.to_csv(result_path, encoding='utf-8', header=1, index=0)


def load_testset(data_path):
    test_dataframe = pd.read_csv(os.path.join(data_path, 'iris_3_data.csv'), header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)
    return x_test, y_test


if __name__ == '__main__':
    # load_data1(data_path='../data/')
    load_data2(data_path='../model2_data/')