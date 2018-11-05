# -*- encoding:utf-8 -*-
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def load_data(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values

    x_train = train_dataset[0:21, 0:-1].astype('float32')
    y_train = train_dataset[0:21, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'iris_1_data.csv'), header=0)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[21:, 0:-1].astype('float32')
    y_dev = dev_dataset[21:, -1].astype('int')
    encoder = LabelBinarizer()
    y_dev = encoder.fit_transform(y_dev)
    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'iris_2_data.csv'), header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[:, 0:-1].astype('float32')

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


if __name__ == '__main__':
    load_data(data_path='../data/')