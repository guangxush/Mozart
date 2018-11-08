# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


# load the data which model1 train
def load_data1(train_file, test_file):
    train_dataframe = pd.read_csv(train_file, header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    total_count = 30
    train_level = int(total_count*0.7)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    y_train = train_dataset[0:train_level, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(train_file, header=0)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[train_level:, 0:-1].astype('float')
    y_dev = dev_dataset[train_level:, -1].astype('int')
    encoder = LabelBinarizer()
    y_dev = encoder.fit_transform(y_dev)
    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(test_file, header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


# load the data which model2 train
def load_data2(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    total_count = 30
    train_level = int(total_count*0.7)
    train_dataset = train_dataframe.values
    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    y_train = train_dataset[0:train_level, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    test_dataframe = pd.read_csv(data_path, header=0)
    test_dataset = test_dataframe.values
    # print(test_dataset)

    x_test = test_dataset[train_level:, 0:-1].astype('float')
    y_test = test_dataset[train_level:, -1].astype('int')
    encoder = LabelBinarizer()
    y_test = encoder.fit_transform(y_test)
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_test, y_test


# load the data which modle2 test
def load_data3(data_path):
    train_dataframe = pd.read_csv(data_path, header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    x_train = train_dataset[:, 0:-1].astype('float')
    y_train = train_dataset[:, -1].astype('int')
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)
    print('finished!')
    return x_train, y_train


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
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    x_dev = train_dataset[train_level:test_level, 0:-1].astype('float')
    y_dev = train_dataset[train_level:test_level, -1].astype('int')
    encoder = LabelBinarizer()
    y_dev = encoder.fit_transform(y_dev)
    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    x_test = train_dataset[test_level:, 0:-1].astype('float')
    y_test = train_dataset[test_level:, -1].astype('int')
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


if __name__ == '__main__':
    # load_data1(data_path='../data/')
    load_data2(data_path='../data/model2_data/iris_2_data.csv')