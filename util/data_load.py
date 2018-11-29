# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras import backend as K
import codecs
from make_predict import generate_result
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

    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

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
    X_test, y_test, num_classes = pickle.load(open(data_path, 'rb'))
    print('X test shape:', X_test.shape)
    print('y test shape:', y_test.shape)
    return X_test, y_test


# load the total data from mnist
def load_all_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    print('X train shape:', X_train.shape)
    print('y train shape:', y_train.shape)

    print('X test shape:', X_test.shape)
    print('y test shape:', y_test.shape)

    print('y test classes:', num_classes)

    return X_train, y_train, X_test, y_test, num_classes


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
def generate_model2_data(model_name, datafile, model_file, testfile, result_path):
    # generate_result(model_name='BiLSTM_Attention',
    #                 datafile="./modfile/data.pkl",
    #                 modle_file="BiLSTM_Attention_fold_",
    #                 testfile='./data/mix_data_test_data.json')
    y1_test = generate_result(model_name=model_name, datafile=datafile + "0.pkl", model_file=model_file + "0.pkl",
                              testfile=testfile)
    y2_test = generate_result(model_name=model_name, datafile=datafile + "1.pkl", model_file=model_file + "1.pkl",
                              testfile=testfile)
    y3_test = generate_result(model_name=model_name, datafile=datafile + "2.pkl", model_file=model_file + "2.pkl",
                              testfile=testfile)
    y4_test = generate_result(model_name=model_name, datafile=datafile + "3.pkl", model_file=model_file + "3.pkl",
                              testfile=testfile)
    y5_test = generate_result(model_name=model_name, datafile=datafile + "4.pkl", model_file=model_file + "4.pkl",
                              testfile=testfile)
    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    y_test = []
    for num, line in enumerate(lines):
        item = json.loads(line.rstrip('\n'))
        label = item['label']
        y_test.append(label)
    z_data = np.c_[y1_test, y2_test, y3_test, y4_test, y5_test, np.array(y_test)]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'test']
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


if __name__ == '__main__':
    # load_data1(data_path='../data/')
    # load_data2(data_path='../data/model2_data/iris_2_data.csv')
    load_all_data()