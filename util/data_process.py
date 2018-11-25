# -*- encoding:utf-8 -*-
import os
import pandas as pd
import numpy as np
from model.model1 import cnn
from data_load import load_testset
import pickle, codecs
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)


# split the raw data to 6 files in model1 or test floders
def split_raw_train_data(out_path):
    # load data
    (X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()

    # shuffle data
    shuffle_index = np.random.permutation(60000)
    X_train_all = X_train_all[shuffle_index]
    y_train_all = y_train_all[shuffle_index]

    # reshape to be [samples][pixels][width][height]
    X_train_all = X_train_all.reshape(X_train_all.shape[0], 1, 28, 28).astype('float32')
    # X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    X_train_all = X_train_all / 255
    # X_test = X_test / 255

    # one hot encode outputs
    y_train_all = np_utils.to_categorical(y_train_all)
    # y_test = np_utils.to_categorical(y_test)
    num_classes = y_train_all.shape[1]
    total_num = 6000
    partition = 6
    group_count = int(total_num/partition)
    for i in range(partition):
        line = i * group_count
        X_train = X_train_all[line:line+group_count]
        print(str(i)+": "+str(line)+":"+str(line+group_count))
        y_train = y_train_all[line:line+group_count]
        num_classes = num_classes
        out = codecs.open("../data/"+out_path+"/mnist_"+str(i+1)+".data", 'wb')
        pickle.dump([X_train, y_train, num_classes], out, 0)
        out.close()


# split the raw data to 6 files in model1 or test floders
def split_raw_test_data(out_path):
    # load data
    (X_train, y_train), (X_test_all, y_test_all) = mnist.load_data()

    # shuffle data
    shuffle_index = np.random.permutation(10000)
    X_test_all = X_test_all[shuffle_index]
    y_test_all = y_test_all[shuffle_index]

    # reshape to be [samples][pixels][width][height]
    X_test_all = X_test_all.reshape(X_test_all.shape[0], 1, 28, 28).astype('float32')

    X_test_all = X_test_all / 255

    # one hot encode outputs
    y_test_all = np_utils.to_categorical(y_test_all)
    # y_test = np_utils.to_categorical(y_test)
    num_classes = y_test_all.shape[1]
    total_num = 6000
    partition = 6
    group_count = int(total_num/partition)
    for i in range(partition):
        line = i * group_count
        X_test = X_test_all[line:line+group_count]
        print(str(i) + ": " + str(line) + ":" + str(line + group_count))
        y_test = y_test_all[line:line+group_count]
        num_classes = num_classes
        out = codecs.open("../data/"+out_path+"/mnist_"+str(i+1)+".data", 'wb')
        pickle.dump([X_test, y_test, num_classes], out, 0)
        out.close()


# get the total dataset
def get_total_data(out_path):
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
    out = codecs.open("../data/" + out_path + "/mnist.data", 'wb')
    pickle.dump([X_train, y_train, X_test, y_test, num_classes], out, 0)
    out.close()


# save the error dataset depends on the model result and the truth label
def make_err_dataset(result_path, label, x_test, y_test):
    count = 0
    x_test = x_test.astype('int')
    y_test = y_test.astype('int')
    err_data_list = []
    for i in label:
        if i != y_test[count]:
            err_result = np.append(x_test[count], y_test[count]).tolist()
            err_data_list.append(err_result)
        count += 1
    err_data = pd.DataFrame(err_data_list)
    err_data.to_csv(result_path, encoding='utf-8', header=1, index=0)


# generate the model labels from model1 result
def generate_model2_label(file_name, cnn_model, x_test, line_count):
    if not os.path.exists(file_name):
        print(file_name)
        print("file not found!")
        # if file not exists, return [0]*30
        return np.array([0] * line_count)
    cnn_model.load_weights(file_name)
    results = cnn_model.predict(x_test)
    label = np.argmax(results, axis=1)
    # print(label)
    return label
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


# generate model2 dataset by load model1 to predict temp result
def generate_model2_data(data_path, result_path, num_classes):
    x_test, y_test = load_testset(data_path=data_path)
    y_test = np.argmax(y_test, axis=1)
    cnn_model = cnn(num_classes=num_classes)
    line_count = 1000
    y1_test = np.array(generate_model2_label(file_name='./modfile/model1file/cnn1.best_model.h5', cnn_model=cnn_model,
                                             x_test=x_test, line_count=line_count).tolist())
    y2_test = np.array(generate_model2_label(file_name='./modfile/model1file/cnn2.best_model.h5', cnn_model=cnn_model,
                                             x_test=x_test, line_count=line_count)).tolist()
    y3_test = np.array(generate_model2_label(file_name='./modfile/model1file/cnn3.best_model.h5', cnn_model=cnn_model,
                                             x_test=x_test, line_count=line_count)).tolist()
    y4_test = np.array(generate_model2_label(file_name='./modfile/model1file/cnn4.best_model.h5', cnn_model=cnn_model,
                                             x_test=x_test, line_count=line_count)).tolist()
    y5_test = np.array(generate_model2_label(file_name='./modfile/model1file/cnn5.best_model.h5', cnn_model=cnn_model,
                                             x_test=x_test, line_count=line_count)).tolist()
    z_data = np.c_[y1_test, y2_test, y3_test, y4_test, y5_test, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'test']
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


if __name__ == '__main__':
    split_raw_train_data(out_path="model1_data")
    split_raw_test_data(out_path="test_data")
    # get_total_data(out_path='total_data')
    # generate_model2_data(data_path='../data/model1_data/iris_3_data.csv',
    #                      result_path='../data/model2_data/iris_3_data.csv')