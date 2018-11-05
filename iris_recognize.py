# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_data1, make_err_dataset, load_data2
from model.model1 import mlp1
from model.model2 import mlp2
import numpy as np


def model1():
    results_flag = True
    train_file = './data/iris_5_data.csv'
    test_file = './data/iris_1_data.csv'
    model_file = './modfile/mlp4.best_model.h5'
    result_file = './err_data/iris_5_data.csv'
    print('***** Start Model1 Train *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data1(train_file=train_file,
                                                                test_file=test_file)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=model_file, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    csv_logger = CSVLogger('logs/mlp.log')
    mlp_model = mlp1(sample_dim=x_train.shape[1], class_count=3)
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model.load_weights(filepath=model_file)
        results = mlp_model.predict(x_test)
        label = np.argmax(results, axis=1)
        print(label)
        print(y_test)
        make_err_dataset(result_path=result_file, label=label, x_test=x_test, y_test=y_test)

    print('***** End Model1 Train *****')


def model2():
    results_flag = True
    data_path = './model2_data/iris_3_data.csv'
    filepath = "./model2file/mlp.best_model.h5"
    print('***** Start Model2 Train *****')
    print('Loading data ...')
    x_train, y_train, x_test, y_test = load_data2(data_path=data_path)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/mlp2.log')
    mlp_model = mlp2(sample_dim=x_train.shape[1], class_count=3)
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1, shuffle=True, validation_split=0.2,
                  callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model.load_weights(filepath=filepath)
        results = mlp_model.predict(x_test)
        label = np.argmax(results, axis=1)
        print(label)
        print(np.argmax(y_test, axis=1))
        # make_err_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)

    print('***** End Model2 Train *****')


if __name__ == '__main__':
    # model1()
    model2()
