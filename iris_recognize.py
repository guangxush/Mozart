# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_data1, load_data2, load_data3
from util.data_process import make_err_dataset
from model.model1 import mlp1
from model.model2 import mlp2
from util.data_process import generate_model2_data
from util.util import cal_err_ratio
import numpy as np
from model_use import model_use


def model1(i):
    results_flag = True
    if i > 5:
        i = i % 5
    train_file = './data/model1_data/iris_'+str(i)+'_data.csv'
    j = i+1
    if j > 5:
        j = j % 5
    test_file = './data/model1_data/iris_'+str(j)+'_data.csv'
    model1_file = './modfile/model1file/mlp'+str(i)+'.best_model.h5'
    model2_file = './modfile/model2file/mlp.best_model.h5'
    result_file = './data/err_data/iris_'+str(i)+'_data.csv'
    data2_path = './data/model2_data/iris_'+str(i)+'_data.csv'
    print('***** Start Model1 Train *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data1(train_file=train_file,
                                                                test_file=test_file)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=model1_file, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    csv_logger = CSVLogger('logs/mlp.log')
    mlp1_model = mlp1(sample_dim=x_train.shape[1], class_count=3)
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    '''
    mlp1_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
                   callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate model2 dataset ...')
        generate_model2_data(data_path=test_file, result_path=data2_path)
        print('Load result ...')
        X_test, Y_test = load_data3(data_path=data2_path)
        print(X_test)
        print(Y_test)
        mlp2_model = mlp2(sample_dim=X_test.shape[1], class_count=3)
        mlp2_model.load_weights(filepath=model2_file)
        results = mlp2_model.predict(X_test)
        label = np.argmax(results, axis=1)
        y_label = np.argmax(Y_test, axis=1)
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(y_label)
        make_err_dataset(result_path=result_file, label=label, x_test=x_test, y_test=y_label)
        cal_err_ratio(file_name='train', label=label, y_test=y_label)
    print('***** End Model1 Train *****')


def model2(i):
    results_flag = True
    data_path = './data/model2_data/iris_'+str(i)+'_data.csv'
    filepath = "./modfile/model2file/mlp.best_model.h5"
    print('***** Start Model2 Train *****')
    print('Loading data ...')
    x_train, y_train, x_test, y_test = load_data2(data_path=data_path)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/mlp2.log')
    mlp_model2 = mlp2(sample_dim=x_train.shape[1], class_count=3)
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    '''
    mlp_model2.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1, shuffle=True, validation_split=0.2,
                   callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model2.load_weights(filepath=filepath)
        results = mlp_model2.predict(x_test)
        label = np.argmax(results, axis=1)
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(np.argmax(y_test, axis=1))
        # make_err_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)
    print('***** End Model2 Train *****')


if __name__ == '__main__':
    model2(0)
    for i in range(1, 6):
        print('***** ' + str(i) + ' START! *****')
        model1(i)
        model2(i)
        model_use(i)
        print('***** '+str(i)+' FINISHED! *****')
