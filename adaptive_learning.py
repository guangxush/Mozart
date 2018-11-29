# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_data2, load_data3
from util.data_load import make_err_dataset
from util import data_process
from model.model2 import mlp2
from util.data_load import generate_model2_data
from util.util import cal_err_ratio
import numpy as np
from model_use import model_use
import os
from sentiment_analysis import train_e2e_model


# train model1
def model1(i):
    results_flag = True
    if i > 6:
        i = i % 6
    # train_file = './data/model1_data/news_'+str(i)+'.data'
    j = i+1
    if j > 6:
        j = j % 6
    # test_file = './data/model1_data/mnist_'+str(j)+'.data'
    # model1_file = './modfile/model1file/cnn'+str(i)+'.best_model.h5'
    model2_file = './modfile/model2file/mlp.best_model.h5'
    result_file = './data/err_data/news_'+str(i)+'.data'
    data2_path = './data/model2_data/news_'+str(i)+'_data.csv'
    # print('***** Start Model1 Train *****')
    # print('Loading data ...')
    # X_train, y_train, X_test, y_test, num_classes = load_data1(trainfile=train_file, testfile=test_file)
    #
    # print('Training CNN model ...')
    # monitor = 'val_acc'
    # check_pointer = ModelCheckpoint(filepath=model1_file, monitor=monitor, verbose=1,
    #                                 save_best_only=True, save_weights_only=True)
    # early_stopping = EarlyStopping(patience=5)
    # csv_logger = CSVLogger('logs/model1_cnn_'+str(i)+'.log')
    # cnn_model = cnn(num_classes=num_classes)
    # cnn_model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_split=0.3,
    #               callbacks=[check_pointer, early_stopping, csv_logger])

    # train model1
    modelname = 'BiLSTM_Attention'
    datafile = "./modfile/data" + "_fold_" + str(i) + ".pkl"
    modelfile = modelname + "_fold_" + str(i) + ".pkl"

    trainfile = "./data/mix_data_train_data.json"
    testfile = "./data/mix_data_test_data.json"
    w2v_file = "./modfile/Word2Vec.mod"
    char2v_file = "./modfile/Char2Vec.mod"
    resultdir = "./result/"
    print(modelname)

    maxlen = 100
    batch_size = 128
    npochos = 100

    if not os.path.exists(datafile):
        print("Precess data " + str(i) + "....")
        data_process.get_part_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100,
                                   maxlen=maxlen, left=i)

    if not os.path.exists("./modfile/" + modelfile):
        print("data has existed: " + datafile)
        print("Training EE " + str(i) + " model....")
        train_e2e_model(modelname, datafile, modelfile, resultdir,
                        npochos=npochos, batch_size=batch_size, retrain=False)

    if results_flag:
        print('Generate model2 dataset ...')
        data_file = "./modfile/data_fold_"
        result_path = './data/test_model2_data/news_' + str(i) + '.data'
        model_name = 'BiLSTM_Attention'
        modle_file = "BiLSTM_Attention_fold_"
        testfile = './data/mix_data_test_data.json'
        filepath = "./modfile/model2file/mlp.best_model.h5"
        batch_size = 128
        generate_model2_data(model_name=model_name, datafile=data_file, model_file=modle_file, testfile=testfile,
                             result_path=result_path, batch_size=batch_size)
        print('Load result ...')

        # x_test, y_test = load_data3(data_path=result_path)
        # model2 = mlp2(sample_dim=x_test.shape[1], class_count=10)
        # model2.load_weights(filepath)
        # results = model2.predict(x_test)
        # label = np.argmax(results, axis=1).astype('int')
        # print("pred:")
        # print(label)
        # print("true:")
        # print(y_test)
        # cal_err_ratio(file_name='test', label=label, y_test=y_test)

        X_test, Y_test = load_data3(data_path=data2_path)
        mlp2_model = mlp2(sample_dim=X_test.shape[1], class_count=1)
        mlp2_model.load_weights(filepath=model2_file)
        results = mlp2_model.predict(X_test)
        label = np.argmax(results, axis=1)
        y_label = Y_test
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(y_label)
        make_err_dataset(result_path=result_file, label=label, x_test=X_test, y_test=y_label)
        cal_err_ratio(file_name='train', label=label, y_test=y_label)
    print('***** End Model1 Train *****')


# train model2
def model2(i):
    results_flag = True
    data_path = './data/model2_data/news_'+str(i)+'_data.csv'
    filepath = "./modfile/model2file/mlp.best_model.h5"
    print('***** Start Model2 Train *****')
    print('Loading data ...')
    x_train, y_train, x_test, y_test = load_data2(data_path=data_path)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/model2_mlp_'+str(i)+'.log')
    mlp_model2 = mlp2(sample_dim=x_train.shape[1], class_count=1)
    mlp_model2.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_split=0.2,
                   callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model2.load_weights(filepath=filepath)
        results = mlp_model2.predict(x_test)
        label = np.argmax(results, axis=1)
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(y_test)
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
