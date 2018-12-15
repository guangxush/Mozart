# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_data2, load_data3
from util.data_load import make_err_dataset
from util import data_process
from model.model2 import mlp2
from util.data_load import generate_imdb_model2_data
from util.util import cal_err_ratio
import numpy as np
from model_use import model_use
from model.model1 import lstm_attention_model


# train model1
def model1(i):
    results_flag = True
    if i > 10:
        i = i % 10
    model2_file = './modfile/model2file/imdb.mlp.best_model.h5'
    result_file = './data/err_data/imdb_'+str(i)+'.data'
    data2_path = './data/model2_data/imdb_'+str(i)+'_data.csv'
    pos_file = "./data/part_data/train_pos_" + str(i) + ".txt"
    neg_file = "./data/part_data/train_neg_" + str(i) + ".txt"
    # train model1
    monitor = 'val_acc'
    filepath = "./modfile/model1file/lstm.best_model_"+str(i)+".h5"
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/imdb_model2_mlp_' + str(i) + '.log')
    Xtrain, Xtest, ytrain, ytest, sourcevocabsize = data_process.get_imdb_part_data(pos_file=pos_file, neg_file=neg_file)
    model = lstm_attention_model(input_dim=800, sourcevocabsize=sourcevocabsize, output_dim=1)
    model.fit(Xtrain, ytrain, batch_size=32, epochs=50, validation_data=(Xtest, ytest), verbose=1, shuffle=True,
              callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate model2 dataset ...')
        result_path = './data/model2_data/imdb_' + str(i) + '_data.csv'
        model_file = './modfile/model1file/lstm.best_model_'
        test_pos_file = './data/part_data/test_pos_0.txt'
        test_neg_file = './data/part_data/test_neg_0.txt'
        generate_imdb_model2_data(model_file=model_file, result_path=result_path, test_pos_file=test_pos_file,
                                  test_neg_file=test_neg_file, count=10, sourcevocabsize=sourcevocabsize)
        print('Load result ...')

        X_test, Y_test = load_data3(data_path=data2_path)
        mlp2_model = mlp2(sample_dim=X_test.shape[1], class_count=2)
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
    data_path = './data/model2_data/imdb_'+str(i)+'_data.csv'
    filepath = "./modfile/model2file/imdb.mlp.best_model.h5"
    print('***** Start Model2 Train *****')
    print('Loading data ...')
    x_train, y_train, x_test, y_test = load_data2(data_path=data_path)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/imdb_model2_mlp_'+str(i)+'.log')
    mlp_model2 = mlp2(sample_dim=x_train.shape[1], class_count=2)
    mlp_model2.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_data=(x_test, y_test),
                   callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model2.load_weights(filepath=filepath)
        results = mlp_model2.predict(x_test)
        label = np.argmax(results, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(y_test)
        # make_err_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)
    print('***** End Model2 Train *****')


if __name__ == '__main__':
    model2(0)
    for i in range(1, 10):
        print('***** ' + str(i) + ' START! *****')
        model1(i)
        model2(i)
        model_use(i)
        print('***** '+str(i)+' FINISHED! *****')
